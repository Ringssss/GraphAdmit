#!/usr/bin/env python3
import argparse
import json
import os
import socket
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing.spawn import ProcessRaisedException


def sizes_from_qwentrace(path, limit, scale):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    lens = [int(r['actual_input_length']) for r in data['requests'][:limit]]
    return [max(1024, x * scale) for x in lens]


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def worker(rank, world_size, args, sizes, queue):
    torch.cuda.set_device(rank)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(args.port)
    dist.init_process_group('nccl', rank=rank, world_size=world_size, timeout=timedelta(seconds=args.timeout_s), device_id=torch.device(f'cuda:{rank}'))
    device = torch.device(f'cuda:{rank}')
    dynamic_bufs = {s: torch.ones(s, dtype=torch.float32, device=device) for s in sorted(set(sizes))}
    dist.barrier()
    for _ in range(args.warmups):
        for s in sizes[: min(len(sizes), 4)]:
            dist.all_reduce(dynamic_bufs[s])
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for s in sizes:
        dist.all_reduce(dynamic_bufs[s])
    torch.cuda.synchronize()
    dist.barrier()
    eager_s = time.perf_counter() - t0

    if args.eager_only:
        dist.destroy_process_group()
        if rank == 0:
            queue.put({
                'world_size': world_size,
                'num_ops': len(sizes),
                'unique_sizes': len(set(sizes)),
                'capture_sizes': [],
                'eager_s': eager_s,
                'graph_s': None,
                'speedup': None,
                'padding_waste_pct': None,
                'mode': 'eager_only',
            })
        return

    graphs = {}
    static_inputs = {}
    if args.fixed_size_only:
        capture_sizes = [max(sizes)]
    else:
        capture_sizes = sorted(set(max(1024, ((s + args.bucket - 1)//args.bucket)*args.bucket) for s in sizes))
    for cap in capture_sizes:
        buf = torch.ones(cap, dtype=torch.float32, device=device)
        static_inputs[cap] = buf
        dist.barrier()
        for _ in range(args.warmups):
            dist.all_reduce(buf)
        torch.cuda.synchronize()
        dist.barrier()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            dist.all_reduce(buf)
        torch.cuda.synchronize()
        dist.barrier()
        graphs[cap] = g
    torch.cuda.synchronize()
    dist.barrier()
    t1 = time.perf_counter()
    for s in sizes:
        cap = min(c for c in capture_sizes if c >= s)
        static_inputs[cap][:s].fill_(1.0)
        torch.cuda.synchronize()
        dist.barrier()
        graphs[cap].replay()
        torch.cuda.synchronize()
        dist.barrier()
    torch.cuda.synchronize()
    graph_s = time.perf_counter() - t1
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        queue.put({
            'world_size': world_size,
            'num_ops': len(sizes),
            'unique_sizes': len(set(sizes)),
            'capture_sizes': capture_sizes,
            'eager_s': eager_s,
            'graph_s': graph_s,
            'speedup': eager_s / graph_s if graph_s > 0 else None,
            'padding_waste_pct': (sum(min(c for c in capture_sizes if c >= s) - s for s in sizes) / sum(sizes)) * 100,
            'mode': 'fixed_size_graph' if args.fixed_size_only else 'bucketed_graph',
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workload', default='results/qwentrace_morspec_qwen_16_4096.json')
    parser.add_argument('--limit', type=int, default=16)
    parser.add_argument('--scale', type=int, default=4096)
    parser.add_argument('--bucket', type=int, default=262144)
    parser.add_argument('--world-size', type=int, default=2)
    parser.add_argument('--warmups', type=int, default=3)
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--timeout-s', type=int, default=30)
    parser.add_argument('--eager-only', action='store_true')
    parser.add_argument('--fixed-size-only', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.port == 0:
        args.port = find_free_port()
    sizes = sizes_from_qwentrace(args.workload, args.limit, args.scale)
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    ctx_spawn = mp.spawn(worker, args=(args.world_size, args, sizes, queue), nprocs=args.world_size, join=False)
    deadline = time.time() + args.timeout_s
    result = None
    while time.time() < deadline:
        if not queue.empty():
            result = queue.get()
            break
        try:
            if ctx_spawn.join(timeout=1.0):
                break
        except ProcessRaisedException as exc:
            result = {
                'world_size': args.world_size,
                'num_ops': len(sizes),
                'unique_sizes': len(set(sizes)),
                'capture_sizes': [],
                'eager_s': None,
                'graph_s': None,
                'speedup': None,
                'padding_waste_pct': None,
                'mode': 'worker_error',
                'error': str(exc),
            }
            break
    if result is None:
        for proc in ctx_spawn.processes:
            if proc.is_alive():
                proc.terminate()
        for proc in ctx_spawn.processes:
            proc.join(timeout=2.0)
        result = {
            'world_size': args.world_size,
            'num_ops': len(sizes),
            'unique_sizes': len(set(sizes)),
            'capture_sizes': [],
            'eager_s': None,
            'graph_s': None,
            'speedup': None,
            'padding_waste_pct': None,
            'mode': 'timeout',
            'timeout_s': args.timeout_s,
        }
    result['workload'] = args.workload
    result['sizes'] = sizes
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
