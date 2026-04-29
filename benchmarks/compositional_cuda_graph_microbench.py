#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenMLP(nn.Module):
    """Token-local block used to isolate graph composition mechanics.

    Because tokens do not communicate, padding, tiling, and adjacent-request
    packing are semantically exact.  Attention/MoE experiments need metadata
    masks/routing guards on top of the same replay mechanisms.
    """

    def __init__(self, hidden: int, inter: int, layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            nn.ModuleDict({
                "norm": nn.LayerNorm(hidden),
                "up": nn.Linear(hidden, inter, bias=False),
                "gate": nn.Linear(hidden, inter, bias=False),
                "down": nn.Linear(inter, hidden, bias=False),
            })
            for _ in range(layers)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            h = layer["norm"](x)
            x = x + layer["down"](F.silu(layer["gate"](h)) * layer["up"](h))
        return x


@dataclass
class StaticGraphTemplate:
    capacity: int
    model: nn.Module
    hidden: int
    device: torch.device
    dtype: torch.dtype
    pool: object

    def __post_init__(self) -> None:
        self.static_input = torch.zeros(
            self.capacity,
            self.hidden,
            device=self.device,
            dtype=self.dtype,
        )
        self.static_output: torch.Tensor | None = None
        self.graph: torch.cuda.CUDAGraph | None = None

    def capture(self, warmups: int) -> float:
        with torch.inference_mode():
            for _ in range(warmups):
                self.static_output = self.model(self.static_input)
        torch.cuda.synchronize()
        start = time.perf_counter()
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode():
            with torch.cuda.graph(graph, pool=self.pool):
                self.static_output = self.model(self.static_input)
        torch.cuda.synchronize()
        self.graph = graph
        return (time.perf_counter() - start) * 1000.0

    def replay(self, x: torch.Tensor, n: int) -> torch.Tensor:
        if self.graph is None or self.static_output is None:
            raise RuntimeError(f"template {self.capacity} was not captured")
        self.static_input.zero_()
        self.static_input[:n].copy_(x[:n])
        self.graph.replay()
        return self.static_output[:n].clone()


class TemplateBank:
    def __init__(
        self,
        model: nn.Module,
        capacities: list[int],
        *,
        hidden: int,
        device: torch.device,
        dtype: torch.dtype,
        warmups: int,
    ):
        self.model = model
        self.hidden = hidden
        self.device = device
        self.dtype = dtype
        self.pool = torch.cuda.graph_pool_handle()
        self.templates: dict[int, StaticGraphTemplate] = {}
        self.capture_ms: dict[int, float] = {}
        for capacity in sorted(set(capacities), reverse=True):
            template = StaticGraphTemplate(
                capacity=capacity,
                model=model,
                hidden=hidden,
                device=device,
                dtype=dtype,
                pool=self.pool,
            )
            self.capture_ms[capacity] = template.capture(warmups)
            self.templates[capacity] = template

    def smallest_ge(self, n: int, capacities: list[int] | None = None) -> int | None:
        choices = capacities or list(self.templates)
        for capacity in sorted(choices):
            if n <= capacity and capacity in self.templates:
                return capacity
        return None

    def replay(self, capacity: int, x: torch.Tensor, n: int) -> torch.Tensor:
        return self.templates[int(capacity)].replay(x, n)


@dataclass
class PlanStats:
    name: str
    total_ms: float
    avg_ms: float
    speedup_vs_eager: float
    graph_replays: int
    eager_fallbacks: int
    capture_count: int
    capture_ms: float
    padding_tokens: int
    useful_tokens: int
    max_abs_error: float
    correct: bool
    notes: str

    @property
    def padding_waste_pct(self) -> float:
        denom = self.padding_tokens + self.useful_tokens
        if denom <= 0:
            return 0.0
        return 100.0 * self.padding_tokens / denom

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "total_ms": self.total_ms,
            "avg_ms": self.avg_ms,
            "speedup_vs_eager": self.speedup_vs_eager,
            "graph_replays": self.graph_replays,
            "eager_fallbacks": self.eager_fallbacks,
            "capture_count": self.capture_count,
            "capture_ms": self.capture_ms,
            "padding_tokens": self.padding_tokens,
            "useful_tokens": self.useful_tokens,
            "padding_waste_pct": self.padding_waste_pct,
            "max_abs_error": self.max_abs_error,
            "correct": self.correct,
            "notes": self.notes,
        }


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def synchronize() -> None:
    torch.cuda.synchronize()


def time_plan(fn: Callable[[], list[torch.Tensor]], repeat: int) -> tuple[float, list[torch.Tensor]]:
    outputs: list[torch.Tensor] = []
    synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        outputs = fn()
    synchronize()
    total_ms = (time.perf_counter() - start) * 1000.0 / max(1, repeat)
    return total_ms, outputs


def max_error(outputs: list[torch.Tensor], reference: list[torch.Tensor]) -> float:
    err = 0.0
    for out, ref in zip(outputs, reference):
        err = max(err, float((out.float() - ref.float()).abs().max().item()))
    return err


def run_eager(
    model: nn.Module,
    inputs: list[torch.Tensor],
) -> list[torch.Tensor]:
    with torch.inference_mode():
        return [model(x).clone() for x in inputs]


def run_exact_only(
    model: nn.Module,
    bank: TemplateBank,
    inputs: list[torch.Tensor],
    exact_sizes: set[int],
    counters: dict[str, int],
) -> list[torch.Tensor]:
    outputs = []
    with torch.inference_mode():
        for x in inputs:
            n = int(x.shape[0])
            if n in exact_sizes:
                outputs.append(bank.replay(n, x, n))
                counters["graph_replays"] += 1
                counters["useful_tokens"] += n
            else:
                outputs.append(model(x).clone())
                counters["eager_fallbacks"] += 1
                counters["useful_tokens"] += n
    return outputs


def run_pad(
    bank: TemplateBank,
    inputs: list[torch.Tensor],
    capacities: list[int],
    counters: dict[str, int],
) -> list[torch.Tensor]:
    outputs = []
    for x in inputs:
        n = int(x.shape[0])
        cap = bank.smallest_ge(n, capacities)
        if cap is None:
            raise ValueError(f"no padding capacity for n={n}")
        outputs.append(bank.replay(cap, x, n))
        counters["graph_replays"] += 1
        counters["useful_tokens"] += n
        counters["padding_tokens"] += cap - n
    return outputs


def run_tile(
    bank: TemplateBank,
    inputs: list[torch.Tensor],
    tile_size: int,
    counters: dict[str, int],
) -> list[torch.Tensor]:
    outputs = []
    for x in inputs:
        chunks = []
        n = int(x.shape[0])
        for start in range(0, n, tile_size):
            end = min(start + tile_size, n)
            chunk_n = end - start
            chunks.append(bank.replay(tile_size, x[start:end], chunk_n))
            counters["graph_replays"] += 1
            counters["useful_tokens"] += chunk_n
            counters["padding_tokens"] += tile_size - chunk_n
        outputs.append(torch.cat(chunks, dim=0))
    return outputs


def run_pack(
    bank: TemplateBank,
    inputs: list[torch.Tensor],
    capacities: list[int],
    counters: dict[str, int],
) -> list[torch.Tensor]:
    outputs: list[torch.Tensor | None] = [None] * len(inputs)
    i = 0
    while i < len(inputs):
        packed = [i]
        total = int(inputs[i].shape[0])
        j = i + 1
        best_cap = bank.smallest_ge(total, capacities)
        while j < len(inputs):
            candidate_total = total + int(inputs[j].shape[0])
            candidate_cap = bank.smallest_ge(candidate_total, capacities)
            if candidate_cap is None:
                break
            if best_cap is None or candidate_cap <= best_cap or candidate_cap - candidate_total <= best_cap - total:
                packed.append(j)
                total = candidate_total
                best_cap = candidate_cap
                j += 1
            else:
                break
        if best_cap is None:
            raise ValueError(f"no packing capacity for total={total}")
        merged = torch.cat([inputs[k] for k in packed], dim=0)
        merged_out = bank.replay(best_cap, merged, total)
        counters["graph_replays"] += 1
        counters["useful_tokens"] += total
        counters["padding_tokens"] += best_cap - total
        offset = 0
        for k in packed:
            n = int(inputs[k].shape[0])
            outputs[k] = merged_out[offset:offset + n].clone()
            offset += n
        i = j
    return [out for out in outputs if out is not None]


def make_residual_runner(
    model: nn.Module,
    bank: TemplateBank,
    inputs: list[torch.Tensor],
    *,
    initial_exact_sizes: set[int],
    residual_capacity: int,
    min_samples: int,
    counters: dict[str, int],
) -> Callable[[], list[torch.Tensor]]:
    seen: dict[int, int] = {}
    admitted: set[int] = set(initial_exact_sizes)

    def run() -> list[torch.Tensor]:
        outputs = []
        with torch.inference_mode():
            for x in inputs:
                n = int(x.shape[0])
                if n in admitted and n <= residual_capacity:
                    cap = n if n in bank.templates else residual_capacity
                    outputs.append(bank.replay(cap, x, n))
                    counters["graph_replays"] += 1
                    counters["useful_tokens"] += n
                    counters["padding_tokens"] += max(0, cap - n)
                    continue
                seen[n] = seen.get(n, 0) + 1
                if n not in admitted and seen[n] >= min_samples and n <= residual_capacity:
                    admitted.add(n)
                    counters["admitted_shapes"] += 1
                    cap = n if n in bank.templates else residual_capacity
                    outputs.append(bank.replay(cap, x, n))
                    counters["graph_replays"] += 1
                    counters["useful_tokens"] += n
                    counters["padding_tokens"] += max(0, cap - n)
                else:
                    outputs.append(model(x).clone())
                    counters["eager_fallbacks"] += 1
                    counters["useful_tokens"] += n
        return outputs

    return run


def make_counters() -> dict[str, int]:
    return {
        "graph_replays": 0,
        "eager_fallbacks": 0,
        "padding_tokens": 0,
        "useful_tokens": 0,
        "admitted_shapes": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/compositional_cuda_graph_microbench.json")
    parser.add_argument("--workload", default="55,41,55,41,60,36,55,41,80,16,55,41,90,6,55,41,64,32,55,41")
    parser.add_argument("--exact-sizes", default="32,64")
    parser.add_argument("--pad-sizes", default="32,64,96,128")
    parser.add_argument("--pack-sizes", default="64,96,128")
    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--inter", type=int, default=8192)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--residual-min-samples", type=int, default=2)
    parser.add_argument("--residual-capacity", type=int, default=96)
    parser.add_argument("--atol", type=float, default=5e-2)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for CUDA Graph microbench")

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    device = torch.device("cuda")
    torch.manual_seed(20260429)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True

    lengths = parse_int_list(args.workload)
    exact_sizes = set(parse_int_list(args.exact_sizes))
    pad_sizes = parse_int_list(args.pad_sizes)
    pack_sizes = parse_int_list(args.pack_sizes)
    all_capacities = sorted(
        set(exact_sizes)
        | set(pad_sizes)
        | set(pack_sizes)
        | {args.tile_size, args.residual_capacity}
        | {n for n in lengths if n in exact_sizes}
    )

    model = TokenMLP(args.hidden, args.inter, args.layers).to(device=device, dtype=dtype).eval()
    inputs = [
        torch.randn(n, args.hidden, device=device, dtype=dtype)
        for n in lengths
    ]

    bank = TemplateBank(
        model,
        all_capacities,
        hidden=args.hidden,
        device=device,
        dtype=dtype,
        warmups=args.warmups,
    )

    # Warm eager path and build reference.
    for _ in range(args.warmups):
        _ = run_eager(model, inputs)
    eager_ms, reference = time_plan(lambda: run_eager(model, inputs), args.repeat)

    plan_payloads = []
    eager_total_tokens = sum(lengths)

    def add_plan(
        name: str,
        runner: Callable[[], list[torch.Tensor]],
        counters: dict[str, int],
        capture_count: int | Callable[[], int],
        capture_ms: float | Callable[[], float],
        notes: str,
    ) -> None:
        total_ms, outputs = time_plan(runner, args.repeat)
        err = max_error(outputs, reference)
        final_capture_count = capture_count() if callable(capture_count) else capture_count
        final_capture_ms = capture_ms() if callable(capture_ms) else capture_ms
        plan_payloads.append(
            PlanStats(
                name=name,
                total_ms=total_ms,
                avg_ms=total_ms / len(inputs),
                speedup_vs_eager=eager_ms / total_ms if total_ms > 0 else math.nan,
                graph_replays=int(counters["graph_replays"] / max(1, args.repeat)),
                eager_fallbacks=int(counters["eager_fallbacks"] / max(1, args.repeat)),
                capture_count=final_capture_count,
                capture_ms=final_capture_ms,
                padding_tokens=int(counters["padding_tokens"] / max(1, args.repeat)),
                useful_tokens=int(counters["useful_tokens"] / max(1, args.repeat)),
                max_abs_error=err,
                correct=err <= args.atol,
                notes=notes,
            ).to_json()
        )

    plan_payloads.append(
            PlanStats(
                name="eager_dynamic",
                total_ms=eager_ms,
                avg_ms=eager_ms / len(inputs),
                speedup_vs_eager=1.0,
                graph_replays=0,
                eager_fallbacks=len(inputs),
                capture_count=0,
                capture_ms=0.0,
                padding_tokens=0,
                useful_tokens=eager_total_tokens,
                max_abs_error=0.0,
                correct=True,
                notes="Dynamic eager execution; no graph coverage.",
            ).to_json()
    )

    counters = make_counters()
    exact_capture_ms = sum(bank.capture_ms.get(size, 0.0) for size in exact_sizes)
    add_plan(
        "exact_only_graph_32_64",
        lambda: run_exact_only(model, bank, inputs, exact_sizes, counters),
        counters,
        capture_count=len(exact_sizes),
        capture_ms=exact_capture_ms,
        notes="Simulates exact-shape CG: 55 and 41 miss and fall back.",
    )

    counters = make_counters()
    pad_capture_ms = sum(bank.capture_ms.get(size, 0.0) for size in pad_sizes)
    add_plan(
        "pad_to_next_template",
        lambda: run_pad(bank, inputs, pad_sizes, counters),
        counters,
        capture_count=len(pad_sizes),
        capture_ms=pad_capture_ms,
        notes="Covers all sizes via static capacity; pays padding compute.",
    )

    counters = make_counters()
    tile_capture_ms = bank.capture_ms.get(args.tile_size, 0.0)
    add_plan(
        "tile_32_composition",
        lambda: run_tile(bank, inputs, args.tile_size, counters),
        counters,
        capture_count=1,
        capture_ms=tile_capture_ms,
        notes="Covers arbitrary token counts with one template; launch count grows with ceil(n/tile).",
    )

    counters = make_counters()
    pack_capture_ms = sum(bank.capture_ms.get(size, 0.0) for size in pack_sizes)
    add_plan(
        "pack_adjacent_requests",
        lambda: run_pack(bank, inputs, pack_sizes, counters),
        counters,
        capture_count=len(pack_sizes),
        capture_ms=pack_capture_ms,
        notes="Packs pairs such as 55+41 into one 96-token graph, reducing launches and padding.",
    )

    counters = make_counters()
    residual_runner = make_residual_runner(
        model,
        bank,
        inputs,
        initial_exact_sizes=exact_sizes,
        residual_capacity=args.residual_capacity,
        min_samples=args.residual_min_samples,
        counters=counters,
    )
    add_plan(
        "residual_learn_then_replay",
        residual_runner,
        counters,
        capture_count=lambda: len(exact_sizes) + counters["admitted_shapes"],
        capture_ms=lambda: sum(bank.capture_ms.get(size, 0.0) for size in exact_sizes)
        + counters["admitted_shapes"] * bank.capture_ms.get(args.residual_capacity, 0.0),
        notes="Starts from 32/64, explores repeated misses, then replays admitted residual shapes.",
    )

    output = {
        "workload_lengths": lengths,
        "model": {
            "type": "TokenMLP",
            "hidden": args.hidden,
            "inter": args.inter,
            "layers": args.layers,
            "dtype": args.dtype,
        },
        "templates": {
            "exact_sizes": sorted(exact_sizes),
            "pad_sizes": pad_sizes,
            "pack_sizes": pack_sizes,
            "tile_size": args.tile_size,
            "capture_ms_by_capacity": bank.capture_ms,
        },
        "repeat": args.repeat,
        "plans": plan_payloads,
        "interpretation": [
            "Exact-only graph replay loses coverage on dynamic residual sizes.",
            "Padding recovers coverage when capacity waste is moderate.",
            "Tiling is universal but can lose the sweet spot through extra graph launches.",
            "Adjacent-request packing is the compositional win: 55+41 can share a 96-token graph instead of two misses or two padded launches.",
            "Residual learning matters when misses repeat; it should be evaluated against capture/warmup amortization, not just raw hit rate.",
        ],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(out),
        "plans": [
            {
                "name": row["name"],
                "total_ms": row["total_ms"],
                "speedup_vs_eager": row["speedup_vs_eager"],
                "graph_replays": row["graph_replays"],
                "eager_fallbacks": row["eager_fallbacks"],
                "padding_waste_pct": row["padding_waste_pct"],
                "max_abs_error": row["max_abs_error"],
                "correct": row["correct"],
            }
            for row in plan_payloads
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
