#!/usr/bin/env python3
import argparse, json, sys, time
from pathlib import Path
from urllib import request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from benchmarks.morspec_loader import load_morspec_prompts


def post_json(url, payload):
    data = json.dumps(payload).encode('utf-8')
    req = request.Request(url, data=data, headers={'Content-Type':'application/json'}, method='POST')
    with request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode('utf-8'))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset', default='/home/zhujianian/morspec/data/gsm8k.jsonl')
    ap.add_argument('--num-samples', type=int, default=8)
    ap.add_argument('--base-url', default='http://127.0.0.1:18080/v1')
    ap.add_argument('--model', default='/mnt/models/Meta-Llama-3-8B-Instruct')
    ap.add_argument('--output', default='results/vllm_morspec_server_client.json')
    args=ap.parse_args()
    prompts,_=load_morspec_prompts(args.dataset,args.num_samples)
    # wait for server
    for _ in range(120):
        try:
            request.urlopen(args.base_url+'/models', timeout=2).read(); break
        except Exception:
            time.sleep(1)
    rows=[]
    for i,p in enumerate(prompts):
        payload={'model': args.model, 'prompt': p, 'max_tokens': 1, 'temperature': 0.0}
        t=time.perf_counter(); resp=post_json(args.base_url+'/completions', payload); dt=time.perf_counter()-t
        rows.append({'idx':i,'seconds':dt,'usage':resp.get('usage',{})})
        print(f'[{i+1}/{len(prompts)}] {dt*1000:.2f} ms usage={resp.get("usage",{})}')
    out={'dataset':args.dataset,'num_samples':len(prompts),'rows':rows,'avg_ms':sum(r['seconds'] for r in rows)*1000/len(rows)}
    Path(args.output).parent.mkdir(exist_ok=True,parents=True)
    Path(args.output).write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({'avg_ms':out['avg_ms'],'num_samples':out['num_samples']},indent=2))

if __name__=='__main__': main()
