#!/usr/bin/env python3
"""Run each framework sequentially; validate matching configurations before reporting ratios."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time

p=argparse.ArgumentParser()
p.add_argument('--gradle', default='./gradlew')
p.add_argument('--python', default=os.sys.executable)
p.add_argument('--device', choices=['CPU','CUDA'], default='CPU')
p.add_argument('--precision', choices=['FP32','FP16_COMPUTE'], default='FP32')
p.add_argument('--moe', action='store_true')
p.add_argument('--output', required=True)
for key, default in [('sequence',16),('vocabulary',32),('hidden',32),('layers',2),('heads',4),('kv_heads',2),('ffn',64),('warmup',5),('repeats',9),('decode_tokens',8)]:p.add_argument('--'+key.replace('_','-'),type=int,default=default)
a=p.parse_args();root=Path(__file__).resolve().parents[1]
if a.device=='CPU' and a.precision!='FP32':raise ValueError('CPU comparison requires FP32')

def execute(cmd):
    samples=[];stop=threading.Event()
    def monitor():
        while not stop.is_set():
            value=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits','-i','0'],text=True,capture_output=True)
            if value.returncode==0:
                try:samples.append([float(v) for v in value.stdout.strip().split(',')])
                except ValueError:pass
            stop.wait(.1)
    thread=None
    if a.device=='CUDA':
        subprocess.run(['nvidia-smi'],check=True,capture_output=True)
        thread=threading.Thread(target=monitor,daemon=True);thread.start()
    try:r=subprocess.run(cmd,cwd=root,text=True,capture_output=True)
    finally:
        stop.set()
        if thread:thread.join()
    if r.returncode:raise RuntimeError('Command failed:\n'+' '.join(cmd)+'\n'+r.stdout+'\n'+r.stderr)
    lines=[line for line in r.stdout.splitlines() if line.startswith('{"framework"')]
    if len(lines)!=1:raise RuntimeError('Expected one benchmark JSON record:\n'+r.stdout)
    result=json.loads(lines[0])
    result['sampled_board_utilization_percent_mean']=statistics.mean(v[0] for v in samples) if samples else None
    result['sampled_peak_board_memory_mib']=max(v[1] for v in samples) if samples else None
    return result

values=[a.device,str(a.moe).lower(),a.sequence,a.vocabulary,a.hidden,a.layers,a.heads,a.kv_heads,a.ffn,a.warmup,a.repeats,a.decode_tokens,a.precision]
synapse=execute([a.gradle,'--console=plain','transformerBenchmark','--args='+' '.join(map(str,values))])
cmd=[a.python,str(root/'benchmarks/pytorch_transformer.py'),'--device',a.device.lower(),'--precision',a.precision]
for key in ['sequence','vocabulary','hidden','layers','heads','kv_heads','ffn','warmup','repeats','decode_tokens']:cmd.extend(['--'+key.replace('_','-'),str(getattr(a,key))])
if a.moe:cmd.append('--moe')
pytorch=execute(cmd)
for key in ['device','precision','moe','sequence','vocabulary','hidden','layers','heads','kv_heads','ffn','batch','decode_tokens','warmup','repeats','parameters']:
    if synapse[key]!=pytorch[key]:raise RuntimeError(f'Mismatched {key}: {synapse[key]} vs {pytorch[key]}')
ratios={key:synapse[key]/pytorch[key] for key in ['prefill_tokens_per_second','training_tokens_per_second','decode_tokens_per_second']}
code_files=sorted([*root.glob('src/**/*.java'),*root.glob('src/**/*.cu'),*root.glob('benchmarks/*.py'),root/'build.gradle'])
source_hash=hashlib.sha256()
for code_file in code_files:source_hash.update(str(code_file.relative_to(root)).encode()+b'\0'+code_file.read_bytes())
result=dict(source_sha256=source_hash.hexdigest(),working_tree_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()),timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),methodology='Median synchronized wall-clock timings, sequential frameworks, batch=1, no sampling in decode; board monitoring includes process startup and other GPU processes if present.',synapse=synapse,pytorch=pytorch,synapse_over_pytorch=ratios)
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(ratios,indent=2))
