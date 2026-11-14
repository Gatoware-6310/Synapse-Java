"""Equivalent pre-RMSNorm, interleaved RoPE, tied, SwiGLU decoder benchmark (batch=1)."""
import argparse
import json
import platform
import statistics
import time
from contextlib import nullcontext
import torch
from torch import nn
from torch.nn import functional as F

p = argparse.ArgumentParser()
for name, default in [('sequence', 16), ('vocabulary', 32), ('hidden', 32), ('layers', 2), ('heads', 4), ('kv_heads', 2), ('ffn', 64), ('warmup', 3), ('repeats', 7), ('decode_tokens', 8)]:
    p.add_argument('--' + name.replace('_', '-'), type=int, default=default)
p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
p.add_argument('--precision', choices=['FP32', 'FP16_COMPUTE'], default='FP32')
p.add_argument('--moe', action='store_true')
a = p.parse_args()
if min(a.sequence, a.hidden, a.layers, a.heads, a.kv_heads, a.ffn, a.warmup, a.decode_tokens) < 1 or a.repeats < 3:
    raise ValueError('Invalid benchmark configuration')
if a.device == 'cuda' and not torch.cuda.is_available():
    raise RuntimeError('CUDA unavailable; benchmark was NOT run')
if a.device == 'cpu' and a.precision != 'FP32':
    raise ValueError('CPU comparison requires FP32')
torch.manual_seed(0)
torch.set_num_threads(1)
# Synapse FP32 SGEMM vs PyTorch IEEE FP32. No TF32 benchmark mismatch.
if a.device == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def sync():
    if a.device == 'cuda': torch.cuda.synchronize()

def amp():
    return torch.autocast('cuda', dtype=torch.float16) if a.device == 'cuda' and a.precision == 'FP16_COMPUTE' else nullcontext()

class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Linear(a.hidden, a.ffn, bias=False)
        self.gate = nn.Linear(a.hidden, a.ffn, bias=False)
        self.down = nn.Linear(a.ffn, a.hidden, bias=False)
    def forward(self, x): return self.down(F.silu(self.up(x)) * self.gate(x))

class MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = nn.Linear(a.hidden, 4, bias=False)
        self.experts = nn.ModuleList([FFN() for _ in range(4)])
        self.shared = FFN()
        self.aux = None
    def forward(self, x):
        shape = x.shape
        flat = x.reshape(-1, a.hidden)
        prob = self.router(flat).float().softmax(-1)
        scores, ids = prob.topk(2, dim=-1)
        weight = scores / scores.sum(-1, keepdim=True)
        freq = F.one_hot(ids, 4).float().mean((0, 1))
        self.aux = .01 * 4 * (prob.mean(0) * freq).sum()
        output = self.shared(flat)
        for e, expert in enumerate(self.experts):
            tokens, slot = torch.where(ids == e)
            if tokens.numel():
                part = expert(flat[tokens])
                output = output.index_add(0, tokens, part * weight[tokens, slot, None].to(part.dtype))
        return output.reshape(shape)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(a.hidden, a.hidden, bias=False)
        self.k = nn.Linear(a.hidden, a.hidden // a.heads * a.kv_heads, bias=False)
        self.v = nn.Linear(a.hidden, a.hidden // a.heads * a.kv_heads, bias=False)
        self.o = nn.Linear(a.hidden, a.hidden, bias=False)
        self.n1 = nn.RMSNorm(a.hidden, eps=1e-5)
        self.n2 = nn.RMSNorm(a.hidden, eps=1e-5)
        self.ffn = MoE() if a.moe else FFN()
        self.cache_k = self.cache_v = None
        self.offset = 0
    def reset(self): self.offset = 0; self.cache_k = self.cache_v = None
    def rope(self, x, start):
        dim = a.hidden // a.heads
        angles = torch.arange(start, start + x.shape[-2], device=x.device).float()[:, None] / (10000 ** (torch.arange(0, dim, 2, device=x.device).float() / dim))
        c, s = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        u, v = x[..., ::2], x[..., 1::2]
        return torch.stack((u*c-v*s, u*s+v*c), -1).flatten(-2)
    def forward(self, x, cache=False):
        b, n, h = x.shape
        y = self.n1(x)
        q = self.q(y).view(b, n, a.heads, -1).transpose(1, 2)
        k = self.k(y).view(b, n, a.kv_heads, -1).transpose(1, 2)
        v = self.v(y).view(b, n, a.kv_heads, -1).transpose(1, 2)
        start = self.offset if cache else 0
        q, k = self.rope(q, start), self.rope(k, start)
        if cache:
            # Match Synapse's preallocated cache (avoid growing torch.cat buffers).
            if self.cache_k is None:
                shape = (b, a.kv_heads, a.sequence+a.decode_tokens+1, h//a.heads)
                self.cache_k = torch.empty(shape, device=k.device, dtype=k.dtype)
                self.cache_v = torch.empty_like(self.cache_k)
            self.cache_k[..., start:start+n, :] = k
            self.cache_v[..., start:start+n, :] = v
            self.offset += n
            k, v = self.cache_k[..., :self.offset, :], self.cache_v[..., :self.offset, :]
        # A single-token decode query can see every cached key. A prefill uses square causal masking.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=(start == 0), enable_gqa=(a.heads != a.kv_heads))
        x = x + self.o(y.transpose(1, 2).reshape(b, n, h))
        return x + self.ffn(self.n2(x))

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(a.vocabulary, a.hidden)
        nn.init.uniform_(self.embedding.weight, -a.vocabulary**-.5, a.vocabulary**-.5)
        self.blocks = nn.ModuleList([Block() for _ in range(a.layers)])
        self.norm = nn.RMSNorm(a.hidden, eps=1e-5)
    def reset(self):
        for block in self.blocks: block.reset()
    def forward(self, ids, cache=False):
        x = self.embedding(ids)
        for block in self.blocks: x = block(x, cache)
        return F.linear(self.norm(x), self.embedding.weight)
    def aux(self): return sum(b.ffn.aux for b in self.blocks) if a.moe else 0

m = Model().to(a.device)
ids = torch.arange(a.sequence+1, device=a.device)[None] % a.vocabulary
x, labels = ids[:, :-1], ids[:, 1:]
optimizer = torch.optim.AdamW(m.parameters(), lr=.0003, eps=1e-7, weight_decay=.01)

def measure(work):
    for _ in range(a.warmup): work()
    samples = []
    for _ in range(a.repeats):
        sync(); start = time.perf_counter(); work(); sync()
        samples.append((time.perf_counter()-start)*1000)
    return statistics.median(samples)

@torch.no_grad()
def prefill():
    with amp(): m(x)

def train():
    optimizer.zero_grad(set_to_none=False)
    for param in m.parameters():
        if param.grad is None: param.grad = torch.zeros_like(param)
    with amp():
        y = m(x)
        loss = F.cross_entropy(y.float().reshape(-1, a.vocabulary), labels.reshape(-1)) + m.aux()
    # Static scaling matches Synapse. FP32 master weights, FP32 AdamW states.
    scale = 128 if a.precision == 'FP16_COMPUTE' else 1
    (loss*scale).backward()
    for param in m.parameters():
        if param.grad is not None: param.grad.div_(scale)
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1)
    optimizer.step()

if a.device == 'cuda': torch.cuda.reset_peak_memory_stats()
prefill_ms = measure(prefill)
backward_samples = []
for i in range(-a.warmup, a.repeats):
    optimizer.zero_grad(set_to_none=False)
    with amp():
        y = m(x)
        loss = y.sum()/(a.vocabulary*a.sequence) + m.aux()
    sync(); start = time.perf_counter(); loss.backward(); sync()
    if i >= 0: backward_samples.append((time.perf_counter()-start)*1000)
train_ms = measure(train)
decode_samples = []
with torch.no_grad():
    for i in range(-a.warmup, a.repeats):
        m.reset()
        with amp():
            m(x, cache=True)
            decode_ids = [torch.tensor([[t % a.vocabulary]], device=a.device) for t in range(a.decode_tokens)]
            sync(); start = time.perf_counter()
            for token in decode_ids: m(token, cache=True)
            sync()
        if i >= 0: decode_samples.append((time.perf_counter()-start)*1000)
decode_ms = statistics.median(decode_samples)
print(json.dumps(dict(framework='pytorch', device=a.device.upper(), precision=a.precision, moe=a.moe, sequence=a.sequence, vocabulary=a.vocabulary, hidden=a.hidden, layers=a.layers, heads=a.heads, kv_heads=a.kv_heads, ffn=a.ffn, batch=1, decode_tokens=a.decode_tokens, warmup=a.warmup, repeats=a.repeats, parameters=sum(p.numel() for p in m.parameters()), prefill_ms=prefill_ms, backward_ms=statistics.median(backward_samples), train_ms=train_ms, decode_ms=decode_ms, prefill_tokens_per_second=a.sequence*1000/prefill_ms, training_tokens_per_second=a.sequence*1000/train_ms, decode_tokens_per_second=a.decode_tokens*1000/decode_ms, peak_torch_allocated_bytes=torch.cuda.max_memory_allocated() if a.device=='cuda' else None, environment=dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name() if a.device=='cuda' else None, python=platform.python_version(), cpu_threads=1))))
