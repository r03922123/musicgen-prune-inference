"""
Best-of-N MusicGen with checkpoint pruning, in MLX.

Generate N candidates as one batch. At each checkpoint step, decode every
survivor's clip-so-far, score it, and keep the top fraction. Dropped rows are
removed from the token buffer AND the KV cache, so later steps run a smaller
batch. No checkpoints = plain best-of-N baseline (same code path).

Key design choice: each candidate's sampling noise is keyed by (seed, step,
original row id) -- never by its position in the current batch. So pruning
cannot change a survivor's tokens. That buys two things:
  1. a strong self-check: pruned survivors == the same rows of the unpruned run
  2. an exact quality comparison: the pruned winner is always one of the
     baseline's candidates, so "did pruning pick a worse clip?" has one answer.

Run `python prune.py` for the self-check (tiny random model, no downloads).
"""
import sys, math, time
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).parent / "vendor"))  # upstream uses flat imports
from musicgen import KVCache, TransformerBlock  # noqa: E402


def undelay(seq, K):
    """(B, 1+S, K) delayed buffer with BOS at 0 -> (B, T, K) aligned frames.
    Codebook i of frame t lives at position 1+t+i (MusicGen's delay pattern),
    so a prefix of length L holds T = L-K complete frames."""
    T = seq.shape[1] - K
    return mx.stack([seq[:, 1 + i : 1 + i + T, i] for i in range(K)], axis=-1)


def sample(logits, noise, top_k, temp):
    """Top-k sampling as Gumbel-max: argmax(l/temp + g) over the top-k set is an
    exact draw from softmax(l/temp) restricted to the top-k. logits (B, V, K)."""
    l = logits / temp
    kth = mx.topk(l, top_k, axis=1).min(axis=1, keepdims=True)
    l = mx.where(l >= kth, l, -mx.inf)
    return mx.argmax(l + noise, axis=1)  # (B, K)


def generate(model, cond, n, max_steps, score_fn=None, checkpoints=(), keep_frac=0.5,
             top_k=250, temp=1.0, guidance=3.0, seed=0, clock=None):
    """cond: (1, T_text, D) text conditioning. score_fn(frames (B,T,K)) -> (B,) scores.
    Returns (final buffer (B,1+max_steps,K), surviving row ids, {step: {row_id: score}}).
    clock: optional list; gets a perf_counter() stamp after every step (for timing)."""
    K, bos, V = model.num_codebooks, model.bos_token_id, model.codebook_size
    seq = mx.full((n, max_steps + 1, K), bos)
    ids = mx.arange(n)                       # original row id of each survivor
    # CFG: rows [0, B) are conditional, rows [B, 2B) unconditional (zero text)
    txt = mx.concatenate([mx.repeat(cond, n, 0), mx.zeros((n, *cond.shape[1:]))], 0)
    txt = txt.astype(model.linears[0].weight.dtype)  # fp32 text into fp16 weights would upcast
    hd = model.hidden_size // model.num_attention_heads
    cache = [KVCache(hd, model.num_attention_heads) for _ in model.layers]
    kidx = mx.arange(K)
    log = {}

    for off in range(max_steps):
        B = seq.shape[0]
        x = mx.concatenate([seq[:, off : off + 1]] * 2, 0)         # (2B, 1, K)
        out = model(x, txt, cache)[:, 0]                            # (2B, V, K)
        logits = out[B:] + (out[:B] - out[B:]) * guidance
        # noise for ALL n original rows, then gather survivors -> row-id keyed
        g = mx.random.gumbel((n, V, K), key=mx.random.key(seed * 1_000_003 + off))
        tok = sample(logits, g[ids], top_k, temp)
        # delay pattern: codebook i starts at step i, and ends K-1-i steps early
        mask = (kidx > off) | (kidx < K - (max_steps - off))
        seq[:, off + 1] = mx.where(mask, bos, tok)
        mx.eval(seq)  # MLX is lazy: without this nothing runs, and timings are fake
        if clock is not None:
            clock.append(time.perf_counter())

        step = off + 1
        if step in checkpoints:
            scores = score_fn(undelay(seq[:, : step + 1], K))
            mx.eval(scores)
            log[step] = dict(zip(ids.tolist(), scores.tolist()))
            keep = max(1, math.ceil(B * keep_frac))
            if keep < B:
                top = mx.sort(mx.argsort(-scores)[:keep])           # keep row order
                seq, ids = seq[top], ids[top]
                rows = mx.concatenate([top, top + B])                # cond + uncond rows
                txt = txt[rows]
                for c in cache:
                    c.keys, c.values = c.keys[rows], c.values[rows]
                mx.eval(seq, txt, *[c.keys for c in cache], *[c.values for c in cache])
    return seq, ids.tolist(), log


def random_model(hidden=64, layers=2, heads=4, ffn=128, K=4, V=2048, seed=0):
    """MusicGen decoder with random weights and real shapes -- no downloads.
    Enough for correctness checks, and for timing (FLOPs don't care about values)."""
    from types import SimpleNamespace as NS
    from musicgen import MusicGen
    mx.random.seed(seed)
    cfg = NS(decoder=NS(hidden_size=hidden, num_attention_heads=heads, ffn_dim=ffn,
                        num_hidden_layers=layers))
    m = MusicGen.__new__(MusicGen)          # skip __init__: it downloads T5 + EnCodec
    nn.Module.__init__(m)
    m.num_codebooks, m.codebook_size, m.bos_token_id = K, V, V
    m.hidden_size, m.num_attention_heads, m.sampling_rate = hidden, heads, 32000
    m.emb = [nn.Embedding(V + 1, hidden) for _ in range(K)]
    m.layers = [TransformerBlock(cfg) for _ in range(layers)]
    m.out_norm = nn.LayerNorm(hidden, eps=1e-5)
    m.linears = [nn.Linear(hidden, V, bias=False) for _ in range(K)]
    mx.eval(m.parameters())
    return m


if __name__ == "__main__":
    import numpy as np
    m, K, n, S = random_model(), 4, 8, 60
    cond = mx.random.normal((1, 5, 64))
    score = lambda f: f[..., 0].astype(mx.float32).mean(axis=1)  # toy, but discriminating
    cps = (20, 40, S)

    # 1. undelay inverts the delay pattern
    F = np.random.randint(0, 2048, (2, 10, K)); buf = np.full((2, 10 + K, K), 2048)
    for i in range(K): buf[:, 1 + i : 11 + i, i] = F[:, :, i]
    assert (np.array(undelay(mx.array(buf), K)) == F).all()

    base, base_ids, _ = generate(m, cond, n, S, seed=1)
    pr, pr_ids, log = generate(m, cond, n, S, score, cps, seed=1)
    base_f = undelay(base, K)
    # 2. the toy exercises the mechanism: candidates differ, pruning drops rows
    assert len({tuple(np.array(base_f[i]).ravel()) for i in range(n)}) == n
    assert [len(log[c]) for c in cps] == [8, 4, 2] and len(pr_ids) == 1
    # 3. survivors reproduce the same rows of the unpruned run token-for-token
    match = (np.array(undelay(pr, K)) == np.array(base_f[mx.array(pr_ids)])).mean()
    # 4. applying the pruning rule offline to the baseline gives the same survivors
    alive = list(range(n))
    for c in cps:
        s = score(undelay(base[:, : c + 1], K)).tolist()
        alive = sorted(sorted(alive, key=lambda r: -s[r])[: math.ceil(len(alive) / 2)])
    print(f"survivor={pr_ids} offline-rule={alive} token match={match:.4f}")
    assert alive == pr_ids and match == 1.0
    print("OK: pruning drops rows from tokens + KV cache without changing survivors")
