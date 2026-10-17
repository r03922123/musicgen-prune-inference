"""
Best-of-N MusicGen with checkpoint pruning, in PyTorch + HF transformers.

Port of ../prune.py (MLX). Same three properties:
  1. pruned rows are removed from the token buffer AND the KV cache (self- and
     cross-attention), so later steps run a smaller batch
  2. sampling noise is keyed by (seed, step, original row id), never by a row's
     position in the current batch, so pruning cannot change a survivor's tokens
  3. a self-check: survivors == the same rows of the unpruned run, token-for-token

Drives MusicgenForCausalLM (the decoder) one step at a time instead of
model.generate(): generate() has no hook for dropping rows mid-run.

Run `python torch/prune.py` for the self-check (tiny random model, no downloads,
CUDA if available else CPU).
"""
import math, time
import torch
import torch.nn.functional as F
from transformers import MusicgenDecoderConfig, MusicgenForCausalLM
from transformers.cache_utils import DynamicCache, EncoderDecoderCache


def undelay(seq, K):
    """(B, 1+S, K) delayed buffer with BOS at 0 -> (B, T, K) aligned frames.
    Codebook i of frame t lives at position 1+t+i (MusicGen's delay pattern),
    so a prefix of length L holds T = L-K complete frames."""
    T = seq.shape[1] - K
    return torch.stack([seq[:, 1 + i : 1 + i + T, i] for i in range(K)], dim=-1)


def sample(logits, noise, top_k, temp):
    """Top-k sampling as Gumbel-max: argmax(l/temp + g) over the top-k set is an
    exact draw from softmax(l/temp) restricted to the top-k. logits (B, K, V)."""
    l = logits / temp
    kth = l.topk(top_k, dim=-1).values[..., -1:]
    l = l.masked_fill(l < kth, -math.inf)
    return (l + noise).argmax(dim=-1)  # (B, K)


def gumbel(n, K, V, seed, device):
    """(n, K, V) Gumbel noise from a generator seeded by (seed, step) alone."""
    gen = torch.Generator(device=device).manual_seed(seed)
    u = torch.rand((n, K, V), generator=gen, device=device)
    return -torch.log(-torch.log(u.clamp_min(1e-20)))


@torch.inference_mode()
def generate(model, cond, n, max_steps, score_fn=None, checkpoints=(), keep_frac=0.5,
             top_k=250, temp=1.0, guidance=3.0, seed=0, clock=None, row_keyed=True):
    """model: MusicgenForCausalLM. cond: (1, T_text, D) text conditioning, already
    projected to the decoder width. score_fn(frames (B,T,K)) -> (B,) scores.
    Returns (final buffer (B,1+max_steps,K), surviving row ids, {step: {row_id: score}}).
    clock: optional list; gets a perf_counter() stamp after every step (for timing).
    row_keyed=False keys noise by batch position instead (self-check mutation only)."""
    cfg, dev = model.config, model.device
    K, V, bos = cfg.num_codebooks, cfg.vocab_size, cfg.bos_token_id
    seq = torch.full((n, max_steps + 1, K), bos, dtype=torch.long, device=dev)
    ids = torch.arange(n, device=dev)        # original row id of each survivor
    # CFG: rows [0, B) are conditional, rows [B, 2B) unconditional (zero text)
    cond = cond.to(dev, model.dtype)
    txt = torch.cat([cond.expand(n, -1, -1), torch.zeros_like(cond).expand(n, -1, -1)])
    cache = EncoderDecoderCache(DynamicCache(), DynamicCache())
    kidx = torch.arange(K, device=dev)
    log = {}

    for off in range(max_steps):
        B = seq.shape[0]
        x = seq[:, off].repeat(2, 1).reshape(-1, 1)                  # (2B*K, 1)
        out = model(input_ids=x, encoder_hidden_states=txt, past_key_values=cache,
                    use_cache=True).logits.view(2 * B, K, V).float()
        logits = out[B:] + (out[:B] - out[B:]) * guidance
        # noise for ALL n original rows, then gather survivors -> row-id keyed
        g = gumbel(n, K, V, seed * 1_000_003 + off, dev)
        tok = sample(logits, g[ids] if row_keyed else g[:B], top_k, temp)
        # delay pattern: codebook i starts at step i, and ends K-1-i steps early
        mask = (kidx > off) | (kidx < K - (max_steps - off))
        seq[:, off + 1] = torch.where(mask, bos, tok)
        if clock is not None:
            if dev.type == "cuda":
                torch.cuda.synchronize()  # CUDA is async: without this timings are launch time
            clock.append(time.perf_counter())

        step = off + 1
        if step in checkpoints:
            scores = score_fn(undelay(seq[:, : step + 1], K)).float()
            log[step] = dict(zip(ids.tolist(), scores.tolist()))
            keep = max(1, math.ceil(B * keep_frac))
            if keep < B:
                top = torch.argsort(-scores, stable=True)[:keep].sort().values  # keep row order
                seq, ids = seq[top], ids[top]
                rows = torch.cat([top, top + B])                     # cond + uncond rows
                txt = txt[rows]
                cache.batch_select_indices(rows)                     # self- AND cross-attn K/V
    return seq, ids.tolist(), log


def random_model(hidden=64, layers=2, heads=4, ffn=128, K=4, V=2048, seed=0):
    """MusicGen decoder with random weights and real shapes -- no downloads.
    Enough for correctness checks, and for timing (FLOPs don't care about values).
    init std 0.2, not HF's 0.02: at 0.02 attention barely moves the logits, so a
    self-check could not notice a KV cache holding the wrong rows."""
    torch.manual_seed(seed)
    cfg = MusicgenDecoderConfig(vocab_size=V, num_codebooks=K, hidden_size=hidden,
                                num_hidden_layers=layers, num_attention_heads=heads,
                                ffn_dim=ffn, pad_token_id=V, bos_token_id=V,
                                max_position_embeddings=2048, initializer_factor=0.2)
    return MusicgenForCausalLM(cfg).eval()


def toy_score(f):
    return f[..., 0].float().mean(dim=1)  # toy, but discriminating


if __name__ == "__main__":
    from transformers.utils import logging
    logging.set_verbosity_error()  # bos/pad = V sits outside the vocab on purpose (embedding is V+1)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m, K, n, S = random_model().to(dev), 4, 8, 60
    torch.manual_seed(1)
    cond = torch.randn(1, 5, 64)
    cps = (20, 40, S)
    print(f"device={dev} torch={torch.__version__}")

    # 1. undelay inverts the delay pattern
    Fr = torch.randint(0, 2048, (2, 10, K)); buf = torch.full((2, 10 + K, K), 2048)
    for i in range(K): buf[:, 1 + i : 11 + i, i] = Fr[:, :, i]
    assert (undelay(buf, K) == Fr).all()

    base, base_ids, _ = generate(m, cond, n, S, seed=3)
    pr, pr_ids, log = generate(m, cond, n, S, toy_score, cps, seed=3)
    base_f = undelay(base, K)
    # 2. the toy exercises the mechanism: candidates differ, pruning drops rows
    assert len({tuple(base_f[i].flatten().tolist()) for i in range(n)}) == n
    assert [len(log[c]) for c in cps] == [8, 4, 2] and len(pr_ids) == 1
    # 3. survivors reproduce the same rows of the unpruned run token-for-token
    match = (undelay(pr, K) == base_f[pr_ids]).float().mean().item()
    # 4. applying the pruning rule offline to the baseline gives the same survivors
    alive = list(range(n))
    for c in cps:
        s = toy_score(undelay(base[:, : c + 1], K)).tolist()
        alive = sorted(sorted(alive, key=lambda r: -s[r])[: math.ceil(len(alive) / 2)])
    print(f"survivor={pr_ids} offline-rule={alive} token match={match:.4f}")
    assert alive == pr_ids and match == 1.0
    # 5. mutation: noise keyed by batch position instead of row id must break 3.
    mu, mu_ids, _ = generate(m, cond, n, S, toy_score, cps, seed=3, row_keyed=False)
    mu_match = (undelay(mu, K) == base_f[mu_ids]).float().mean().item()
    print(f"mutation (position-keyed noise): token match={mu_match:.4f}")
    assert mu_match < 1.0
    print("OK: pruning drops rows from tokens + KV cache without changing survivors")
