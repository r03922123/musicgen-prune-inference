"""The self-checks of torch/prune.py as tests, plus mutation tests that must fail.

Tiny random decoder on CPU: no downloads, a few seconds."""
import math
import pytest
import torch
from transformers.cache_utils import EncoderDecoderCache
from transformers.utils import logging
from prune import generate, random_model, toy_score, undelay

logging.set_verbosity_error()
K, N, S, CPS, SEED = 4, 8, 60, (20, 40, 60), 3   # seed 3: survivors [1,4,5,6] -> [1,5] -> [1]


@pytest.fixture(scope="module")
def setup():
    m = random_model()
    torch.manual_seed(1)
    cond = torch.randn(1, 5, 64)
    base, _, _ = generate(m, cond, N, S, seed=SEED)
    return m, cond, base


def pruned_match(setup, **kw):
    """Fraction of survivor tokens equal to the same rows of the unpruned run."""
    m, cond, base = setup
    pr, ids, log = generate(m, cond, N, S, toy_score, CPS, seed=SEED, **kw)
    return (undelay(pr, K) == undelay(base, K)[ids]).float().mean().item(), ids, log


def test_undelay_inverts_delay_pattern():
    F = torch.randint(0, 2048, (2, 10, K))
    buf = torch.full((2, 10 + K, K), 2048)
    for i in range(K):
        buf[:, 1 + i : 11 + i, i] = F[:, :, i]
    assert (undelay(buf, K) == F).all()


def test_candidates_differ(setup):
    f = undelay(setup[2], K)
    assert len({tuple(f[i].flatten().tolist()) for i in range(N)}) == N


def test_survivors_match_unpruned_rows(setup):
    match, ids, log = pruned_match(setup)
    assert [len(log[c]) for c in CPS] == [8, 4, 2] and len(ids) == 1
    assert ids != [0]  # a non-prefix survivor, so "keep the first rows" bugs can't pass
    assert match == 1.0


def test_offline_rule_picks_online_survivors(setup):
    _, ids, _ = pruned_match(setup)
    base, alive = setup[2], list(range(N))
    for c in CPS:
        s = toy_score(undelay(base[:, : c + 1], K)).tolist()
        alive = sorted(sorted(alive, key=lambda r: -s[r])[: math.ceil(len(alive) / 2)])
    assert alive == ids


# --- mutation tests: each breaks one property; the token match must drop below 1 ---

def test_mutation_position_keyed_noise(setup):
    assert pruned_match(setup, row_keyed=False)[0] < 1.0


def _patch_select(monkeypatch, self_flip, cross_flip):
    def select(cache, rows):
        cache.self_attention_cache.batch_select_indices(rows.flip(0) if self_flip else rows)
        cache.cross_attention_cache.batch_select_indices(rows.flip(0) if cross_flip else rows)
    monkeypatch.setattr(EncoderDecoderCache, "batch_select_indices", select)


def test_mutation_wrong_self_attention_cache_rows(setup, monkeypatch):
    _patch_select(monkeypatch, self_flip=True, cross_flip=False)
    assert pruned_match(setup)[0] < 1.0


def test_mutation_wrong_cross_attention_cache_rows(setup, monkeypatch):
    _patch_select(monkeypatch, self_flip=False, cross_flip=True)
    assert pruned_match(setup)[0] < 1.0


def test_unpatched_select_is_exact(setup, monkeypatch):
    """Control for the two tests above: the same patch without flips must still match."""
    _patch_select(monkeypatch, self_flip=False, cross_flip=False)
    assert pruned_match(setup)[0] == 1.0
