"""listen.py: trial plan and analysis (no audio playback in tests)."""
import json
import pytest
from listen import analyse, make_plan


@pytest.fixture
def corpus(tmp_path):
    for i in range(30):
        clap = [0.1 * ((r + i) % 8) for r in range(8)]
        rec = {"id": f"mc_{i:02d}", "prompt": f"caption {i}", "n": 8, "scores": {"clap": {"30": clap}}}
        (tmp_path / f"mc_{i:02d}.json").write_text(json.dumps(rec))
    (tmp_path / "rt_00.json").write_text(json.dumps({"id": "rt_00", "prompt": "x", "n": 8,
                                                      "scores": {"clap": {"30": [0] * 8}}}))
    return tmp_path


def test_plan_is_deterministic_musiccaps_only_and_blind(corpus):
    a, b = make_plan(corpus, n=20, seed=0), make_plan(corpus, n=20, seed=0)
    assert a == b and len(a) == 20 and len({t["id"] for t in a}) == 20
    assert all(t["id"].startswith("mc_") for t in a)
    for t in a:
        clap = json.loads((corpus / f"{t['id']}.json").read_text())["scores"]["clap"]["30"]
        assert t["pick"] == max(range(8), key=lambda r: clap[r]) and t["other"] != t["pick"]
        assert {t["A"], t["B"]} == {t["pick"], t["other"]}
    assert 3 <= sum(t["A"] == t["pick"] for t in a) <= 17       # order is randomised, not fixed


def test_analyse_two_sided_binomial():
    rows = [{"chose_pick": "1"}] * 15 + [{"chose_pick": "0"}] * 5
    r = analyse(rows)
    assert r["n"] == 20 and r["k_pick"] == 15
    assert r["p_two_sided"] == pytest.approx(0.04139, abs=1e-4)
    assert r["share_lo"] < 0.75 < r["share_hi"]
