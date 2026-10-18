"""frontier.py on a synthetic corpus written to tmp_path."""
import json
import pytest
from evaluate import checkpoints, simulate
from frontier import GRID, frontier, load_corpus


def fake(pid, clap, judge):
    """clap: {t: 8 scores} for t in 5..30; judge: 8 scores."""
    return {"id": pid, "scores": {"clap": {str(t): v for t, v in clap.items()}, "cmi_qual": judge,
                                  "aes_CE": [-x for x in judge]}}


@pytest.fixture
def corpus(tmp_path):
    good = [8, 7, 6, 5, 4, 3, 2, 1]
    recs = [fake("mc_a", {t: good for t in range(5, 31, 5)}, good),        # pruner == judge
            fake("mc_b", {t: good for t in range(5, 31, 5)}, good[::-1]),  # pruner == -judge
            fake("rt_00", {t: good for t in range(5, 31, 5)}, good)]
    for r in recs:
        (tmp_path / f"{r['id']}.json").write_text(json.dumps(r))
    return tmp_path


def test_load_and_split(corpus):
    recs = load_corpus(corpus)
    assert sorted(r["id"] for r in recs) == ["mc_a", "mc_b", "rt_00"]


def test_frontier_grid_sets_and_numbers(corpus):
    res = frontier(load_corpus(corpus), judge="cmi_qual")
    assert len(res["schedules"]) == len(GRID) + 1               # + the unpruned baseline
    base = res["schedules"][0]
    assert base["first"] is None and base["sets"]["musiccaps"]["compute"]["mean"] == 1.0
    s = next(x for x in res["schedules"] if x["first"] == 5 and x["keep"] == 0.5)
    mc = s["sets"]["musiccaps"]
    assert mc["compute"]["n"] == 2 and s["sets"]["redteam"]["compute"]["n"] == 1
    good = [8, 7, 6, 5, 4, 3, 2, 1]
    exp = simulate({t: good for t in range(5, 30, 5)}, good, checkpoints(5), 0.5)["compute"]
    assert mc["compute"]["mean"] == pytest.approx(exp)
    # the pruner keeps row 0 throughout = the unpruned pick, so regret is 0 whatever the judge
    assert mc["regret"]["mean"] == 0 and mc["same_winner"]["mean"] == 1
    assert "aes_CE" in s["secondary"]


def test_recommendation_is_cheapest_with_same_winner_at_least_090(corpus):
    # mc_c: CLAP at 5 s ranks the final winner (row 0) last, from 10 s on it ranks it first.
    # Every 5 s schedule kills row 0 on mc_c -> MusicCaps same-winner 2/3 < 0.90; 10 s ones keep it.
    good = [8, 7, 6, 5, 4, 3, 2, 1]
    c = fake("mc_c", {t: (good[::-1] if t == 5 else good) for t in range(5, 31, 5)}, good)
    (corpus / "mc_c.json").write_text(json.dumps(c))
    res = frontier(load_corpus(corpus), judge="cmi_qual")
    for s in res["schedules"]:
        if s["first"] is not None:
            assert s["sets"]["musiccaps"]["same_winner"]["mean"] == pytest.approx(2 / 3 if s["first"] == 5 else 1)
    assert (res["recommended"]["first"], res["recommended"]["keep"]) == (10, 0.5)


def test_final_survivor_count_per_schedule(corpus):
    res = frontier(load_corpus(corpus), judge="cmi_qual")
    surv = {(s["first"], s["keep"]): s["sets"]["musiccaps"]["survivors"]["mean"] for s in res["schedules"]}
    assert surv[(None, None)] == 8
    assert surv[(5, 0.5)] == 1                                   # 8 -> 4 -> 2 -> 1
    assert surv[(10, 0.75)] == 5                                 # 8 -> 6 -> 5


def test_selection_block_per_judge_and_set(corpus):
    res = frontier(load_corpus(corpus), judge="cmi_qual")
    sel = res["selection"]
    assert set(sel) == {"cmi_qual", "aes_CE"}
    # mc_a: pick = judge's best (gain 8 - 4.5); mc_b: pick = judge's worst (gain 1 - 4.5)
    assert sel["cmi_qual"]["musiccaps"]["gain_vs_random"]["mean"] == pytest.approx(0.0)
    assert sel["cmi_qual"]["musiccaps"]["oracle_gain"]["mean"] == pytest.approx(3.5)
    assert sel["cmi_qual"]["redteam"]["spearman"]["mean"] == pytest.approx(1.0)
