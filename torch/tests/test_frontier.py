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


def test_recommendation_is_cheapest_without_detected_loss(corpus):
    res = frontier(load_corpus(corpus), judge="cmi_qual")
    rec = res["recommended"]
    cheapest = min(res["schedules"], key=lambda x: x["sets"]["musiccaps"]["compute"]["mean"])
    assert rec["first"] == cheapest["first"] and rec["keep"] == cheapest["keep"]
