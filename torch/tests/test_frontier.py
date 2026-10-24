"""frontier.py on a synthetic corpus written to tmp_path."""
import json
import pytest
from evaluate import checkpoints, simulate
from frontier import GRID, frontier, load_corpus


def fake(pid, clap):
    """clap: {t: 8 scores} for t in 5..30."""
    return {"id": pid, "scores": {"clap": {str(t): v for t, v in clap.items()}}}


@pytest.fixture
def corpus(tmp_path):
    good = [8, 7, 6, 5, 4, 3, 2, 1]
    recs = [fake(pid, {t: good for t in range(5, 31, 5)}) for pid in ("mc_a", "mc_b", "rt_00")]
    for r in recs:
        (tmp_path / f"{r['id']}.json").write_text(json.dumps(r))
    return tmp_path


def test_load_and_split(corpus):
    recs = load_corpus(corpus)
    assert sorted(r["id"] for r in recs) == ["mc_a", "mc_b", "rt_00"]


def test_frontier_grid_sets_and_numbers(corpus):
    res = frontier(load_corpus(corpus))
    assert len(res["schedules"]) == len(GRID) + 1               # + the unpruned baseline
    base = res["schedules"][0]
    assert base["first"] is None and base["sets"]["musiccaps"]["compute"]["mean"] == 1.0
    s = next(x for x in res["schedules"] if x["first"] == 5 and x["keep"] == 0.5)
    mc = s["sets"]["musiccaps"]
    assert mc["compute"]["n"] == 2 and s["sets"]["redteam"]["compute"]["n"] == 1
    good = [8, 7, 6, 5, 4, 3, 2, 1]
    exp = simulate({t: good for t in range(5, 30, 5)}, good, checkpoints(5), 0.5)["compute"]
    assert mc["compute"]["mean"] == pytest.approx(exp)
    assert mc["same_winner"]["mean"] == 1                       # the pruner keeps row 0 throughout


def test_recommendation_is_cheapest_with_same_winner_at_least_090(corpus):
    # mc_c: CLAP at 5 s ranks the final winner (row 0) last, from 10 s on it ranks it first.
    # Every 5 s schedule kills row 0 on mc_c -> MusicCaps same-winner 2/3 < 0.90; 10 s ones keep it.
    good = [8, 7, 6, 5, 4, 3, 2, 1]
    c = fake("mc_c", {t: (good[::-1] if t == 5 else good) for t in range(5, 31, 5)})
    (corpus / "mc_c.json").write_text(json.dumps(c))
    res = frontier(load_corpus(corpus))
    for s in res["schedules"]:
        if s["first"] is not None:
            assert s["sets"]["musiccaps"]["same_winner"]["mean"] == pytest.approx(2 / 3 if s["first"] == 5 else 1)
    assert (res["recommended"]["first"], res["recommended"]["keep"]) == (10, 0.5)
