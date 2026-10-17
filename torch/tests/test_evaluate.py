"""evaluate.py on hand-computed toy cases.

Toy prompt, 4 candidates, checkpoints 10 s and 20 s, keep 0.5:
  pruner @10s  [.1, .4, .3, .2]  -> keep rows 1, 2
  pruner @20s  [.9, .5, .6, .0]  -> keep row 2 (row 0's .9 is dead and must stay dead)
  pruner @30s  [.9, .2, .3, .1]  -> unpruned best-of-4 = row 0; pruned winner = row 2
  compute = (4*10 + 2*10 + 1*10) / (4*30) = 70/120
"""
import pytest
from evaluate import checkpoints, simulate

EARLY = {10: [.1, .4, .3, .2], 20: [.9, .5, .6, .0]}
FINAL = [.9, .2, .3, .1]


def test_checkpoints_every_first_interval():
    assert checkpoints(5) == [5, 10, 15, 20, 25]
    assert checkpoints(10) == [10, 20]
    assert checkpoints(10, every=5) == [10, 15, 20, 25]


def test_toy_survivors_winner_compute():
    r = simulate(EARLY, FINAL, [10, 20], 0.5)
    assert r["alive"] == {10: [1, 2], 20: [2]}
    assert r["survivors"] == [2] and r["winner"] == 2
    assert r["compute"] == pytest.approx(70 / 120)


def test_no_checkpoints_is_plain_best_of_n():
    r = simulate(EARLY, FINAL, [], 0.5)
    assert r["survivors"] == [0, 1, 2, 3] and r["winner"] == 0 and r["compute"] == 1.0


def test_keep_is_ceil_of_fraction():
    s = [.8, .7, .6, .5, .4, .3, .2, .1]
    assert simulate({5: s}, s, [5], 0.65)["alive"][5] == [0, 1, 2, 3, 4, 5]   # ceil(5.2) = 6
    assert simulate({5: s}, s, [5], 0.75)["alive"][5] == [0, 1, 2, 3, 4, 5]   # ceil(6.0) = 6


def test_ties_keep_lower_row_ids():
    r = simulate({10: [.5, .5, .5, .5]}, [0, 0, 0, 0], [10], 0.5)
    assert r["alive"][10] == [0, 1] and r["winner"] == 0


def test_pruning_stops_at_one_survivor():
    s = [.4, .3, .2, .1]
    r = simulate({5: s, 10: s, 15: s, 20: s}, s, [5, 10, 15, 20], 0.5)
    assert r["alive"] == {5: [0, 1], 10: [0]}            # no entries after one is left
    assert r["compute"] == pytest.approx((4 * 5 + 2 * 5 + 1 * 20) / (4 * 30))


def test_winner_is_best_final_among_survivors():
    r = simulate({10: [.1, .9, .8, .2]}, [.99, .1, .5, .3], [10], 0.5)
    assert r["survivors"] == [1, 2] and r["winner"] == 2   # row 0 has the best final but is dead
