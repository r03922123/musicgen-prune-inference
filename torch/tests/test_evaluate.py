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


# --- part 2: per-prompt metrics and bootstrap CIs ---
import numpy as np
from evaluate import prompt_metrics, bootstrap_ci, aggregate

JUDGE = [1.0, 2.0, 3.0, 4.0]


def test_toy_prompt_metrics():
    m = prompt_metrics(EARLY, FINAL, JUDGE, [10, 20], 0.5)
    assert m["compute"] == pytest.approx(70 / 120)
    assert m["regret"] == pytest.approx(-2.0)        # judge(best-of-4 by pruner = row 0) - judge(row 2)
    assert m["oracle_regret"] == pytest.approx(1.0)  # judge max (row 3) - judge(row 2)
    assert m["same_winner"] == 0.0
    assert m["false_kill"] == 0.5                    # judge top-2 = {3, 2}; row 3 was pruned
    assert m["spearman"] == pytest.approx(0.2)       # pruner @10s vs judge: ranks [1,4,3,2] vs [1,2,3,4]
    assert m["survivors"] == 1                       # false-kill >= 0.5 whenever this is 1


def test_unpruned_schedule_has_zero_regret():
    m = prompt_metrics(EARLY, FINAL, JUDGE, [], 0.5)
    assert m["regret"] == 0 and m["same_winner"] == 1.0 and m["false_kill"] == 0 and m["compute"] == 1


def test_bootstrap_is_seeded_and_brackets_the_mean():
    x = np.array([0.0, 1.0, 2.0, 5.0, -1.0, 3.0])
    a, b = bootstrap_ci(x, seed=0), bootstrap_ci(x, seed=0)
    assert a == b
    assert a["mean"] == pytest.approx(x.mean()) and a["lo"] < a["mean"] < a["hi"] and a["n"] == 6


def test_bootstrap_constant_input_has_zero_width():
    ci = bootstrap_ci(np.full(10, 0.25))
    assert ci["lo"] == ci["hi"] == ci["mean"] == pytest.approx(0.25)


def test_bootstrap_drops_nan():
    assert bootstrap_ci(np.array([1.0, np.nan, 3.0]))["n"] == 2


def test_aggregate_reports_every_metric_with_n():
    rows = [prompt_metrics(EARLY, FINAL, JUDGE, [10, 20], 0.5)] * 3
    agg = aggregate(rows)
    assert set(agg) == {"compute", "regret", "oracle_regret", "same_winner", "false_kill", "spearman", "survivors"}
    assert agg["regret"]["mean"] == pytest.approx(-2.0) and agg["regret"]["n"] == 3


def test_bootstrap_all_nan_is_empty_not_a_warning():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ci = bootstrap_ci(np.array([np.nan, np.nan]))
    assert ci["n"] == 0 and np.isnan(ci["mean"])


# Q2 selection value (tasks/decisions.md, Day 2): toy prompt, pruner FINAL [.9, .2, .3, .1]
# picks row 0. Judge [2, 4, 3, 1]: mean 2.5, so gain vs a uniformly random candidate =
# 2 - 2.5 = -0.5; oracle gain = 4 - 2.5 = 1.5; Spearman(FINAL, judge) on ranks
# [4,2,3,1] vs [2,4,3,1] = 1 - 6*(4+4+0+0)/(4*15) = 0.2.
def test_toy_selection_metrics():
    from evaluate import selection_metrics
    m = selection_metrics(FINAL, [2, 4, 3, 1])
    assert m["gain_vs_random"] == pytest.approx(-0.5)
    assert m["oracle_gain"] == pytest.approx(1.5)
    assert m["spearman"] == pytest.approx(0.2)


def test_selection_constant_pruner_picks_row_0_and_spearman_is_nan():
    from evaluate import selection_metrics
    m = selection_metrics([.5, .5, .5, .5], [2, 4, 3, 1])   # ties: first row picked
    assert m["gain_vs_random"] == pytest.approx(-0.5) and m["oracle_gain"] == pytest.approx(1.5)
    assert m["spearman"] != m["spearman"]                     # constant input: NaN, dropped by bootstrap


# Bracket (single elimination) vs top-k successive halving. Toy, 4 rows, one checkpoint:
#   early [.4, .3, .2, .1], final [.1, .2, .9, .3]; the 3 distinct brackets
#   (0-1, 2-3) -> 0, 2 -> final 2 | (0-2, 1-3) -> 0, 1 -> final 1 | (0-3, 1-2) -> 0, 1 -> 1
#   top-k keep 0.5 keeps {0, 1} -> winner 1; unpruned pick = argmax final = 2.
#   judge [1, 2, 3, 4]: regret vs the unpruned pick (3) = 0, 1, 1.
def test_all_brackets_counts():
    from evaluate import all_brackets
    assert len(all_brackets(2)) == 1 and len(all_brackets(4)) == 3 and len(all_brackets(8)) == 315
    assert all(sorted(b) == list(range(8)) for b in all_brackets(8))


def test_bracket_champion_pairs_adjacent_and_ties_keep_lower_row():
    from evaluate import bracket_champion
    assert bracket_champion((0, 1, 2, 3), [[.4, .3, .2, .1], [.1, .2, .9, .3]]) == 2
    assert bracket_champion((0, 3, 1, 2), [[.4, .3, .2, .1], [.1, .2, .9, .3]]) == 1
    assert bracket_champion((3, 2), [[.5, .5, .5, .5]]) == 2


def test_toy_bracket_metrics():
    from evaluate import bracket_metrics
    m = bracket_metrics({10: [.4, .3, .2, .1]}, [.1, .2, .9, .3], [1, 2, 3, 4], [10])
    assert m["brackets"] == 3
    assert m["same_as_unpruned"] == pytest.approx(1 / 3)
    assert m["same_as_topk"] == pytest.approx(2 / 3)
    assert m["regret"] == pytest.approx(2 / 3)
    assert m["topk_regret"] == pytest.approx(1.0)
