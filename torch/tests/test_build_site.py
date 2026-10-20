"""build_site.py: display titles, the per-prompt payload and the headline copy."""
import json
from pathlib import Path
import numpy as np
from build_site import ROUNDS, gain_db, headline, prompts, title

ROOT = Path(__file__).resolve().parents[2]


def test_title_takes_first_clause_and_cuts_at_word():
    assert title("orchestral build-up, starts with a quiet solo cello") == "orchestral build-up"
    assert title("The music is slow. It has a piano.") == "The music is slow"
    t = title("word " * 40)
    assert len(t) <= 65 and t.endswith("…") and not t.startswith(" ")


def test_prompts_payload_keeps_only_game_rounds_and_q2_judges():
    s = {"clap": {str(t): [t] * 8 for t in range(5, 31, 5)},
         **{j: [1.0] * 8 for j in ("aes_CE", "aes_PQ", "mulan", "cmi_qual", "cmi_ali")}}
    p = prompts([{"id": "rt_00", "prompt": "slow build, then loud", "n": 8, "scores": s}])[0]
    assert p["set"] == "redteam" and p["title"] == "slow build"
    assert list(p["clap"]) == [str(t) for t in ROUNDS] and p["clap"]["20"] == [20] * 8
    assert set(p["judges"]) == {"aes_CE", "mulan", "cmi_qual", "cmi_ali"}


def test_headline_copies_committed_numbers():
    h = headline(ROOT / "results")
    fr = json.loads((ROOT / "results/frontier.json").read_text())
    s = next(x for x in fr["schedules"] if x["first"] == 10 and x["keep"] == 0.5)
    assert h["schedule_10_0.5"]["musiccaps"]["same_winner"]["mean"] == s["sets"]["musiccaps"]["same_winner"]["mean"]
    assert h["schedule_10_0.5"]["musiccaps"]["same_winner"]["n"] == 40
    assert all("source" in v for v in h.values())
    ag = json.loads((ROOT / "results/judge_agreement.json").read_text())
    assert h["agreement"]["clap~mulan"]["within"]["mean"] == ag["pairs"]["clap~mulan"]["within"]["mean"]


def test_gain_respects_peak_limit():
    sr = 32000
    loud_peaky = np.zeros(sr * 3); loud_peaky[::1000] = 0.9      # quiet overall, high peaks
    assert gain_db(loud_peaky, sr) <= -1.0 - 20 * np.log10(0.9) + 1e-9


def test_gain_boost_is_capped_for_near_silent_takes():
    sr = 32000
    x = 0.001 * np.sin(np.linspace(0, 2000, sr * 3))
    assert gain_db(x, sr) == 15.0
