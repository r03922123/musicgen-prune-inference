"""build_post_data.py: display titles, the per-prompt payload, loudness gain."""
import json
from pathlib import Path
import numpy as np
from build_post_data import PREFIXES, gain_db, prompts, title

ROOT = Path(__file__).resolve().parents[2]


def test_title_takes_first_clause_and_cuts_at_word():
    assert title("orchestral build-up, starts with a quiet solo cello") == "orchestral build-up"
    assert title("The music is slow. It has a piano.") == "The music is slow"
    t = title("word " * 40)
    assert len(t) <= 65 and t.endswith("…") and not t.startswith(" ")


def test_prompts_payload_keeps_prefixes_chunks_and_q2_judges():
    s = {"clap": {str(t): [t] * 8 for t in range(5, 31, 5)},
         **{j: [1.0] * 8 for j in ("aes_CE", "aes_PQ", "mulan", "cmi_qual", "cmi_ali")}}
    p = prompts([{"id": "rt_00", "prompt": "slow build, then loud", "n": 8, "scores": s}],
                {"rt_00": [[0.1, 0.2, 0.3]] * 8})[0]
    assert p["set"] == "redteam" and p["title"] == "slow build"
    assert list(p["clap"]) == [str(t) for t in PREFIXES] and p["clap"]["20"] == [20] * 8
    assert p["chunks"][0] == [0.1, 0.2, 0.3]
    assert set(p["judges"]) == {"aes_CE", "mulan", "cmi_qual", "cmi_ali"}


def test_committed_data_json_is_reproducible():
    """post/data.json is what the builder makes from the committed corpus and chunk scores."""
    recs = [json.loads(p.read_text()) for p in sorted((ROOT / "data/corpus").glob("*.json"))]
    chunks = json.loads((ROOT / "results/clap_chunks.json").read_text())
    got = json.loads((ROOT / "post/data.json").read_text())
    assert got["prompts"] == json.loads(json.dumps(prompts(recs, chunks)))


def test_gain_respects_peak_limit():
    sr = 32000
    loud_peaky = np.zeros(sr * 3); loud_peaky[::1000] = 0.9      # quiet overall, high peaks
    assert gain_db(loud_peaky, sr) <= -1.0 - 20 * np.log10(0.9) + 1e-9


def test_gain_boost_is_capped_for_near_silent_takes():
    sr = 32000
    x = 0.001 * np.sin(np.linspace(0, 2000, sr * 3))
    assert gain_db(x, sr) == 15.0
