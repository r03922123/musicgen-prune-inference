"""Every headline number in the posts (post/post.md, post/decoding.md) must be what the committed results say.

Formats the value from the results JSON the way the post prints it and asserts that exact
string appears in the post. A changed result without an updated post fails CI."""
import json, re
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
TEXT = "\n".join((ROOT / "post" / f).read_text() for f in ("post.md", "decoding.md"))
R = lambda name: json.loads((ROOT / "results" / name).read_text())


def s(x, d=2):
    """Signed, with a real minus sign, as the post's tables print it."""
    return f"{'+' if x >= 0 else '−'}{abs(x):.{d}f}"


def ci(m, d=2, signed=False):
    f = (lambda x: s(x, d)) if signed else (lambda x: f"{x:.{d}f}")
    return f"{f(m['mean'])} [{f(m['lo'])}, {f(m['hi'])}]"


def expected():
    fr = R("frontier.json")
    pr = R("profile_decode_small_float16_pretrained_t4.json")["batches"]
    sched = {(x["first"], x["keep"]): x for x in fr["schedules"]}
    out = []
    for key in [(5, 0.5), (10, 0.5), (10, 0.65)]:
        mc, rt = sched[key]["sets"]["musiccaps"], sched[key]["sets"]["redteam"]
        out += [f"{mc['compute']['mean']:.3f}", ci(mc["same_winner"])]
    out.append(ci(sched[(5, 0.5)]["sets"]["redteam"]["same_winner"]))
    b1, b8 = pr["1"], pr["8"]
    out += [f"| 1 (2) | {b1['wall_ms_per_step']:.1f} | {b1['gpu_busy_ms_per_step']:.1f} | {b1['gpu_share_of_wall']:.0%} | {b1['kernels_per_step']:.0f} |",
            f"| 8 (16) | {b8['wall_ms_per_step']:.1f} | {b8['gpu_busy_ms_per_step']:.1f} | {b8['gpu_share_of_wall']:.0%} | {b8['kernels_per_step']:.0f} |"]
    return out


@pytest.mark.parametrize("literal", expected())
def test_post_quotes_the_results(literal):
    assert literal in TEXT, f"the posts do not contain {literal!r} (from results/)"


def test_timeline_figure_matches_the_profile():
    """The schematic's text quotes the profiler: wall and busy ms at batch 1, kernels per step."""
    svg = (ROOT / "post/figures/timeline.svg").read_text()
    b1 = R("profile_decode_small_float16_pretrained_t4.json")["batches"]["1"]
    k = f"~{round(b1['kernels_per_step'], -2):.0f} kernel"
    assert f"{b1['wall_ms_per_step']:.1f} ms" in svg and f"busy {b1['gpu_busy_ms_per_step']:.1f} ms" in svg
    assert svg.count(k) == 2 and len(re.findall(r"~\d+ kernel", svg)) == 2, "kernel counts must agree with the profile"
