"""Every headline number in post/post.md must be what the committed results files say.

Formats the value from the results JSON the way the post prints it and asserts that exact
string appears in the post. A changed result without an updated post fails CI."""
import json, re
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
TEXT = (ROOT / "post/post.md").read_text()
R = lambda name: json.loads((ROOT / "results" / name).read_text())


def s(x, d=2):
    """Signed, with a real minus sign, as the post's tables print it."""
    return f"{'+' if x >= 0 else '−'}{abs(x):.{d}f}"


def ci(m, d=2, signed=False):
    f = (lambda x: s(x, d)) if signed else (lambda x: f"{x:.{d}f}")
    return f"{f(m['mean'])} [{f(m['lo'])}, {f(m['hi'])}]"


def expected():
    fr, fu, br, ag = R("frontier.json"), R("future_only.json"), R("bracket.json"), R("judge_agreement.json")
    pr = R("profile_decode_small_float16_pretrained_t4.json")["batches"]
    sched = {(x["first"], x["keep"]): x for x in fr["schedules"]}
    out = []
    for key in [(5, 0.5), (10, 0.5), (10, 0.65)]:
        mc, rt = sched[key]["sets"]["musiccaps"], sched[key]["sets"]["redteam"]
        out += [f"{mc['compute']['mean']:.3f}", ci(mc["same_winner"])]
    out.append(ci(sched[(5, 0.5)]["sets"]["redteam"]["same_winner"]))
    for j in ("cmi_qual", "cmi_ali", "aes_CE", "mulan"):
        m = fr["selection"][j]["musiccaps"]
        d = 3
        out += [ci(m["gain_vs_random"], d, signed=True), f"{m['oracle_gain']['mean']:.3f}"]
    b = fr["selection"]["cmi_qual"]["musiccaps"]["gain_vs_random_bonferroni"]
    out.append(f"[{s(b['lo'], 3)}, {s(b['hi'], 3)}]")
    fut = {(x["first"], x["keep"]): x["sets"] for x in fu["schedules"]}
    for key in [(10, 0.5), (10, 0.65), (10, 0.75)]:
        mc = fut[key]["musiccaps"]
        out += [f"{mc['kept']['mean']:.2f} vs {mc['chance']['mean']:.2f}", ci(mc["excess"], signed=True),
                ci(fut[key]["redteam"]["excess"], signed=True)]
    p = fu["predictability"]
    out += [s(p["musiccaps"]["clap10_vs_chunks23"]["mean"]),
            ci(p["redteam"]["clap10_vs_chunks23"], signed=True), s(p["musiccaps"]["clap10_vs_chunk1"]["mean"])]
    out += [ci(ag["pairs"]["clap~mulan"]["within"], signed=True), s(ag["pairs"]["clap~mulan"]["between"])]
    out += [ci(br["judges"]["cmi_qual"]["musiccaps"]["same_as_unpruned"])]
    b1, b8 = pr["1"], pr["8"]
    out += [f"| 1 (2) | {b1['wall_ms_per_step']:.1f} | {b1['gpu_busy_ms_per_step']:.1f} | {b1['gpu_share_of_wall']:.0%} | {b1['kernels_per_step']:.0f} |",
            f"| 8 (16) | {b8['wall_ms_per_step']:.1f} | {b8['gpu_busy_ms_per_step']:.1f} | {b8['gpu_share_of_wall']:.0%} | {b8['kernels_per_step']:.0f} |"]
    return out


@pytest.mark.parametrize("literal", expected())
def test_post_quotes_the_results(literal):
    assert literal in TEXT, f"post.md does not contain {literal!r} (from results/)"


def test_timeline_figure_matches_the_profile():
    """The schematic's text quotes the profiler: wall and busy ms at batch 1, kernels per step."""
    svg = (ROOT / "post/figures/timeline.svg").read_text()
    b1 = R("profile_decode_small_float16_pretrained_t4.json")["batches"]["1"]
    k = f"~{round(b1['kernels_per_step'], -2):.0f} kernel"
    assert f"{b1['wall_ms_per_step']:.1f} ms" in svg and f"busy {b1['gpu_busy_ms_per_step']:.1f} ms" in svg
    assert svg.count(k) == 2 and len(re.findall(r"~\d+ kernel", svg)) == 2, "kernel counts must agree with the profile"
