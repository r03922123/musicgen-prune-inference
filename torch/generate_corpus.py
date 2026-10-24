"""
Unpruned best-of-N corpus: every schedule is later evaluated offline from it.

  python torch/generate_corpus.py --out corpus                 # all 50 prompts, 30 s, N=8
  python torch/generate_corpus.py --out c --limit 1 --seconds 6 --judges none   # local check

Per prompt <id>:
  <id>.tokens.npy   int16 [N, K, frames]  undelayed codes (50 frames/s)
  <id>_<row>.wav    16-bit 32 kHz final clips (not committed)
  <id>.json         scores: clap at every 5 s prefix + final, judges at the final clip.
                    Written last and atomically, so its presence means the prompt is done.
Prefixes are decoded from token prefixes, not cut from the final waveform: that is what a
pruned run would see at its checkpoint.
"""
import argparse, csv, json, os, time, zlib
from pathlib import Path
import numpy as np
import torch
from prune import generate, undelay
from scorers import SR, Clap, Audiobox, CmiRm

ROOT = Path(__file__).resolve().parents[1]
FPS = 50


def load_prompts(files):
    rows = []
    for f in files:
        rows += [(r["id"], r["caption"]) for r in csv.DictReader(open(f), delimiter="\t")]
    return rows


def load_musicgen(size, dev, dtype):
    from transformers import AutoProcessor, MusicgenForConditionalGeneration
    name = f"facebook/musicgen-{size}"
    full = MusicgenForConditionalGeneration.from_pretrained(name).eval()
    full.decoder.to(dev, dtype)          # T5 + EnCodec stay fp32 (fp16 T5 overflows)
    full.audio_encoder.to(dev)
    return full, AutoProcessor.from_pretrained(name)


@torch.inference_mode()
def text_cond(full, proc, prompt):
    inp = proc(text=[prompt], return_tensors="pt")
    h = full.text_encoder(**inp).last_hidden_state
    if hasattr(full, "enc_to_dec_proj"):
        h = full.enc_to_dec_proj(h)
    return h * inp["attention_mask"][..., None]


@torch.inference_mode()
def decode(full, frames):
    """frames (B, T, K) long -> (B, samples) float32 numpy."""
    codes = frames.permute(0, 2, 1)[None].to(full.audio_encoder.device)   # (1, B, K, T)
    wav = full.audio_encoder.decode(codes, audio_scales=[None] * frames.shape[0]).audio_values
    return wav[:, 0].float().cpu().numpy()


def write_wav(path, w):
    from scipy.io import wavfile
    wavfile.write(path, SR, (np.clip(w, -1, 1) * 32767).astype(np.int16))


def run_prompt(pid, prompt, a, full, proc, scorers):
    out = Path(a.out)
    K = full.decoder.config.num_codebooks
    seed = zlib.crc32(pid.encode())                 # order-independent, stable per prompt
    t0 = time.time()
    seq, _, _ = generate(full.decoder, text_cond(full, proc, prompt), a.n,
                         a.seconds * FPS + K - 1, seed=seed)          # +K-1: delay pattern tail
    frames = undelay(seq, K).cpu()                                    # (N, seconds*50, K)
    t_gen = time.time() - t0
    np.save(out / f"{pid}.tokens.npy", frames.permute(0, 2, 1).numpy().astype(np.int16))
    prefixes = sorted(set(range(5, a.seconds, 5)) | {a.seconds})
    scores = {"clap": {}}
    for s in prefixes:
        wav = decode(full, frames[:, : s * FPS])
        scores["clap"][str(s)] = scorers["clap"](wav, prompt)["clap"].tolist()
    for r, w in enumerate(wav):                                        # wav = final clip now
        write_wav(out / f"{pid}_{r}.wav", w)
    for name, sc in scorers.items():
        if name != "clap":
            scores.update({k: v.tolist() for k, v in sc(wav, prompt).items()})
    rec = {"id": pid, "prompt": prompt, "seed": seed, "n": a.n, "seconds": a.seconds,
           "model": f"facebook/musicgen-{a.size}", "dtype": a.dtype,
           "gen_s": round(t_gen, 2), "total_s": round(time.time() - t0, 2), "scores": scores}
    tmp = out / f"{pid}.json.tmp"
    tmp.write_text(json.dumps(rec, indent=1))
    os.replace(tmp, out / f"{pid}.json")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", nargs="+", default=[str(ROOT / "data/prompts/musiccaps_40.tsv"),
                                                   str(ROOT / "data/prompts/redteam_10.tsv")])
    ap.add_argument("--only", nargs="*", help="prompt ids to run")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--seconds", type=int, default=30)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--size", default="small")
    ap.add_argument("--dtype", default=None, help="decoder dtype; default fp16 on CUDA, fp32 on CPU")
    ap.add_argument("--judges", default="audiobox,cmi", help="comma list of audiobox,cmi or 'none'")
    ap.add_argument("--cmi-repo", default="CMI-RewardBench")
    ap.add_argument("--cmi-ckpt", default="CMI-RewardBench/baselines/model/model.safetensors")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a.dtype = a.dtype or ("float16" if dev.type == "cuda" else "float32")
    Path(a.out).mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(a.prompts)
    if a.only:
        prompts = [p for p in prompts if p[0] in a.only]
    prompts = prompts[: a.limit] if a.limit else prompts
    todo = [p for p in prompts if not (Path(a.out) / f"{p[0]}.json").exists()]
    print(f"{len(prompts)} prompts, {len(prompts) - len(todo)} already done, device={dev}", flush=True)
    if not todo:
        return
    full, proc = load_musicgen(a.size, dev, getattr(torch, a.dtype))
    scorers = {"clap": Clap(dev)}
    judges = [] if a.judges == "none" else a.judges.split(",")
    if "audiobox" in judges:
        scorers["audiobox"] = Audiobox()
    if "cmi" in judges:
        scorers["cmi"] = CmiRm(a.cmi_repo, a.cmi_ckpt, str(dev))
    for i, (pid, prompt) in enumerate(todo):
        rec = run_prompt(pid, prompt, a, full, proc, scorers)
        print(f"[{i+1}/{len(todo)}] {pid} gen {rec['gen_s']}s total {rec['total_s']}s "
              f"keys={sorted(rec['scores'])}", flush=True)


if __name__ == "__main__":
    from transformers.utils import logging
    logging.set_verbosity_error()
    main()
