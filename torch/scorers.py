"""
Audio scorers for the corpus. Each takes (wavs (B, samples) float32 numpy at 32 kHz, prompt)
and returns {score name: (B,) numpy}.

  Clap      pruner: text-audio cosine, chunked 10 s mean (deterministic, see ../bench.py)
  Audiobox  judge: Audiobox Aesthetics CE/CU/PC/PQ (pip install audiobox_aesthetics)
  CmiRm     judge: CMI-RM alignment + quality (github.com/Haiwen-Xia/CMI-RewardBench)
"""
import sys
import numpy as np
import torch

SR = 32000  # musicgen / EnCodec 32 kHz


class Clap:
    """laion/clap-htsat-unfused sees 10 s max and its processor RANDOMLY crops longer clips
    ("rand_trunc"), so: cut each clip into 10 s chunks, embed every chunk, average per clip.
    Deterministic, and covers the whole clip."""
    def __init__(self, device="cpu", name="laion/clap-htsat-unfused"):
        from transformers import ClapModel, ClapProcessor
        self.dev = device
        self.model = ClapModel.from_pretrained(name).eval().to(device)
        self.proc = ClapProcessor.from_pretrained(name)

    @staticmethod
    def _feat(out):  # transformers 5 may return a ModelOutput instead of a tensor
        return out if hasattr(out, "norm") else out.pooler_output

    @torch.inference_mode()
    def __call__(self, wavs, prompt):
        from scipy.signal import resample_poly
        t = self._feat(self.model.get_text_features(**self.proc.tokenizer([prompt], return_tensors="pt").to(self.dev)))
        t = t / t.norm(dim=-1, keepdim=True)
        W = 480_000                                                   # 10 s at 48 kHz
        chunks, owner = [], []
        for b, w in enumerate(wavs):
            w = resample_poly(w, 3, 2).astype(np.float32)             # CLAP wants 48 kHz
            for s in range(0, max(len(w) - W // 10, 1), W):           # drop tails < 1 s
                chunks.append(w[s : s + W]); owner.append(b)
        x = self.proc.feature_extractor(chunks, sampling_rate=48000, return_tensors="pt").to(self.dev)
        a = self._feat(self.model.get_audio_features(**x))
        a = a / a.norm(dim=-1, keepdim=True)
        owner = torch.tensor(owner, device=self.dev)
        clip = torch.stack([a[owner == b].mean(0) for b in range(len(wavs))])
        return {"clap": ((clip / clip.norm(dim=-1, keepdim=True)) @ t.T)[:, 0].float().cpu().numpy()}


class Audiobox:
    """Audiobox Aesthetics: resamples to 16 kHz and averages 10 s windows internally."""
    AXES = ("CE", "CU", "PC", "PQ")

    def __init__(self, device=None):
        from audiobox_aesthetics.infer import initialize_predictor
        self.pred = initialize_predictor()

    def __call__(self, wavs, prompt=None):  # prompt-free by design
        out = self.pred.forward([{"path": torch.from_numpy(w)[None], "sample_rate": SR} for w in wavs])
        return {f"aes_{k}": np.array([o[k] for o in out], dtype=np.float32) for k in self.AXES}


class CmiRm:
    """CMI-RM from a clone of CMI-RewardBench (repo_dir) + weights from HaiwenXia/CMI-RM.
    T4 has no native bf16, so bf16=False."""
    def __init__(self, repo_dir, checkpoint, device="cuda:0"):
        sys.path.insert(0, f"{repo_dir}/baselines")
        from inference import RewardModelInference
        self.m = RewardModelInference(checkpoint=checkpoint, device=device, bf16=False)
        self.sr = self.m.sr if hasattr(self.m, "sr") else 24000

    def __call__(self, wavs, prompt):
        import torchaudio
        x = torchaudio.functional.resample(torch.from_numpy(np.stack(wavs)), SR, self.sr)
        out = self.m.score_batch([{"audio": w[None], "text": prompt} for w in x], max_dur=30.0)
        out = np.asarray(out, dtype=np.float32)
        return {"cmi_ali": out[:, 0], "cmi_qual": out[:, 1]}
