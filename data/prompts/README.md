# Prompts (SPEC 0.5: 50 prompts)

| File | Rows | Source | Licence |
|---|---|---|---|
| `musiccaps_40.tsv` | 40 | [google/MusicCaps](https://huggingface.co/datasets/google/MusicCaps) captions, seed-0 sample of the genre-balanced subset (`is_balanced_subset`) | CC BY-SA 4.0 (Agostinelli et al., 2023). Captions unchanged except whitespace collapsed; this file is shared under the same licence. |
| `redteam_10.tsv` | 10 | Hand-written slow-build prompts (orchestral build-up, post-rock crescendo, ambient intro then drop, solo intro then full band) | MIT, as the repo |

Regenerate with `python torch/make_prompts.py`. The script checks the upstream CSV's
sha256 and rewrites both files byte-for-byte. `ytid` + `start_s` identify the MusicCaps clip
each caption describes.
