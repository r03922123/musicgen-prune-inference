"""
Build the prompt set: 40 MusicCaps captions + 10 slow-build red-team prompts.

  python torch/make_prompts.py     # rewrites data/prompts/*.tsv byte-for-byte

MusicCaps (google/MusicCaps, CC BY-SA 4.0): sampled with a fixed seed from the
genre-balanced 1k subset, from a CSV pinned by sha256.
"""
import csv, hashlib, io, random, urllib.request
from pathlib import Path

URL = "https://huggingface.co/datasets/google/MusicCaps/resolve/main/musiccaps-public.csv"
SHA256 = "8cc109ac507b5e5aed8ea73a5c8f897a9d0793f7397427e000adf4c05f6021c7"
OUT = Path(__file__).resolve().parents[1] / "data" / "prompts"
N_MUSICCAPS, SEED = 40, 0

# Slow-build pieces: the payoff comes late, so an early checkpoint may judge only the intro.
REDTEAM = [
    "orchestral build-up, starts with a quiet solo cello, strings and brass join, ends in a loud full-orchestra climax with timpani",
    "cinematic trailer music, soft piano at first, rising string ostinato, huge percussion hits in the second half",
    "post-rock crescendo, clean delayed guitar arpeggios, drums enter slowly, builds into a wall of distorted guitars",
    "slow-building post-rock, ambient guitar swells for a long time, then bass and drums, ending in an intense loud climax",
    "ambient intro with soft synth pads and field recordings, then a sudden drop into a heavy electronic beat with deep bass",
    "minimal techno that starts with only a hi-hat and slowly layers kick, bass and synths until a big drop",
    "atmospheric drone intro that leads into an energetic drum and bass track after a long riser",
    "solo acoustic guitar intro, then the full band comes in with drums, bass and electric guitar for an upbeat rock chorus",
    "lone piano ballad intro, joined later by drums, bass and a choir, ending in a powerful gospel finale",
    "solo saxophone opening, unaccompanied and slow, then a swinging big band joins at full volume",
]


def main():
    raw = urllib.request.urlopen(URL).read()
    assert hashlib.sha256(raw).hexdigest() == SHA256, "MusicCaps CSV changed upstream"
    rows = [r for r in csv.DictReader(io.StringIO(raw.decode())) if r["is_balanced_subset"] == "True"]
    rows.sort(key=lambda r: r["ytid"])
    picked = sorted(random.Random(SEED).sample(rows, N_MUSICCAPS), key=lambda r: r["ytid"])
    OUT.mkdir(parents=True, exist_ok=True)
    clean = lambda s: " ".join(s.split())  # no tabs / newlines inside a TSV field
    with open(OUT / "musiccaps_40.tsv", "w") as f:
        f.write("id\tytid\tstart_s\tcaption\n")
        for r in picked:
            f.write(f"mc_{r['ytid']}\t{r['ytid']}\t{r['start_s']}\t{clean(r['caption'])}\n")
    with open(OUT / "redteam_10.tsv", "w") as f:
        f.write("id\tcaption\n")
        for i, p in enumerate(REDTEAM):
            f.write(f"rt_{i:02d}\t{p}\n")
    print(f"wrote {len(picked)} MusicCaps + {len(REDTEAM)} red-team prompts to {OUT}")


if __name__ == "__main__":
    main()
