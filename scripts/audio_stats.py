"""Compute corpus stats (files, duration, sample rates, speakers) for a directory of audio."""
import sys, os, wave, contextlib, subprocess, json
from collections import Counter

root = sys.argv[1]
exts = (".wav", ".flac", ".mp3", ".ogg", ".opus")

def wav_info(path):
    with contextlib.closing(wave.open(path, "rb")) as w:
        return w.getframerate(), w.getnframes() / float(w.getframerate())

def ffprobe_info(path):
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", path],
        capture_output=True, text=True).stdout
    s = json.loads(out)["streams"][0]
    return int(s["sample_rate"]), float(s.get("duration") or 0)

files = []
for dp, _, fns in os.walk(root):
    for fn in fns:
        if fn.lower().endswith(exts):
            files.append(os.path.join(dp, fn))

total = 0.0
rates = Counter()
bad = 0
for i, f in enumerate(files):
    try:
        if f.lower().endswith(".wav"):
            try:
                r, d = wav_info(f)
            except Exception:
                r, d = ffprobe_info(f)
        else:
            r, d = ffprobe_info(f)
        rates[r] += 1
        total += d
    except Exception:
        bad += 1

print(f"root: {root}")
print(f"files: {len(files)}  unreadable: {bad}")
print(f"total_hours: {total/3600:.2f}")
print(f"sample_rates: {dict(rates)}")
if files:
    print(f"avg_clip_sec: {total/max(1,len(files)-bad):.2f}")
