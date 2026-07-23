"""Derive controllable prosody-style labels from Sinhala WAV recordings.

This does not claim to detect human emotion. It creates repeatable acoustic
style buckets from pitch, energy, speaking rate, and silence so a from-scratch
multi-style VITS model can learn naturalness controls without emotion labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import librosa
import numpy as np


def read_metadata(path: Path, split: str) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="|"))
    for row in rows:
        row["split"] = split
    return rows


def extract_features(path: Path, text: str) -> dict[str, float]:
    audio, sample_rate = librosa.load(path, sr=None, mono=True)
    if len(audio) == 0 or sample_rate <= 0:
        raise ValueError("empty audio")
    duration = len(audio) / sample_rate
    rms = librosa.feature.rms(y=audio, frame_length=1024, hop_length=256)[0]
    rms_db = float(20.0 * np.log10(max(float(np.median(rms)), 1e-7)))
    silence_ratio = float(np.mean(rms < max(float(np.max(rms)) * 0.08, 1e-5)))
    f0 = librosa.yin(
        audio,
        fmin=70.0,
        fmax=min(400.0, sample_rate / 2.0 - 1.0),
        sr=sample_rate,
        frame_length=1024,
        hop_length=256,
    )
    voiced = f0[np.isfinite(f0) & (f0 > 70.0) & (f0 < 400.0)]
    median_f0 = float(np.median(voiced)) if voiced.size else 0.0
    f0_spread = float(np.percentile(voiced, 75) - np.percentile(voiced, 25)) if voiced.size else 0.0
    return {
        "duration_sec": duration,
        "char_rate": len(text) / max(duration, 1e-6),
        "rms_db": rms_db,
        "silence_ratio": silence_ratio,
        "median_f0": median_f0,
        "f0_spread": f0_spread,
    }


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def assign_style(row: dict[str, str], thresholds: dict[str, float]) -> str:
    rate = float(row["char_rate"])
    energy = float(row["rms_db"])
    spread = float(row["f0_spread"])
    if rate >= thresholds["rate_hi"] and energy >= thresholds["energy_mid"]:
        return "energetic"
    if spread >= thresholds["spread_hi"]:
        return "expressive"
    if rate <= thresholds["rate_lo"] and energy <= thresholds["energy_mid"]:
        return "calm"
    if energy <= thresholds["energy_lo"]:
        return "intimate"
    if rate >= thresholds["rate_mid"]:
        return "lively"
    return "conversational"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--min-duration", type=float, default=1.5)
    parser.add_argument("--max-duration", type=float, default=12.0)
    args = parser.parse_args()
    if args.min_duration <= 0 or args.max_duration < args.min_duration:
        raise ValueError("duration bounds are invalid")

    source_rows = []
    for split in ("train", "val"):
        source_rows.extend(read_metadata(args.dataset_root / f"metadata_{split}.csv", split))
    accepted = []
    rejected = Counter()
    for row in source_rows:
        path = args.dataset_root / row["audio_file"]
        try:
            features = extract_features(path, row["text"])
        except (OSError, ValueError, RuntimeError) as exc:
            rejected[type(exc).__name__] += 1
            continue
        if not args.min_duration <= features["duration_sec"] <= args.max_duration:
            rejected["duration_out_of_range"] += 1
            continue
        accepted.append({**row, **{key: f"{value:.6f}" for key, value in features.items()}})

    if not accepted:
        raise RuntimeError("no recordings passed prosody extraction")
    thresholds = {
        "rate_lo": percentile([float(row["char_rate"]) for row in accepted], 25),
        "rate_mid": percentile([float(row["char_rate"]) for row in accepted], 50),
        "rate_hi": percentile([float(row["char_rate"]) for row in accepted], 75),
        "energy_lo": percentile([float(row["rms_db"]) for row in accepted], 35),
        "energy_mid": percentile([float(row["rms_db"]) for row in accepted], 50),
        "spread_hi": percentile([float(row["f0_spread"]) for row in accepted], 75),
    }
    for row in accepted:
        row["style"] = assign_style(row, thresholds)
        row["speaker_name"] = row["style"]
        row["emotion_name"] = row["style"]

    args.output_root.mkdir(parents=True, exist_ok=True)
    output_wavs = args.output_root / "wavs"
    source_wavs = args.dataset_root / "wavs"
    if not source_wavs.exists():
        raise FileNotFoundError(source_wavs)
    if output_wavs.exists() or output_wavs.is_symlink():
        if not output_wavs.is_symlink() or output_wavs.resolve() != source_wavs.resolve():
            raise FileExistsError(f"refusing to replace existing {output_wavs}")
    else:
        output_wavs.symlink_to(source_wavs.resolve(), target_is_directory=True)
    columns = ["audio_file", "text", "speaker_name", "emotion_name"]
    for split in ("train", "val"):
        with (args.output_root / f"metadata_{split}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, delimiter="|", lineterminator="\n")
            writer.writeheader()
            writer.writerows(
                {key: row[key] for key in columns}
                for row in accepted if row["split"] == split
            )
    feature_columns = [
        "audio_file", "text", "split", "style", "duration_sec", "char_rate",
        "rms_db", "silence_ratio", "median_f0", "f0_spread",
    ]
    with (args.output_root / "prosody_features.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=feature_columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row[key] for key in feature_columns} for row in accepted)
    report = {
        "input_count": len(source_rows),
        "accepted_count": len(accepted),
        "rejections": dict(sorted(rejected.items())),
        "style_counts": dict(Counter(row["style"] for row in accepted)),
        "thresholds": thresholds,
        "note": "Styles are acoustic proxies, not human emotion annotations.",
    }
    (args.output_root / "prosody_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f'accepted {len(accepted)}/{len(source_rows)} recordings')
    print("styles:", dict(Counter(row["style"] for row in accepted)))
    print(f"output: {args.output_root}")


if __name__ == "__main__":
    main()
