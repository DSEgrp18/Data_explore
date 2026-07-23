"""Prepare a rights-cleared target-speaker recording pack for Coqui VITS.

Expected input:

    prompts.csv: id,style,text
    recordings_dir/<id>.wav

The recordings must be mono, 16-bit, 22050 Hz WAV files. The script keeps
the original recordings untouched, creates a wavs symlink in the output
directory, and writes train/validation metadata after checking duration,
transcript vocabulary, duplicates, and missing recordings.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import wave
from collections import Counter
from pathlib import Path


def read_prompts(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "style", "text"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        return [{key: (value or "").strip() for key, value in row.items()} for row in reader]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wav_info(path: Path) -> tuple[float, int, int, int] | None:
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            duration = handle.getnframes() / max(rate, 1)
            return duration, rate, channels, width
    except (OSError, wave.Error):
        return None


def split_for_validation(identifier: str, validation_fraction: float) -> str:
    value = int(hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if value < validation_fraction else "train"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--recordings-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--romanizer", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--speaker", required=True)
    parser.add_argument("--min-duration", type=float, default=1.5)
    parser.add_argument("--max-duration", type=float, default=12.0)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    args = parser.parse_args()

    if not 0 < args.validation_fraction < 1:
        raise ValueError("validation-fraction must be between 0 and 1")
    if args.min_duration <= 0 or args.max_duration < args.min_duration:
        raise ValueError("duration bounds are invalid")
    if not args.recordings_dir.is_dir():
        raise FileNotFoundError(args.recordings_dir)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from sinhala_text_normalizer import clean_input

    romanizer = load_module(args.romanizer, "target_speaker_romanizer")
    config = json.loads(args.base_config.read_text(encoding="utf-8"))
    characters = set(config["characters"]["characters"])
    prompts = read_prompts(args.prompts)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    wav_link = output_root / "wavs"
    if wav_link.exists() or wav_link.is_symlink():
        if wav_link.is_symlink() and wav_link.resolve() == args.recordings_dir.resolve():
            pass
        else:
            raise FileExistsError(f"refusing to replace existing {wav_link}")
    else:
        wav_link.symlink_to(args.recordings_dir.resolve(), target_is_directory=True)

    accepted: list[dict[str, str]] = []
    rejected = Counter()
    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    for row in prompts:
        identifier = row["id"]
        reason = None
        if not identifier:
            reason = "missing_id"
        elif identifier in seen_ids:
            reason = "duplicate_id"
        elif not row["text"]:
            reason = "empty_text"
        else:
            normalized = clean_input(row["text"], normalize_numbers=True)
            romanized = romanizer.sinhala_to_roman(normalized)
            unknown = sorted(set(romanized) - characters)
            if unknown:
                reason = "unknown_model_character:" + "".join(unknown)
            elif normalized.casefold() in seen_text:
                reason = "duplicate_text"

        recording = args.recordings_dir / f"{identifier}.wav"
        info = wav_info(recording)
        if reason is None and info is None:
            reason = "missing_or_invalid_wav"
        if reason is None:
            duration, rate, channels, width = info
            if rate != 22050:
                reason = "wrong_sample_rate"
            elif channels != 1:
                reason = "not_mono"
            elif width != 2:
                reason = "not_16_bit"
            elif not args.min_duration <= duration <= args.max_duration:
                reason = "duration_out_of_range"

        if reason is not None:
            rejected[reason] += 1
            continue

        seen_ids.add(identifier)
        seen_text.add(normalized.casefold())
        split = split_for_validation(identifier, args.validation_fraction)
        accepted.append({
            "audio_file": f"wavs/{identifier}.wav",
            "text": romanized,
            "speaker_name": args.speaker,
            "emotion_name": row["style"] or "neutral",
            "id": identifier,
            "normalized_text": normalized,
            "split": split,
            "duration_sec": f"{info[0]:.3f}",
        })

    columns = ["audio_file", "text", "speaker_name", "emotion_name"]
    for split in ("train", "val"):
        rows = [row for row in accepted if row["split"] == split]
        with (output_root / f"metadata_{split}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, delimiter="|", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    report = {
        "speaker": args.speaker,
        "input_count": len(prompts),
        "accepted_count": len(accepted),
        "rejected_count": sum(rejected.values()),
        "rejections": dict(sorted(rejected.items())),
        "accepted_by_split": dict(Counter(row["split"] for row in accepted)),
        "accepted_by_style": dict(Counter(row["emotion_name"] for row in accepted)),
        "sample_rate": 22050,
        "min_duration": args.min_duration,
        "max_duration": args.max_duration,
        "recordings_dir": str(args.recordings_dir.resolve()),
    }
    (output_root / "preparation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f'accepted {report["accepted_count"]}/{report["input_count"]} recordings')
    print(f"metadata: {output_root}/metadata_train.csv")
    print(f"report: {output_root}/preparation_report.json")


if __name__ == "__main__":
    main()
