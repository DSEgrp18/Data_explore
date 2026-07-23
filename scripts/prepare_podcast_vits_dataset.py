"""Convert the downloaded Sinhala caption dataset to a Coqui VITS manifest.

The source release stores Sinhala text in LJSpeech-style metadata while the
Dialog/UoM VITS checkpoint expects romanized text. This script preserves the
source split, applies the project's existing romanizer, drops duplicate or
overlong clips, and writes the ``coqui`` formatter format used by the GPU
fine-tuning script.

It assumes the audio has already been obtained under a valid dataset license.
It does not download media or alter audio.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
import wave
from collections import Counter
from pathlib import Path


def load_romanizer(path: Path):
    spec = importlib.util.spec_from_file_location("project_romanizer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load romanizer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.sinhala_to_roman


def duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def read_source(path: Path) -> list[tuple[str, str]]:
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, delimiter="|"):
            if len(row) < 2:
                continue
            rows.append((row[0].strip(), row[1].strip()))
    return rows


def convert_split(
    source_metadata: Path,
    audio_root: Path,
    output_metadata: Path,
    romanize,
    *,
    min_seconds: float,
    max_seconds: float,
    limit: int | None,
    speaker_name: str,
    clean_input,
    allowed_characters: set[str],
) -> dict:
    accepted = []
    rejected = Counter()
    seen_text = set()
    for item_id, sinhala_text in read_source(source_metadata):
        if limit is not None and len(accepted) >= limit:
            break
        if not item_id or not sinhala_text:
            rejected["empty_metadata"] += 1
            continue
        audio_path = audio_root / "wavs" / f"{item_id}.wav"
        if not audio_path.is_file():
            rejected["missing_audio"] += 1
            continue
        try:
            duration = duration_seconds(audio_path)
        except (OSError, wave.Error, ZeroDivisionError):
            rejected["invalid_audio"] += 1
            continue
        if not min_seconds <= duration <= max_seconds:
            rejected["duration_out_of_range"] += 1
            continue
        text = clean_input(sinhala_text, normalize_numbers=True)
        text_key = text.casefold()
        if text_key in seen_text:
            rejected["duplicate_text"] += 1
            continue
        romanized = romanize(text).strip()
        if not romanized:
            rejected["empty_romanization"] += 1
            continue
        unknown = {char for char in romanized if char not in allowed_characters}
        if unknown:
            rejected["unknown_model_character"] += 1
            continue
        seen_text.add(text_key)
        accepted.append({
            "audio_file": f"wavs/{item_id}.wav",
            "text": romanized,
            "speaker_name": speaker_name,
            "emotion_name": "conversational",
            "source_text": text,
            "duration_sec": round(duration, 3),
        })

    output_metadata.parent.mkdir(parents=True, exist_ok=True)
    with output_metadata.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["audio_file", "text", "speaker_name", "emotion_name"],
            delimiter="|",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames} for row in accepted)
    return {
        "source": str(source_metadata),
        "output": str(output_metadata),
        "accepted": len(accepted),
        "rejected": sum(rejected.values()),
        "rejections": dict(sorted(rejected.items())),
        "total_audio_seconds": round(sum(row["duration_sec"] for row in accepted), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True, help="directory containing metadata and wavs/")
    parser.add_argument("--romanizer", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--speaker-name", default="podcast_host")
    parser.add_argument("--min-seconds", type=float, default=2.0)
    parser.add_argument("--max-seconds", type=float, default=12.0)
    parser.add_argument("--limit", type=int, default=None, help="limit each split for a smoke test")
    args = parser.parse_args()
    if args.min_seconds <= 0 or args.max_seconds < args.min_seconds:
        raise ValueError("invalid duration bounds")

    romanize = load_romanizer(args.romanizer)
    model_config = json.loads(args.model_config.read_text(encoding="utf-8"))
    allowed_characters = set(model_config["characters"]["characters"])
    scripts_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(scripts_root))
    from sinhala_text_normalizer import clean_input
    reports = []
    for split in ("train", "val"):
        source = args.dataset_root / f"metadata_{split}.csv"
        if not source.is_file():
            raise FileNotFoundError(source)
        reports.append(convert_split(
            source,
            args.dataset_root,
            args.output_root / f"metadata_{split}.csv",
            romanize,
            min_seconds=args.min_seconds,
            max_seconds=args.max_seconds,
            limit=args.limit if split == "train" else None,
            speaker_name=args.speaker_name,
            clean_input=clean_input,
            allowed_characters=allowed_characters,
        ))

    payload = {
        "dataset_root": str(args.dataset_root),
        "output_root": str(args.output_root),
        "romanizer": str(args.romanizer),
        "reports": reports,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    report_path = args.output_root / "preparation_report.json"
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for report in reports:
        print(f'{Path(report["source"]).name}: accepted {report["accepted"]}, rejected {report["rejected"]}')
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
