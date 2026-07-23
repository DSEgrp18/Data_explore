#!/usr/bin/env python3
"""Build F5-TTS pipe-delimited manifests while preserving Sinhala Unicode text.

The official F5-TTS dataset preparation script expects ``audio_file|text`` with
absolute audio paths.  The source corpus contains the original Sinhala text in
its three-column metadata; this script deliberately does not use the earlier
romanized VITS metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_rows(path: Path, wav_dir: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, delimiter="|"):
            if len(row) < 2:
                continue
            stem, text = row[0].strip(), row[1].strip()
            audio = (wav_dir / f"{stem}.wav").resolve()
            if audio.is_file() and text:
                rows.append((str(audio), text))
    return rows


def write_manifest(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="|", lineterminator="\n")
        writer.writerow(["audio_file", "text"])
        writer.writerows(rows)


def write_vocab(path: Path, rows: list[tuple[str, str]]) -> None:
    # F5's custom tokenizer reserves index 0 for the space/unknown symbol.
    chars = {char for _, text in rows for char in text}
    chars.discard(" ")
    path.write_text("\n".join([" "] + sorted(chars)) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    source = args.source_root.resolve()
    wav_dir = source / "wavs"
    train = read_rows(source / "metadata_train.csv", wav_dir)
    val = read_rows(source / "metadata_val.csv", wav_dir)
    all_rows = train + val
    write_manifest(args.output_root / "train.csv", train)
    write_manifest(args.output_root / "val.csv", val)
    write_manifest(args.output_root / "all.csv", all_rows)
    write_vocab(args.output_root / "vocab.txt", all_rows)
    report = {
        "source_root": str(source),
        "train_items": len(train),
        "val_items": len(val),
        "total_items": len(all_rows),
        "text_script": "Sinhala Unicode (not romanized)",
        "audio_paths": "absolute",
        "vocab_size": len({char for _, text in all_rows for char in text} - {" "}) + 1,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
