#!/usr/bin/env python3
"""Prepare the OpenSLR + PathNirvana Sinhala corpus for F5-TTS.

The public combined package uses ``sin_*`` IDs for the 3,300 PathNirvana
utterances and numeric IDs for the OpenSLR multi-speaker utterances.  We keep
that distinction in a sidecar report and retain the original Sinhala text.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=Path, default=Path("data/openslr_pathnirvana/file_index.tsv"))
    p.add_argument("--audio-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, default=Path("runs/f5_openslr_pathnirvana"))
    args = p.parse_args()
    audio_root = args.audio_root.resolve()
    rows = []
    missing = []
    with args.index.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for item in reader:
            name, text = item["file_path"].strip(), item["sentence"].strip()
            matches = list(audio_root.rglob(name))
            if not matches or not text:
                missing.append(name)
                continue
            audio = matches[0].resolve()
            source = "pathnirvana" if name.startswith("sin_") else "openslr"
            rows.append((str(audio), text, source, name))
    # Deterministic speaker/source-stratified holdout. PathNirvana remains in
    # both source groups, while numeric OpenSLR IDs are kept disjoint by ID.
    train, val = [], []
    for row in rows:
        name = row[3]
        numeric = name.split("_")[1] if name.startswith("sin_") else name.split("_")[0]
        (val if numeric[-1:] in {"0", "1"} else train).append(row)
    args.output_root.mkdir(parents=True, exist_ok=True)
    for filename, subset in (("train.csv", train), ("val.csv", val), ("all.csv", rows)):
        with (args.output_root / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="|", lineterminator="\n")
            writer.writerow(["audio_file", "text"])
            writer.writerows((r[0], r[1]) for r in subset)
    chars = {char for _, text, _, _ in rows for char in text}
    chars.discard(" ")
    (args.output_root / "vocab.txt").write_text("\n".join([" "] + sorted(chars)) + "\n", encoding="utf-8")
    report = {
        "total": len(rows), "train": len(train), "val": len(val),
        "missing": len(missing), "sources": Counter(r[2] for r in rows),
        "unicode": True, "vocab_size": len(chars) + 1,
    }
    (args.output_root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if missing:
        print("First missing files:", missing[:10])


if __name__ == "__main__":
    main()
