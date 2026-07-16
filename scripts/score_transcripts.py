"""Compute dependency-free CER/WER from reference and ASR transcript columns."""

import argparse
import csv
import re
import unicodedata
from pathlib import Path


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).strip().lower()
    return " ".join(re.findall(r"[\w\u0D80-\u0DFF]+", text, flags=re.UNICODE))


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, ref_token in enumerate(reference, start=1):
        current = [i]
        for j, hyp_token in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (ref_token != hyp_token),
                )
            )
        previous = current
    return previous[-1]


def error_rate(reference: list[str], hypothesis: list[str]) -> tuple[int, int]:
    return edit_distance(reference, hypothesis), len(reference)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, help="CSV with reference and hypothesis columns")
    args = parser.parse_args()
    totals = {"char_errors": 0, "chars": 0, "word_errors": 0, "words": 0}
    with args.results.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"reference", "hypothesis"}.issubset(reader.fieldnames or []):
            raise ValueError("results CSV needs reference and hypothesis columns")
        for row in reader:
            ref = normalize_text(row["reference"])
            hyp = normalize_text(row["hypothesis"])
            errors, count = error_rate(list(ref.replace(" ", "")), list(hyp.replace(" ", "")))
            totals["char_errors"] += errors
            totals["chars"] += count
            errors, count = error_rate(ref.split(), hyp.split())
            totals["word_errors"] += errors
            totals["words"] += count
    cer = totals["char_errors"] / max(1, totals["chars"])
    wer = totals["word_errors"] / max(1, totals["words"])
    print(f"CER: {cer:.4f} ({totals['char_errors']}/{totals['chars']})")
    print(f"WER: {wer:.4f} ({totals['word_errors']}/{totals['words']})")


if __name__ == "__main__":
    main()
