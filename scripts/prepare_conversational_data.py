"""Validate and prepare a rights-cleared conversational Sinhala TTS manifest.

This utility operates on audio that the team already has permission to use. It
does not download podcast/video media, scrape platforms, or cut audio. Upstream
segmentation and transcript correction can be performed with the team's chosen
tools; this script is the reproducible final gate before TTS training.

Input ``segments.csv`` columns:

    utterance_id,source_id,episode_id,speaker_id,audio_path,start_sec,end_sec,
    text,style,transcript_status,license_id

Input ``rights.csv`` columns:

    license_id,source_id,permission_to_train,permission_to_release_model,
    permission_to_release_audio,consent_scope

The generated metadata is pipe-delimited and contains only accepted rows. A
JSON report records rejection reasons so filtering is auditable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
import wave
from collections import Counter
from pathlib import Path


SEGMENT_COLUMNS = {
    "utterance_id", "source_id", "episode_id", "speaker_id", "audio_path",
    "start_sec", "end_sec", "text", "style", "transcript_status", "license_id",
}
RIGHTS_COLUMNS = {
    "license_id", "source_id", "permission_to_train",
    "permission_to_release_model", "permission_to_release_audio", "consent_scope",
}
APPROVED_TRANSCRIPT_STATUSES = {
    "manual_verified", "manually_verified", "forced_aligned_manual_verified",
}
TRUE_VALUES = {"1", "true", "yes", "y"}
DEFAULT_STYLES = {
    "neutral", "conversational", "question", "empathetic", "sad", "excited",
    "emphasis", "encouragement", "unknown",
}


def read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = required - columns
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        return [{key: (value or "").strip() for key, value in row.items()} for row in reader]


def is_true(value: str) -> bool:
    return value.strip().lower() in TRUE_VALUES


def normalized_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).strip()
    return re.sub(r"\s+", " ", text)


def wav_duration(path: Path) -> float | None:
    if path.suffix.lower() != ".wav":
        return None
    try:
        with wave.open(str(path), "rb") as handle:
            if handle.getframerate() <= 0:
                return None
            return handle.getnframes() / handle.getframerate()
    except (OSError, wave.Error):
        return None


def assign_split(row: dict[str, str], split_unit: str) -> str:
    if split_unit == "source":
        group = row["source_id"]
    elif split_unit == "speaker":
        group = row["speaker_id"]
    else:
        group = f'{row["source_id"]}::{row["episode_id"]}'
    bucket = int(hashlib.sha256(group.encode("utf-8")).hexdigest()[:8], 16) % 10
    return "test" if bucket == 0 else "dev" if bucket == 1 else "train"


def prepare(
    segments: list[dict[str, str]],
    rights: list[dict[str, str]],
    *,
    root: Path,
    min_duration: float,
    max_duration: float,
    split_unit: str,
    allowed_styles: set[str],
    require_audio_files: bool,
    require_model_release_permission: bool,
) -> tuple[list[dict[str, str]], dict]:
    rights_by_key = {(row["license_id"], row["source_id"]): row for row in rights}
    accepted: list[dict[str, str]] = []
    rejected = Counter()
    seen_ids: set[str] = set()
    seen_text: set[str] = set()

    for row in segments:
        reason = None
        utterance_id = row["utterance_id"]
        if not utterance_id:
            reason = "missing_utterance_id"
        elif utterance_id in seen_ids:
            reason = "duplicate_utterance_id"
        elif not row["source_id"] or not row["episode_id"] or not row["speaker_id"]:
            reason = "missing_provenance_or_speaker"
        elif not row["text"]:
            reason = "empty_transcript"
        elif row["transcript_status"].lower() not in APPROVED_TRANSCRIPT_STATUSES:
            reason = "transcript_not_manually_verified"
        elif row["style"].lower() not in allowed_styles:
            reason = "unknown_style"
        else:
            rights_row = rights_by_key.get((row["license_id"], row["source_id"]))
            if rights_row is None:
                reason = "missing_rights_record"
            elif not is_true(rights_row["permission_to_train"]):
                reason = "training_permission_missing"
            elif require_model_release_permission and not is_true(rights_row["permission_to_release_model"]):
                reason = "model_release_permission_missing"
            elif not rights_row["consent_scope"]:
                reason = "consent_scope_missing"

        try:
            start = float(row["start_sec"])
            end = float(row["end_sec"])
            segment_duration = end - start
        except (TypeError, ValueError):
            start = end = segment_duration = -1.0
            if reason is None:
                reason = "invalid_time_range"

        if reason is None and not (0 <= start < end):
            reason = "invalid_time_range"
        if reason is None and not (min_duration <= segment_duration <= max_duration):
            reason = "duration_out_of_range"

        audio_path = Path(row["audio_path"])
        resolved_audio = audio_path if audio_path.is_absolute() else root / audio_path
        if reason is None and require_audio_files and not resolved_audio.is_file():
            reason = "audio_file_missing"
        if reason is None and require_audio_files and resolved_audio.suffix.lower() != ".wav":
            reason = "audio_is_not_wav"
        if reason is None and require_audio_files:
            full_duration = wav_duration(resolved_audio)
            if full_duration is None or end > full_duration + 0.05:
                reason = "segment_exceeds_audio"

        text = normalized_text(row["text"])
        text_key = text.casefold()
        if reason is None and text_key in seen_text:
            reason = "duplicate_text"

        if reason is not None:
            rejected[reason] += 1
            continue

        seen_ids.add(utterance_id)
        seen_text.add(text_key)
        accepted.append({
            "id": utterance_id,
            "text": text,
            "audio_path": str(audio_path),
            "speaker_id": row["speaker_id"],
            "style": row["style"].lower(),
            "split": assign_split(row, split_unit),
            "source_id": row["source_id"],
            "episode_id": row["episode_id"],
            "license_id": row["license_id"],
            "start_sec": f"{start:.3f}",
            "end_sec": f"{end:.3f}",
            "duration_sec": f"{segment_duration:.3f}",
        })

    report = {
        "input_count": len(segments),
        "accepted_count": len(accepted),
        "rejected_count": sum(rejected.values()),
        "rejections": dict(sorted(rejected.items())),
        "accepted_by_split": dict(Counter(row["split"] for row in accepted)),
        "accepted_by_style": dict(Counter(row["style"] for row in accepted)),
        "accepted_by_speaker": dict(Counter(row["speaker_id"] for row in accepted)),
        "split_unit": split_unit,
        "min_duration": min_duration,
        "max_duration": max_duration,
        "require_audio_files": require_audio_files,
        "require_model_release_permission": require_model_release_permission,
    }
    return accepted, report


def write_metadata(rows: list[dict[str, str]], path: Path) -> None:
    columns = [
        "id", "text", "audio_path", "speaker_id", "style", "split", "source_id",
        "episode_id", "license_id", "start_sec", "end_sec", "duration_sec",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="|", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--rights", type=Path, required=True)
    parser.add_argument("--output-metadata", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("."), help="base for relative audio paths")
    parser.add_argument("--min-duration", type=float, default=2.0)
    parser.add_argument("--max-duration", type=float, default=12.0)
    parser.add_argument("--split-unit", choices=("episode", "source", "speaker"), default="episode")
    parser.add_argument(
        "--allowed-styles", default=",".join(sorted(DEFAULT_STYLES)),
        help="comma-separated style labels accepted by the manifest",
    )
    parser.add_argument(
        "--skip-audio-check", action="store_true",
        help="only validate metadata paths; useful before audio is copied locally",
    )
    parser.add_argument(
        "--allow-private-only", action="store_true",
        help="allow training when the rights record does not permit model release",
    )
    args = parser.parse_args()
    if args.min_duration <= 0 or args.max_duration < args.min_duration:
        raise ValueError("duration bounds are invalid")

    segments = read_csv(args.segments, SEGMENT_COLUMNS)
    rights = read_csv(args.rights, RIGHTS_COLUMNS)
    rows, report = prepare(
        segments, rights, root=args.root.resolve(), min_duration=args.min_duration,
        max_duration=args.max_duration, split_unit=args.split_unit,
        allowed_styles={item.strip().lower() for item in args.allowed_styles.split(",") if item.strip()},
        require_audio_files=not args.skip_audio_check,
        require_model_release_permission=not args.allow_private_only,
    )
    args.output_metadata.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    write_metadata(rows, args.output_metadata)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f'accepted {report["accepted_count"]}/{report["input_count"]} utterances')
    print(f"metadata: {args.output_metadata}")
    print(f"report: {args.output_report}")


if __name__ == "__main__":
    main()
