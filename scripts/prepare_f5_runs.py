"""Expand an evaluation manifest into deterministic F5-TTS experiment runs.

This script is deliberately dependency-free. It prepares a run sheet now; model
inference can be attached after the checkpoint and dataset are available.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path


REQUIRED_COLUMNS = {"id", "category", "text", "prompt_audio", "prompt_text"}


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"manifest is missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    if not rows:
        raise ValueError("manifest contains no evaluation items")
    ids = [row["id"].strip() for row in rows]
    if any(not item_id for item_id in ids):
        raise ValueError("every item must have a non-empty id")
    if len(ids) != len(set(ids)):
        raise ValueError("manifest item ids must be unique")
    return rows


def make_runs(manifest: list[dict[str, str]], config: dict) -> list[dict]:
    runs = []
    for item in manifest:
        for condition in config["conditions"]:
            for seed in config["fixed"]["seeds"]:
                run_id = f'{item["id"]}__{condition["id"]}__s{seed}'
                runs.append(
                    {
                        "run_id": run_id,
                        "item_id": item["id"],
                        "category": item["category"],
                        "text": item["text"],
                        "prompt_audio": item["prompt_audio"],
                        "prompt_text": item["prompt_text"],
                        "condition": condition["id"],
                        "nfe": condition["nfe"],
                        "sway_coefficient": condition["sway_coefficient"],
                        "normalize": condition["normalize"],
                        "seed": seed,
                        "output_audio": f'outputs/{run_id}.wav',
                    }
                )
    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    runs = make_runs(manifest, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": config["name"],
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "item_count": len(manifest),
        "run_count": len(runs),
        "runs": runs,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"prepared {len(runs)} runs from {len(manifest)} items: {args.output}")


if __name__ == "__main__":
    main()
