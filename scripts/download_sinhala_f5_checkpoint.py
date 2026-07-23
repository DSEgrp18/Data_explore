#!/usr/bin/env python3
"""Download the gated Sinhala F5-TTS checkpoint after HF access is granted.

Use HF_TOKEN in the environment; do not put tokens in shell history or source
files. The account must first accept the model's access conditions online.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO = "tharindumihi/tts-si-F5-TTS"
FILES = (
    "ckpts/f5_TTS/model_230000_reduced.pt",
    "ckpts/f5_TTS/vocab.txt",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("runs/f5_sinhala_checkpoint"))
    parser.add_argument("--token", default=None, help="Optional token; HF_TOKEN environment variable is preferred")
    args = parser.parse_args()
    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise SystemExit(
            "No Hugging Face token found. Accept access at "
            "https://huggingface.co/tharindumihi/tts-si-F5-TTS, then export HF_TOKEN."
        )
    out = args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        downloaded = hf_hub_download(repo_id=REPO, filename=filename, token=token)
        target = out / Path(filename).name
        target.write_bytes(Path(downloaded).read_bytes())
        print(target)


if __name__ == "__main__":
    main()
