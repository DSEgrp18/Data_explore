"""Generate a challenge manifest with any Coqui VITS checkpoint on CUDA."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import wave
from pathlib import Path

import torch

CPU_TTS_SITE_PACKAGES = Path("/home/kusal/Documents/DSE_project_grp18/tts_demo/venv/lib/python3.12/site-packages")
if str(CPU_TTS_SITE_PACKAGES) not in sys.path:
    sys.path.append(str(CPU_TTS_SITE_PACKAGES))


def duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--romanizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raw-text", action="store_true", help="do not normalize numbers before synthesis")
    parser.add_argument("--length-scale", type=float, default=1.0, help="duration multiplier; >1 slows speech")
    parser.add_argument("--speaker-name", default="", help="speaker/style embedding name for a multi-speaker checkpoint")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    from TTS.utils.synthesizer import Synthesizer
    import importlib.util
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from sinhala_text_normalizer import clean_input

    spec = importlib.util.spec_from_file_location("project_romanizer", args.romanizer)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load romanizer: {args.romanizer}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    romanize = module.sinhala_to_roman

    args.output_dir.mkdir(parents=True, exist_ok=True)
    synth = Synthesizer(
        tts_checkpoint=str(args.checkpoint),
        tts_config_path=str(args.config),
        use_cuda=True,
    )
    if args.length_scale <= 0:
        raise ValueError("length scale must be positive")
    synth.tts_model.length_scale = args.length_scale
    results = []
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            normalized_text = row["text"] if args.raw_text else clean_input(row["text"], normalize_numbers=True)
            romanized = romanize(normalized_text)
            output = args.output_dir / f'{row["id"]}.wav'
            started = time.perf_counter()
            wav = synth.tts(romanized, speaker_name=args.speaker_name)
            synthesis_seconds = time.perf_counter() - started
            synth.save_wav(wav, str(output))
            audio_seconds = duration(output)
            results.append({
                **row,
                "normalized_text": normalized_text,
                "romanized": romanized,
                "audio_path": str(output),
                "audio_seconds": round(audio_seconds, 4),
                "synthesis_seconds": round(synthesis_seconds, 4),
                "rtf": round(synthesis_seconds / max(audio_seconds, 1e-6), 4),
            })
            print(f'{row["id"]}: {audio_seconds:.2f}s audio, RTF={synthesis_seconds / max(audio_seconds, 1e-6):.3f}')
    payload = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "device": torch.cuda.get_device_name(0),
        "length_scale": args.length_scale,
        "speaker_name": args.speaker_name,
        "item_count": len(results),
        "mean_rtf": round(sum(row["rtf"] for row in results) / max(len(results), 1), 4),
        "items": results,
    }
    result_path = args.output_dir / "results.json"
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"results: {result_path}")


if __name__ == "__main__":
    main()
