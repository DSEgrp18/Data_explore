"""Run the local Dialog/UoM Sinhala VITS baseline on a CSV challenge set."""

import argparse
import csv
import importlib.util
import json
import time
import wave
from pathlib import Path


def load_romanizer(path: Path):
    spec = importlib.util.spec_from_file_location("dialoglk_romanizer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load romanizer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.sinhala_to_roman


def audio_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    from TTS.utils.synthesizer import Synthesizer

    checkpoint = args.model_dir / "Nipunika_210000.pth"
    config = args.model_dir / "Nipunika_config.json"
    romanize = load_romanizer(args.model_dir / "romanizer.py")
    for required in (checkpoint, config):
        if not required.is_file():
            raise FileNotFoundError(required)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    synth = Synthesizer(
        tts_checkpoint=str(checkpoint), tts_config_path=str(config),
        use_cuda=args.cuda,
    )
    load_seconds = time.perf_counter() - started

    results = []
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            romanized = romanize(row["text"])
            output = args.output_dir / f'{row["id"]}.wav'
            started = time.perf_counter()
            wav = synth.tts(romanized)
            synthesis_seconds = time.perf_counter() - started
            synth.save_wav(wav, str(output))
            duration = audio_duration(output)
            results.append({
                **row,
                "romanized": romanized,
                "audio_path": str(output),
                "audio_seconds": round(duration, 4),
                "synthesis_seconds": round(synthesis_seconds, 4),
                "rtf": round(synthesis_seconds / duration, 4),
            })
            print(f'{row["id"]}: {duration:.2f}s audio, RTF={synthesis_seconds / duration:.3f}')

    metadata = {
        "model": "dialoglk/SinhalaVITS-TTS-F1",
        "device": "cuda" if args.cuda else "cpu",
        "model_load_seconds": round(load_seconds, 4),
        "item_count": len(results),
        "mean_rtf": round(sum(item["rtf"] for item in results) / len(results), 4),
        "items": results,
    }
    result_path = args.output_dir / "results.json"
    result_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"results: {result_path}")


if __name__ == "__main__":
    main()
