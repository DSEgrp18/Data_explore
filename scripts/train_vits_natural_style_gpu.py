"""Train a from-scratch Sinhala VITS model with acoustic style embeddings.

The style names are pseudo-speakers derived from pitch, energy, speaking rate,
and silence. They are controls for delivery, not claims of human emotion.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


CPU_TTS_SITE_PACKAGES = Path("/home/kusal/Documents/DSE_project_grp18/tts_demo/venv/lib/python3.12/site-packages")
if str(CPU_TTS_SITE_PACKAGES) not in sys.path:
    sys.path.append(str(CPU_TTS_SITE_PACKAGES))


def make_config(args: argparse.Namespace, output_config: Path, style_names: list[str]) -> None:
    config = json.loads(args.base_config.read_text(encoding="utf-8"))
    config.update({
        "output_path": str(args.output_root),
        "run_name": args.run_name,
        "project_name": "sinhala_natural_style",
        "run_description": "From-scratch Sinhala VITS with acoustic prosody style embeddings",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": max(1, args.batch_size),
        "num_loader_workers": args.workers,
        "num_eval_loader_workers": max(1, min(args.workers, 2)),
        "mixed_precision": True,
        "save_step": args.save_step,
        "save_n_checkpoints": args.save_n_checkpoints,
        "save_best_after": args.save_step,
        "print_step": 25,
        "plot_step": args.save_step,
        "eval_split_size": 0.1,
        "eval_split_max_size": 200,
        "shuffle": True,
        "drop_last": True,
        "lr_gen": args.learning_rate,
        "lr_disc": args.learning_rate,
        "lr_scheduler_gen_params": {"gamma": 0.99995, "last_epoch": -1},
        "lr_scheduler_disc_params": {"gamma": 0.99995, "last_epoch": -1},
        "use_speaker_embedding": True,
        "num_speakers": len(style_names),
        "speakers_file": None,
        "test_sentences": [
            ["hāyi kohomada oyālaṭa?", style_names[0], None, None],
            ["mama hitanne apiṭa loku væḍak karanna puluvan veyi.", style_names[0], None, None],
        ],
        "datasets": [{
            "formatter": "coqui",
            "dataset_name": "sinhala_prosody_styles",
            "path": str(args.dataset_root),
            "meta_file_train": "metadata_train.csv",
            "meta_file_val": "metadata_val.csv",
            "ignored_speakers": None,
            "language": "si",
            "phonemizer": "",
            "meta_file_attn_mask": "",
        }],
    })
    model_args = config.setdefault("model_args", {})
    model_args.update({
        "use_speaker_embedding": True,
        "num_speakers": len(style_names),
        "speaker_embedding_channels": args.style_embedding_channels,
        "speakers_file": None,
        "freeze_encoder": False,
        "freeze_DP": False,
        "freeze_PE": False,
        "freeze_flow_decoder": False,
        "freeze_waveform_decoder": False,
    })
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(json.dumps(config, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--prosody-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name", default="vits_natural_style_gpu")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--save-step", type=int, default=5000)
    parser.add_argument("--save-n-checkpoints", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--style-embedding-channels", type=int, default=128)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch size must be positive")
    if args.save_n_checkpoints < 1 or args.style_embedding_channels < 1:
        raise ValueError("checkpoint count and style embedding size must be positive")
    if not args.base_config.is_file():
        raise FileNotFoundError(args.base_config)
    if not (args.dataset_root / "metadata_train.csv").is_file():
        raise FileNotFoundError(args.dataset_root / "metadata_train.csv")
    report = json.loads(args.prosody_report.read_text(encoding="utf-8"))
    style_names = sorted(report["style_counts"])
    if len(style_names) < 2:
        raise ValueError("prosody report must contain at least two style buckets")

    args.output_root.mkdir(parents=True, exist_ok=True)
    config_path = args.output_root / "config.json"
    make_config(args, config_path, style_names)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"styles: {len(style_names)} {style_names}")
    print(f"config: {config_path}")

    from TTS.bin.train_tts import main as train_tts

    train_tts([
        "--config_path", str(config_path),
        "--gpu", "0",
    ])


if __name__ == "__main__":
    main()
