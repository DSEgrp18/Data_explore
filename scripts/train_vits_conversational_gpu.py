"""Fine-tune the existing Sinhala VITS checkpoint on conversational speech.

Run this script with a CUDA-enabled Python environment. Coqui TTS is reused
from the project's existing environment after CUDA PyTorch is imported, which
avoids replacing the working baseline environment.
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


def make_config(args: argparse.Namespace, output_config: Path) -> None:
    config = json.loads(args.base_config.read_text(encoding="utf-8"))
    config.update({
        "output_path": str(args.output_root),
        "run_name": args.run_name,
        "project_name": "sinhala_naturalness",
        "run_description": "GPU conversational naturalness adaptation from Dialog Sinhala VITS",
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
        "datasets": [{
            "formatter": "coqui",
            "dataset_name": "sinhala_podcast_cc_v1_ten",
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
    model_args["freeze_encoder"] = args.freeze_text_encoder
    model_args["freeze_DP"] = False
    model_args["freeze_PE"] = not args.unfreeze_acoustic
    model_args["freeze_flow_decoder"] = not args.unfreeze_acoustic
    model_args["freeze_waveform_decoder"] = not args.unfreeze_acoustic
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(json.dumps(config, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name", default="vits_conversational_gpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--save-step", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.00003)
    parser.add_argument("--unfreeze-text-encoder", action="store_true")
    parser.add_argument(
        "--unfreeze-acoustic",
        action="store_true",
        help="also update posterior/flow/waveform modules; disabled by default to protect clarity",
    )
    parser.add_argument("--save-n-checkpoints", type=int, default=2)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run with the CUDA-enabled environment.")
    if not args.base_checkpoint.is_file():
        raise FileNotFoundError(args.base_checkpoint)
    if not args.base_config.is_file():
        raise FileNotFoundError(args.base_config)
    if not (args.dataset_root / "metadata_train.csv").is_file():
        raise FileNotFoundError(args.dataset_root / "metadata_train.csv")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch size must be positive")
    if args.save_n_checkpoints < 1:
        raise ValueError("save-n-checkpoints must be positive")

    args.output_root.mkdir(parents=True, exist_ok=True)
    config_path = args.output_root / "training_config.json"
    args.freeze_text_encoder = not args.unfreeze_text_encoder
    make_config(args, config_path)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM free/total: {torch.cuda.mem_get_info()[0] / 1024**3:.2f}/{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GiB")
    print(f"config: {config_path}")

    from TTS.bin.train_tts import main as train_tts

    train_tts([
        "--config_path", str(config_path),
        "--restore_path", str(args.base_checkpoint),
        "--gpu", "0",
    ])


if __name__ == "__main__":
    main()
