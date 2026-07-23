#!/usr/bin/env python3
"""Fine-tune a Sinhala F5-TTS checkpoint on the prepared Unicode dataset.

This intentionally uses the official F5 model/trainer, but loads the project
dataset by path and the Sinhala character vocabulary instead of the upstream
Chinese pinyin defaults.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from f5_tts.model import CFM, DiT, Trainer
from f5_tts.model.dataset import CustomDataset
from f5_tts.model.utils import get_tokenizer
from datasets import Dataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=Path("runs/f5_sinhala_data/arrow_train"))
    p.add_argument("--output-root", type=Path, default=Path("runs/f5_sinhala_finetune"))
    p.add_argument("--pretrained", type=Path, required=True, help="Matching Sinhala F5 .pt/.safetensors checkpoint")
    p.add_argument("--vocab-path", type=Path, default=None, help="Checkpoint vocabulary; defaults to dataset vocabulary")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-frames", type=int, default=3200)
    p.add_argument("--max-samples", type=int, default=2)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    args = p.parse_args()

    root = args.dataset_root.resolve()
    out = args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.pretrained.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    # Trainer.load_checkpoint() discovers pretrained_*.pt/.safetensors.
    target = out / f"pretrained_{checkpoint.name}"
    if not target.exists():
        shutil.copy2(checkpoint, target)

    vocab_path = (args.vocab_path or (root / "vocab.txt")).resolve()
    vocab_char_map, vocab_size = get_tokenizer(str(vocab_path), "custom")
    mel_kwargs = dict(
        target_sample_rate=24_000,
        n_mel_channels=100,
        hop_length=256,
        win_length=1024,
        n_fft=1024,
        mel_spec_type="vocos",
    )
    model = CFM(
        transformer=DiT(
            dim=1024,
            depth=22,
            heads=16,
            ff_mult=2,
            text_dim=512,
            conv_layers=4,
            text_num_embeds=vocab_size,
            mel_dim=100,
            checkpoint_activations=True,
        ),
        mel_spec_kwargs=mel_kwargs,
        vocab_char_map=vocab_char_map,
    )
    trainer = Trainer(
        model,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        num_warmup_updates=200,
        save_per_updates=500,
        keep_last_n_checkpoints=2,
        checkpoint_path=str(out),
        batch_size_per_gpu=args.batch_frames,
        batch_size_type="frame",
        max_samples=args.max_samples,
        grad_accumulation_steps=4,
        max_grad_norm=1.0,
        logger=None,
        last_per_updates=250,
        bnb_optimizer=True,
        mel_spec_type="vocos",
        model_cfg_dict={"dataset": str(root), "vocab": str(vocab_path)},
    )
    update = trainer.load_checkpoint()
    dataset = Dataset.from_file(str(root / "raw.arrow"))
    durations = json.loads((root / "duration.json").read_text(encoding="utf-8"))["duration"]
    train_dataset = CustomDataset(dataset, durations=durations, **mel_kwargs)
    trainer.train(train_dataset, num_workers=args.workers, resumable_with_seed=666)
    print(f"Finished F5 Sinhala fine-tuning from update {update}; checkpoints: {out}")


if __name__ == "__main__":
    main()
