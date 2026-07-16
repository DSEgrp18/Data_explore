# Sinhala VITS baseline test — 17 July 2026

## Setup

- Model: `dialoglk/SinhalaVITS-TTS-F1`, Nipunika checkpoint at 210,000 steps
- Pipeline: Sinhala text → model-provided romanizer → Coqui VITS → waveform
- Device: CPU (`torch 2.13.0+cpu`); GPU was intentionally not used because the
  existing isolated environment contains a CPU-only PyTorch build
- Challenge set: `evaluation/baseline_challenge.csv`
- Outputs: local ignored directory `outputs/vits_baseline_20260717/`
- Reproducible runner: `scripts/run_vits_baseline.py`

Command:

```bash
../tts_demo/venv/bin/python scripts/run_vits_baseline.py \
  --manifest evaluation/baseline_challenge.csv \
  --model-dir ../tts_demo/model \
  --output-dir outputs/vits_baseline_20260717
```

## Measured results

All 10 items generated valid mono 22,050 Hz WAV files.

| Item | Category | Audio seconds | CPU synthesis seconds | RTF |
|---|---|---:|---:|---:|
| `plain_01` | plain | 4.854 | 1.950 | 0.402 |
| `plain_02` | plain | 2.810 | 0.877 | 0.312 |
| `aspiration_01` | phonology | 4.157 | 1.442 | 0.347 |
| `prenasal_01` | phonology | 4.923 | 1.840 | 0.374 |
| `conjunct_01` | rare conjunct | 3.820 | 1.478 | 0.387 |
| `number_raw` | raw numeral | 3.124 | 1.015 | 0.325 |
| `number_spoken` | normalized numeral | 3.820 | 1.463 | 0.383 |
| `date_raw` | raw date | 3.530 | 1.233 | 0.349 |
| `date_spoken` | normalized date | 4.436 | 1.537 | 0.347 |
| `code_switch_01` | code-switching | 4.111 | 1.325 | 0.322 |

- Model load time: **2.427 seconds**
- Mean RTF: **0.355** on CPU
- Mean generated duration: **3.96 seconds**
- Output size: **1.8 MB**

An RTF below 1 means the model generated audio faster than its playback duration,
even on CPU. This establishes a usable latency baseline for the final system.

## Confirmed failure

The model vocabulary rejected every digit in the raw numeral and raw date:

```text
Character '2' not found in the vocabulary. Discarding it.
Character '5' not found in the vocabulary. Discarding it.
Character '0' not found in the vocabulary. Discarding it.
Character '6' not found in the vocabulary. Discarding it.
Character '1' not found in the vocabulary. Discarding it.
```

Consequently, `250`, `2026`, and `16` were omitted rather than spoken. The
word-form versions generated 0.70 seconds and 0.91 seconds more audio than their
raw counterparts respectively. Duration alone is not an intelligibility metric,
but together with the explicit vocabulary warnings it confirms that normalization
changes the spoken content and fixes a real baseline failure.

## What is not yet measured

This run verifies execution, waveform validity, latency, and the digit failure.
It does **not** establish naturalness, correct phonological realization, CER/WER,
or listener preference. Those require:

1. manual listening by the group;
2. transcription with one fixed Sinhala ASR model;
3. a blinded native-speaker listening study;
4. direct comparison with the future SiFi-TTS outputs on the same manifest.

The phonology, conjunct, and code-switching clips are deliberately included for
that next qualitative review; no quality claim is made before listening/scoring.
