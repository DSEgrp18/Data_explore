---
title: "Project Proposal: A Neural Text-to-Speech System for Sinhala"
subtitle: "DSE Project — Group 18"
date: "12 July 2026"
---

# Project Proposal: A Neural Text-to-Speech System for Sinhala

**Module:** Data Science & Engineering Project
**Group:** 18
**Mentor:** Dr. Buddhika (buddhika@cse.mrt.ac.lk)
**Members:** _Member 1_, _Member 2_, _Member 3_ *(fill in names/index numbers)*

---

## 1. Overview

We propose to build a working neural Text-to-Speech (TTS) application for Sinhala: a system that converts written Sinhala text into natural-sounding speech, delivered as a web application, with speech quality evaluated against all publicly available Sinhala TTS baselines. The expected outcomes match the mentor's brief: **(primary)** a working Sinhala TTS application and **(secondary)** a research publication based on the work.

## 2. Problem Statement & Motivation

Sinhala is spoken by ~17 million people yet remains a low-resource language for speech technology. Our preliminary investigation (July 2026) verified a striking gap:

- **Meta's MMS-TTS project covers 1,100+ languages but skips Sinhala entirely** (`facebook/mms-tts-sin` does not exist; MMS supports Sinhala only for speech *recognition*).
- Large multilingual TTS corpora (CMU Wilderness, OpenBibleTTS) contain **no Sinhala** — we checked their full language lists.
- Mozilla Common Voice has effectively **no usable Sinhala data** (59 unvalidated clips, 0.4 h, 1 speaker).
- The only published Sinhala TTS models are community/industry efforts (Dialog/UoM VITS voices, one F5-TTS fine-tune) with no rigorous published evaluation.

This means: (a) a well-built Sinhala TTS system fills a real gap, and (b) even a careful fine-tuning + evaluation study is a **credible publication**, since no big-tech baseline exists to overshadow it.

## 3. Objectives

1. Build a complete Sinhala TTS pipeline: text normalization → grapheme/phoneme processing → neural acoustic model + vocoder → audio.
2. Fine-tune at least two modern architectures (VITS; F5-TTS or similar flow-matching model) on the best available Sinhala corpora.
3. Systematically evaluate against all existing Sinhala TTS baselines using native-speaker MOS (Mean Opinion Score) and ASR-based intelligibility (Whisper WER).
4. Deliver a usable web application (text in → audio out, with Singlish input support as a stretch goal).
5. Write up methodology and results as a paper targeting a suitable venue (ICTer, MERCon, or an Interspeech/LREC low-resource workshop).

## 4. Datasets (verified hands-on, July 2026)

We downloaded and measured the candidate corpora rather than trusting dataset cards; several descriptions turned out to be inaccurate.

| Dataset | Measured / verified details | License | Role |
|---|---|---|---|
| **PathNirvana (pnfo)** — [github.com/pnfo/sinhala-tts-dataset](https://github.com/pnfo/sinhala-tts-dataset/releases) | 13.8 h studio-quality, 2 speakers (male 11.8 h / 5,200 clips; female 2 h / 1,000 clips), 22.05 kHz, LJSpeech format, silence-trimmed | GPL-3.0 + non-offensive-use restriction | **Primary training set** (single-speaker voice) |
| **OpenSLR SLR30** — [openslr.org/30](https://www.openslr.org/30/) | Measured: 2,064 wavs, 12 speakers, 3.38 h, 48 kHz — **only 1,251 clips (~2 h) have transcripts** | CC BY-SA 4.0 | Multi-speaker experiments (transcribed subset only) |
| **OpenSLR SLR52** — [openslr.org/52](https://www.openslr.org/52/) | ~185,000 crowdsourced utterances, 16 kHz, many speakers (ASR quality) | CC BY-SA 4.0 | Pre-training, vocoder training, ASR fine-tuning for evaluation |
| **SafnasKaldeen** — [Kaggle](https://www.kaggle.com/datasets/safnask/sinhalatts-dataset-publication-by-voicemakers) | 4 native speakers, phonetically balanced (UoM CSE FYP 2025). Note: the Hugging Face mirror is a card only — data is on Kaggle | Apache-2.0 | Speaker-variety fine-tuning |
| **FLEURS (si_lk)** — [HF: google/fleurs](https://huggingface.co/datasets/google/fleurs) | ~10–12 h read speech with standard splits | CC BY 4.0 | **Evaluation only** (intelligibility test sentences) |
| **Common Voice** | 59 clips / 0.4 h / 1 speaker / 0 validated | CC0 | Not usable — documented for completeness |

**Supporting linguistic resources (verified):** Google's [language-resources `si/`](https://github.com/google/language-resources) provides a Sinhala text-normalization grammar (`sinhala.grm`), pronunciation dictionaries, and IPA mappings — the hardest parts of a TTS frontend already exist. [Dakshina](https://github.com/google-research-datasets/dakshina) provides a Singlish↔Sinhala romanization lexicon for a transliterated-input feature.

## 5. Baselines (verified available)

1. **Dialog/UoM SinhalaVITS** (F1 female / M2 male) — Coqui VITS checkpoints, MPL-2.0, with a romanizer frontend.
2. **PathNirvana VITS checkpoint** — trained on the pnfo dataset, shipped with it.
3. **tharindumihi F5-TTS Sinhala** — flow-matching fine-tune, CC-BY-NC-4.0.

For automatic evaluation we will use **seniruk/whisper-small-si** (fine-tuned Sinhala Whisper, Apache-2.0) to compute WER on synthesized speech, alongside native-speaker MOS panels.

## 6. Methodology

1. **Data engineering:** unified preprocessing pipeline (resampling to a common rate, loudness normalization, silence trimming, transcript cleaning, train/val/test splits); corpus statistics and data quality report.
2. **Frontend:** adapt google/language-resources Sinhala normalization (numbers, dates, currency, English loanwords) into a Python module; grapheme-based input first, phoneme-based if time allows.
3. **Modeling:** (a) fine-tune/train VITS on pnfo male voice; (b) fine-tune F5-TTS for comparison; (c) optional multi-speaker model on SLR30 + SafnasKaldeen.
4. **Evaluation:** MOS study with ≥15 native speakers (naturalness + intelligibility), Whisper-WER on FLEURS sentences, side-by-side preference tests vs. the three baselines; error analysis on numbers/loanwords/rare syllables.
5. **Application:** FastAPI backend serving the model + simple web UI (text box → audio player), Singlish input via Dakshina transliteration as stretch goal; deploy demo on Hugging Face Spaces.

## 7. Work Division (3 members — equal participation in the data science core)

Since this is a *data science* project, the division is by **phase, not by role**: every member works on data engineering, modeling, and evaluation. Within each phase, tasks are split into parallel slices of similar weight so each member owns a complete data-science workflow end to end. Software-engineering tasks (app, deployment) are secondary duties spread across all three.

| Phase (all members participate) | Member 1 | Member 2 | Member 3 |
|---|---|---|---|
| **Data collection & preprocessing** | pnfo corpus: cleaning, resampling, splits, corpus statistics | SLR30 + SafnasKaldeen: transcript matching, multi-speaker preprocessing | SLR52: large-scale filtering; FLEURS eval set; text-normalization data (`sinhala.grm`, Dakshina) |
| **Exploratory data analysis** | Audio quality & duration distributions | Speaker/gender coverage, phonetic coverage analysis | Text statistics: rare syllables, loanwords, numbers frequency |
| **Modeling** | VITS fine-tuning on pnfo male voice | F5-TTS fine-tuning + comparison | Multi-speaker VITS experiments + reproducing the 3 published baselines |
| **Evaluation** | Whisper-WER automatic evaluation harness | MOS study design, rater recruitment, execution | Error analysis (numbers, loanwords, rare syllables) + statistical significance testing |
| **Application (secondary)** | Inference API backend | Model serving & frontend-model integration | Web UI + demo deployment |
| **Paper & report** | Data & experimental-setup sections | Modeling & results sections | Evaluation, error-analysis & related-work sections |

Members pair on blockers across slices, and slice assignments can be swapped by agreement — the constraint is that **each member ends the project having done data work, model training, and evaluation**, which is also what each of us should be able to defend at the viva.

## 8. Timeline (14 weeks)

| Weeks | Milestone |
|---|---|
| 1–2 | Proposal finalized; environments set up; all datasets downloaded & preprocessed; baselines running locally |
| 3–4 | Frontend v1 (normalization + romanization); first VITS fine-tuning run on pnfo |
| 5–7 | Model iteration (VITS vs F5-TTS); interim demo; mid-project review with mentor |
| 8–9 | Multi-speaker experiments; frontend-model integration; web app v1 |
| 10–11 | Frozen models; MOS study + automatic evaluation; error analysis |
| 12–13 | Paper draft; app polish; demo deployment |
| 14 | Final report, presentation, submission-ready paper |

## 9. Risks & Mitigation

| Risk | Mitigation |
|---|---|
| GPU access insufficient for training | Fine-tune from existing checkpoints (pnfo VITS) instead of training from scratch; use Colab/Kaggle/university cluster; F5-TTS fine-tuning is feasible on a single consumer GPU |
| pnfo GPL-3.0 + usage restriction complicates release | Keep model weights under compatible license; document restriction; SLR30/SLR52 (CC BY-SA) as fallback training data |
| MOS study recruitment | We are native speakers with campus access; 15–20 raters is realistic; prepare rating UI early |
| Baselines hard to reproduce | Already verified all three run from published checkpoints; romanizer + configs downloaded |
| Publication timeline vs module deadline | Write the report in paper format from the start so conversion is trivial; arXiv preprint as minimum outcome |

## 10. Expected Outcomes

1. **Working Sinhala TTS web application** (primary outcome per project brief)
2. Fine-tuned Sinhala TTS models + training/evaluation code (public GitHub repo)
3. First systematic evaluation of all available Sinhala TTS systems (MOS + WER)
4. **Research paper** targeting ICTer / MERCon / low-resource speech workshop (secondary outcome per project brief)

## 11. References

- Sodimana et al., *A Step-by-Step Process for Building TTS Voices Using Open Source Data and Frameworks for Bangla, Javanese, Khmer, Nepali, Sinhala, and Sundanese*, SLTU 2018
- Sodimana et al., *Text Normalization for Bangla, Khmer, Nepali, Javanese, Sinhala and Sundanese TTS Systems*, 2018
- Kim et al., *VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech*, ICML 2021
- Chen et al., *F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching*, 2024
- Pratap et al., *Scaling Speech Technology to 1,000+ Languages* (MMS), 2023
- Dataset/resource links as cited in §4–5 (all verified July 2026; details in the group's `Data_explore` repository)
