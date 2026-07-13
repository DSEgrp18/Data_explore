# Sinhala TTS — Data Source Exploration

Working notes for Group 18's DSE project (Sinhala Text-to-Speech).
This repo tracks every data source we find, with license and suitability notes, so we can decide what goes into training, fine-tuning, and evaluation.

**Legend:** ✅ verified hands-on (downloaded / measured / API-checked) · 🔍 found, needs verification · ❌ dead end (checked, not usable)

> **Update 2026-07-12:** all items from the original to-do were verified hands-on. Several descriptions from dataset cards turned out to be wrong or misleading — corrected numbers below come from actually downloading and measuring the data (see `scripts/audio_stats.py`).

---

## 1. Core training datasets (verified)

| Dataset | Link | Measured details | License | Verdict for TTS |
|---|---|---|---|---|
| ✅ **PathNirvana (pnfo)** | [github.com/pnfo/sinhala-tts-dataset](https://github.com/pnfo/sinhala-tts-dataset/releases) | **Measured (v2.1): 6,386 clips, 13.61 h, all 22,050 Hz, avg clip 7.7 s.** Male voice (mettananda): 5,401 clips / 11.58 h; female (oshadi): 985 clips / 2.03 h. `metadata.csv` has **both romanized and Sinhala-script transcripts** (`id\|roman\|sinhala\|speaker`) — ready for grapheme or romanized training. Card claims (13.8 h) check out. | GPL-3.0 + restriction: **non-obscene, non-offensive speech generation only** | **Best single-speaker training set available.** Bonus: the releases also ship a **pretrained VITS checkpoint** (`checkpoint_80000.pth`, 997 MB + config) — a ready baseline trained on this data. |
| ✅ **OpenSLR SLR30** | [openslr.org/30](https://www.openslr.org/30/) (openslr.trmal.net is a mirror) | **Measured: 2,064 wavs, 12 speakers, 3.38 h total, 48 kHz, avg clip 5.9 s — but only 1,251 clips (~2 h) have transcripts** in `si_lk.lines.txt`. Collected by Google in Sri Lanka 2015–16 | CC BY-SA 4.0 | Multi-speaker complement to pnfo, but much smaller than card descriptions suggest. Use only the 1,251 transcribed clips for TTS. Corpus behind Google's SLTU 2018 Sinhala TTS paper. |
| ✅ **OpenSLR SLR52** | [openslr.org/52](https://www.openslr.org/52/) | **~185,000 transcribed utterances**, crowdsourced, 16 kHz, many speakers; manually quality checked | CC BY-SA 4.0 | The big one. ASR-quality, not TTS-clean — use for pre-training, vocoder training, or fine-tuning Whisper for our evaluation loop. |
| ✅ **SafnasKaldeen** | [Kaggle](https://www.kaggle.com/datasets/safnask/sinhalatts-dataset-publication-by-voicemakers) | 4 native speakers, phonetically balanced sentences. UoM CSE final-year project (Dec 2025). ⚠️ **The [HF page](https://huggingface.co/datasets/SafnasKaldeen/Sinhala-Text-speech-Dataset-TTS) is a card only — 0 bytes of data; download from Kaggle.** | Apache-2.0 (per HF card) | Multi-speaker fine-tuning / speaker variety. Nice contact point: same department as us. |
| ✅ **FLEURS (si_lk)** | [huggingface.co/datasets/google/fleurs](https://huggingface.co/datasets/google/fleurs) | ~10–12 h read speech, standard train/dev/test splits | CC BY 4.0 | Not for training — use as the **standard evaluation split** (Whisper-WER intelligibility scoring of our synthesized speech). |

## 2. Dead ends — checked so nobody wastes time on them ❌

| Source | What we found |
|---|---|
| ❌ **Common Voice** | Sinhala is **not in the main scripted-speech corpus at all** (checked v26.0 metadata, 294 locales). It appears only in the spontaneous-speech corpus: **59 clips, 0.4 h, 1 speaker, 0 validated**. Unusable. |
| ❌ **CMU Wilderness** | No Sinhala. Checked the full ~700-language index at [festvox.org/cmu_wilderness](http://festvox.org/cmu_wilderness/) — no `SIN*` code, no "Sinhala" mention. |
| ❌ **OpenBibleTTS** | No Sinhala among its 37 languages (checked full paper text of [arxiv 2606.09553](https://arxiv.org/abs/2606.09553)). |
| ❌ **facebook/mms-tts-sin** | **Does not exist.** Meta's MMS-TTS (1,100+ languages) skips Sinhala — MMS *ASR* supports Sinhala but TTS does not (checked HF model list: `sid`, `sig`, `sil` exist, no `sin`). **Implication: there is no big-tech Sinhala TTS baseline — good for our publication's novelty claim.** |
| ❌ **sinscribe** ([seniruk/sinscribe-sinhala-stt](https://huggingface.co/datasets/seniruk/sinscribe-sinhala-stt)) | 12.6 GB / 161K rows, but it's **just SLR30 + SLR52 + pnfo re-packaged** under a restrictive personal license. No new audio. However, its author's fine-tuned **[seniruk/whisper-small-si](https://huggingface.co/seniruk/whisper-small-si)** (Apache-2.0) is directly useful for our evaluation loop. |

## 3. Baseline models to compare against (verified)

| Model | Link | Notes |
|---|---|---|
| ✅ **Dialog/UoM SinhalaVITS** | [dialoglk/SinhalaVITS-TTS-F1](https://huggingface.co/dialoglk/SinhalaVITS-TTS-F1) (female) · [dialoglk/SinhalaVITS-TTS-M2](https://huggingface.co/dialoglk/SinhalaVITS-TTS-M2) (male) | **Coqui-TTS VITS checkpoints, MPL-2.0**, tagged UoM + Dialog. Pipeline: Sinhala text → bundled `romanizer.py` → VITS. Checkpoints ~1 GB each (full training state). Our strongest published baseline. **We ran F1 locally** (samples in `baseline_samples/`): speech quality is decent, but the model **silently discards digits** — no text normalization at all. That's a demonstrable gap our frontend fixes. |
| ✅ **pnfo VITS checkpoint** | [v2.0-model release](https://github.com/pnfo/sinhala-tts-dataset/releases) | `checkpoint_80000.pth` + `config.json` — trained on the pnfo dataset. Second baseline for free. |
| ✅ **F5-TTS Sinhala** | [tharindumihi/tts-si-F5-TTS](https://huggingface.co/tharindumihi/tts-si-F5-TTS) | Flow-matching model + voice cloning, CC-BY-NC-4.0, ships sample outputs. Modern-architecture reference point. |
| ✅ **Whisper for eval** | [seniruk/whisper-small-si](https://huggingface.co/seniruk/whisper-small-si) (Apache-2.0) | Fine-tuned Sinhala ASR → automatic intelligibility metric (WER on our synthesized speech). |

## 4. Text & linguistic resources (TTS frontend)

- ✅ **google/language-resources `si/`** — [github.com/google/language-resources](https://github.com/google/language-resources): verified contents include `sinhala.grm` (text-normalization grammar), `textnorm/`, pronunciation dictionaries (`dict_regular.txt`, `dict_exceptions.txt`, `dict_homographs.txt`), IPA mapping (`si-si_FONIPA.txt`), festvox + merlin recipes. **The hardest part of a TTS frontend already exists here.** Papers: [SLTU 2018](https://www.isca-archive.org/sltu_2018/sodimana18_sltu.pdf), [Text Normalization](https://research.google/pubs/pub47344).
- ✅ **Dakshina** — [github.com/google-research-datasets/dakshina](https://github.com/google-research-datasets/dakshina): Sinhala Wikipedia text + romanization lexicon (Singlish ↔ Sinhala) → enables a "type in Singlish" demo feature.
- 🔍 **Swa-bhasha Resource Hub** — [arxiv 2507.09245](https://arxiv.org/pdf/2507.09245): 2025 romanized-Sinhala transliteration systems/data.
- 🔍 **UCSC LTRL** — [language.lk lexical resources](https://www.language.lk/en/resources/lexical-resources/): 10M-word Sinhala corpus; original unit-selection [Sinhala TTS](https://dl.ucsc.cmb.ac.lk/jspui/handle/123456789/2503). Some resources need an email request; good citation material.

## 5. Recommended stack (updated after verification)

1. **Primary training:** pnfo male voice (11.8 h, cleanest single speaker)
2. **Multi-speaker experiments:** SLR30 (1,251 transcribed clips only) + SafnasKaldeen (Kaggle)
3. **Pre-training / vocoder / ASR-eval:** SLR52
4. **Frontend:** google/language-resources `si` grammar + lexicon; Dakshina for Singlish input
5. **Baselines to beat:** dialoglk SinhalaVITS (F1/M2), pnfo checkpoint, F5-TTS Sinhala — *note: no Meta/Google baseline exists, strengthen this in the paper's novelty section*
6. **Evaluation:** native-speaker MOS + WER via seniruk/whisper-small-si on FLEURS sentences

## 6. Repo contents

- `scripts/audio_stats.py` — corpus statistics tool (files, hours, sample rates); run as `python3 audio_stats.py <dir>`
- `scripts/vits_baseline.py` — generates baseline samples from dialoglk SinhalaVITS-F1 (needs `pip install coqui-tts` + model files from HF)
- `baseline_samples/` — baseline TTS outputs for comparison (see §3)

## 7. Remaining to-do

- [x] Verify Sinhala in CMU Wilderness / OpenBibleTTS → ❌ absent from both
- [x] Check SafnasKaldeen + sinscribe license/size → Apache-2.0 card-only / re-packaged remix
- [x] Common Voice Sinhala hours → 0.4 h unvalidated, unusable
- [x] Download SLR30 + pnfo, measure real stats → both done, see §1 (measured with `scripts/audio_stats.py`)
- [x] Baseline samples from dialoglk VITS → done, see `baseline_samples/` (key finding: baseline drops digits — no normalization)
- [ ] Try pnfo's own VITS checkpoint as second baseline
- [ ] Download SafnasKaldeen from Kaggle, measure stats (needs Kaggle login)
- [ ] Listen to the baseline samples (all members) and note quality issues for the proposal defence
