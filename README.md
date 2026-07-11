# Sinhala TTS — Data Source Exploration

Working notes for Group 18's DSE project (Sinhala Text-to-Speech).
This repo tracks every data source we find, with license and suitability notes, so we can decide what goes into training, fine-tuning, and evaluation.

**Legend:** ✅ verified (link opened and details checked) · 🔍 found, needs verification · 💡 idea, not yet explored

---

## 1. Datasets the team already found (now verified)

| Dataset | Link | Details | License | Verdict for TTS |
|---|---|---|---|---|
| ✅ **PathNirvana (pnfo)** | [github.com/pnfo/sinhala-tts-dataset](https://github.com/pnfo/sinhala-tts-dataset/releases) | **13.8 h**, 2 speakers (male ~11.8 h / 5,200 clips, female ~2 h / 1,000 clips), 22,050 Hz 16-bit, silence-trimmed, LJSpeech-format metadata, sentences chosen to cover rare Sanskrit/Pali-origin syllables | GPL-3.0 + restriction: **non-obscene, non-offensive speech generation only** | **Best single-speaker training set we have.** Clean, studio-like, already in LJSpeech format. Restriction is fine for our use case but must be stated in the report. |
| ✅ **OpenSLR SLR30** | official page: [openslr.org/30](https://www.openslr.org/30/) (the trmal.net link the team found is one of its mirrors) | Multi-speaker Sinhala TTS corpus collected by Google in Sri Lanka (2015–16), ~699 MB audio + transcription lines, manually quality-checked | **CC BY-SA 4.0** | Good multi-speaker complement to pnfo. This is the corpus behind Google's 2018 SLTU Sinhala TTS paper. |
| ✅ **SafnasKaldeen** | [Kaggle](https://www.kaggle.com/datasets/safnask/sinhalatts-dataset-publication-by-voicemakers) · also on HF: [SafnasKaldeen/Sinhala-Text-speech-Dataset-TTS](https://huggingface.co/datasets/SafnasKaldeen/Sinhala-Text-speech-Dataset-TTS) | 4 native speakers reading phonetically balanced sentences, built specifically for TTS research | check dataset card | Multi-speaker fine-tuning / speaker-variety evaluation. HF version is easier to load programmatically than Kaggle. |
| ✅ **FLEURS (si_lk)** | [huggingface.co/datasets/google/fleurs](https://huggingface.co/datasets/google/fleurs) | ~10–12 h per language, read speech, standard train/dev/test splits | CC BY 4.0 | Too small/noisy to train TTS on, but the **standard evaluation split** — good for ASR-based intelligibility scoring (WER of Whisper on our synthesized speech). |
| ✅ **Common Voice (Sinhala)** | [commonvoice.mozilla.org/datasets](https://commonvoice.mozilla.org/en/datasets) — newest versions now on [Mozilla Data Collective](https://community.mozilladatacollective.com/common-voice-23-0-live-on-mozilla-data-collective/) | Crowdsourced, many speakers/devices; Sinhala subset is small (check hours in latest release, v23 as of 2026) | CC0 | Too noisy for TTS training; useful for **evaluation variety** and possibly vocoder robustness. Note: newest releases moved from the old datasets page to Mozilla Data Collective. |

---

## 2. New findings — speech datasets

### 2.1 OpenSLR SLR52 — Large Sinhala ASR corpus ✅ (biggest one)
- **Link:** [openslr.org/52](https://www.openslr.org/52/) · HF mirror: [openslr/openslr](https://huggingface.co/datasets/openslr/openslr)
- **~185,000 transcribed utterances**, collected by Google, manually quality checked. License: **CC BY-SA 4.0**.
- ASR-quality (crowdsourced, 16 kHz, many speakers) — *not* clean enough to train a single TTS voice directly, but very valuable for:
  - pre-training / multi-speaker TTS stage before fine-tuning on pnfo,
  - training our own Sinhala vocoder or duration model,
  - building an ASR evaluation loop (or fine-tuning Whisper for Sinhala to score our TTS output).

### 2.2 Hugging Face datasets 🔍
- [keshan/multispeaker-tts-sinhala](https://huggingface.co/datasets/keshan/multispeaker-tts-sinhala) — packaged multi-speaker Sinhala TTS data tied to Google's SLTU 2018 paper (same lineage as SLR30); convenient `datasets`-loadable format. Verify it against SLR30 to avoid duplication.
- [seniruk/sinscribe-sinhala-stt](https://huggingface.co/datasets/seniruk/sinscribe-sinhala-stt) — Sinhala speech-to-text dataset; check size/license, potentially more eval data.

### 2.3 Bible-recording corpora 🔍 (large but domain-limited)
- **CMU Wilderness** — [festvox.org/cmu_wilderness](http://festvox.org/cmu_wilderness/) · [github.com/festvox/datasets-CMU_Wilderness](https://github.com/festvox/datasets-CMU_Wilderness): ~20 h of aligned New Testament audio per language across ~700 languages from bible.is. Distributed as *scripts that rebuild alignments*, not raw audio. **To verify:** Sinhala inclusion and bible.is licensing terms.
- **Meta MMS data pipeline** — the [MMS paper](https://arxiv.org/pdf/2305.13516) used religious recordings covering 1,100+ languages including Sinhala; the recordings themselves aren't redistributed, but the pretrained models are (see §3).
- **OpenBibleTTS** — [arxiv.org/html/2606.09553](https://arxiv.org/html/2606.09553): TTS-ready corpora from open.bible for 37 low-resource languages. **To verify:** whether Sinhala is one of the 37.
- ⚠️ Caveat for all of these: single domain (religious text), often a single professional narrator — fine as *extra training hours*, must be flagged as domain bias in the paper.

---

## 3. Pretrained models we can fine-tune or benchmark against

| Model | Link | Notes |
|---|---|---|
| **Meta MMS-TTS Sinhala** | [facebook/mms-tts-sin](https://huggingface.co/facebook/mms-tts-sin) (part of [facebook/mms-tts](https://huggingface.co/facebook/mms-tts), 1,100+ languages, per-language VITS checkpoints, in 🤗 Transformers ≥ 4.33) | **Strongest ready-made baseline.** CC-BY-NC license — fine for research/publication, not commercial. Our fine-tuned model must beat this in MOS to claim a contribution. |
| SinhalaVITS-TTS (community) | [dialoglk/SinhalaVITS-TTS-F1](https://huggingface.co/dialoglk/SinhalaVITS-TTS-F1) · [dialoglk/SinhalaVITS-TTS-M2](https://huggingface.co/dialoglk/SinhalaVITS-TTS-M2) | Community VITS voices (female/male) — check what data they trained on; useful baselines. |
| F5-TTS Sinhala (community) | [tharindumihi/tts-si-F5-TTS](https://huggingface.co/tharindumihi/tts-si-F5-TTS) | Flow-matching architecture fine-tuned for Sinhala — shows the modern-architecture path is viable; a good reference point for our own F5/VITS fine-tuning. |
| Whisper (OpenAI) | supports Sinhala | Not TTS — but fine-tuning Whisper on SLR52 gives us an **automatic intelligibility metric** (WER on synthesized speech), which reviewers like. |

---

## 4. Text & linguistic resources (TTS frontend — just as important as audio)

- ✅ **google/language-resources (`si/`)** — [github.com/google/language-resources](https://github.com/google/language-resources): Sinhala **pronunciation lexicon, phonology definition, and text-normalization grammars** (numbers, dates, currency → words). This is the hardest part of a TTS frontend and it already exists. Companion papers: [TTS voices for Sinhala (SLTU 2018)](https://www.isca-archive.org/sltu_2018/sodimana18_sltu.pdf), [Text Normalization for Sinhala TTS](https://research.google/pubs/pub47344).
- ✅ **Dakshina** — [github.com/google-research-datasets/dakshina](https://github.com/google-research-datasets/dakshina): Sinhala native-script Wikipedia text + **romanization lexicon** (Singlish ↔ Sinhala). Enables a "type in Singlish, hear Sinhala" demo feature.
- 🔍 **Swa-bhasha Resource Hub** — [arxiv.org/pdf/2507.09245](https://arxiv.org/pdf/2507.09245): recent (2025) romanized-Sinhala transliteration systems and data.
- 🔍 **UCSC LTRL resources** — [language.lk lexical resources](https://www.language.lk/en/resources/lexical-resources/): 10M-word Sinhala corpus, corpus-based lexicon; UCSC also built the original unit-selection [Sinhala TTS](https://dl.ucsc.cmb.ac.lk/jspui/handle/123456789/2503) and an [ASR project](https://ucsc.cmb.ac.lk/speech-recognition-system-sinhala/). Some resources may need an email request — good citation material either way.
- 💡 Large web text for sentence selection / normalization testing: MADLAD-400, OSCAR, Sinhala news crawls.

---

## 5. Ideas for creating our own data 💡

- **Forced alignment pipeline:** long-form Sinhala audio (audiobooks, news bulletins) + Montreal Forced Aligner / CTC segmentation with Whisper → segmented TTS clips. Copyright is the blocker — only use openly licensed or self-recorded audio.
- **Record a small studio set ourselves** (1–2 h, phonetically balanced sentences from the UCSC corpus) — even a small clean set is a publishable contribution if released openly.
- **Data statement for the paper:** whatever we use, log speaker counts, hours, license, and domain here so the paper's dataset table writes itself.

## 6. Recommended stack (current thinking)

1. **Train/fine-tune on:** pnfo (primary voice) + SLR30 & SafnasKaldeen (multi-speaker experiments)
2. **Pre-train / auxiliary:** SLR52
3. **Frontend:** google/language-resources `si` normalization + lexicon; Dakshina for Singlish input
4. **Baselines:** facebook/mms-tts-sin, dialoglk VITS voices
5. **Evaluation:** native-speaker MOS + Whisper-WER on FLEURS/Common Voice sentences

## 7. To-do

- [ ] Verify Sinhala presence in CMU Wilderness and OpenBibleTTS
- [ ] Check license + size of SafnasKaldeen and sinscribe datasets
- [ ] Get exact validated hours of Sinhala in Common Voice v23
- [ ] Download SLR30 + pnfo, run stats (hours, speakers, sample-rate consistency)
- [ ] Try facebook/mms-tts-sin locally and save sample outputs as baseline
