# Sinhala Naturalness Data Acquisition Strategy

## Decision

Do not scrape ordinary YouTube podcasts into the Roshan or Nipunika VITS checkpoints. The previous experiment changed the acoustic decoder using noisy auto-caption segments from a different recording source and reduced clarity. YouTube content should only be used when the rights holder has granted written permission or the exact source is clearly licensed for this use; attribution alone is not permission.

## Source ranking

| Priority | Source | Use | Decision |
|---|---|---|---|
| 1 | Safely recorded speech from the Roshan and Nipunika speakers | Main naturalness fine-tuning | Best match. Record 20–60 minutes per voice with conversational, empathetic, question/answer, pause, number, and emotional prompts. |
| 2 | SafnasKaldeen Sinhala TTS dataset | Clean multi-speaker phonetic coverage | Promising Apache-2.0 card, four speakers and manually verified text, but the current Hugging Face repository contains only the card/metadata and no audio files. Do not train from it until the audio is obtained and provenance is confirmed. |
| 3 | outlawmold Sinhala TTS dataset | Separate naturalness/style baseline or general Sinhala pretraining | CC BY 4.0 and 23.23 hours, but it is single-speaker YouTube audio with auto-generated captions. Keep separate from the target-speaker VITS adaptation. |
| 4 | ehzawad Sinhala Emotion Dataset | Research candidate after permission review | 2,999 samples at 22.05 kHz, approximately 4.5 hours, but the card has no explicit license and no emotion-label field in the published schema. Hold until the author confirms rights and labels. |
| 5 | Mozilla Common Voice Spontaneous Sinhala | Conversational validation only | CC0, but the current release has only 59 clips, 0 validated transcripts, and 0.4 hours. Too small and noisy for primary TTS training. |
| 6 | OpenSLR SLR30 | Multi-speaker pronunciation/pretraining | CC BY-SA 4.0 and manually quality checked, but it is read speech and should not be merged directly into a single-speaker voice checkpoint. |

## Training recipe

1. Preserve the existing Roshan and Nipunika checkpoints as immutable baselines.
2. Build one dataset per target speaker. Do not mix podcast speakers into either target voice.
3. Start with 70% clean target-speaker read speech, 20% target-speaker conversational speech, and 10% target-speaker expressive speech. Sample clean speech in every batch so clarity cannot drift.
4. Use 2–12 second clips, one speaker per clip, no music/overlap, verified Sinhala transcript, and an ASR agreement check. Reject auto-caption segments with high character error rate, abnormal speaking rate, repeated text, or clipping.
5. First adaptation stage: freeze posterior encoder, flow, and waveform decoder; update only duration/prosody-related components at a very low learning rate. Stop if challenge-set intelligibility or speaker identity drops.
6. Only after a successful first stage, test a short second stage with a partially unfrozen flow. Never select a checkpoint by training loss alone.
7. Add dedicated short-utterance prompts such as ර, රා, රි, රු, ර්, and words containing them. Sentence-trained models are unreliable for isolated letters without this coverage.

## Recording pack specification

For each target speaker, record approximately:

- 300 clean sentences covering all Sinhala consonant/vowel combinations;
- 150 conversational replies and questions;
- 100 empathetic, sad, surprised, happy, and reassuring lines;
- 50 number/date/code-switching lines;
- 50 isolated-letter and short-word prompts.

Record in the same microphone/environment, with natural pauses but no background music. Keep the speaker consent and intended-use permission beside the audio manifest.

## Evaluation gate

A candidate is accepted only if native listeners prefer it over the baseline on naturalness and it does not lose intelligibility. Report MOS/pairwise preference, Sinhala ASR WER/CER, speaker-similarity score, and the fixed phonology/number/date challenge. A lower VITS loss or faster RTF is not evidence of naturalness.

## References

- [SafnasKaldeen Sinhala TTS dataset](https://huggingface.co/datasets/SafnasKaldeen/Sinhala-Text-speech-Dataset-TTS)
- [Sinhala Emotion Dataset](https://huggingface.co/datasets/ehzawad/sinhala-emotion-dataset)
- [Sinhala TTS CC dataset](https://huggingface.co/datasets/outlawmold/sinhala-tts-dataset)
- [Mozilla Common Voice Spontaneous Sinhala](https://mozilladatacollective.com/datasets/cmqi25bfw004qo507j776vn95)
- [OpenSLR SLR30](https://openslr.trmal.net/30/)
- [YouTube copyright guidance](https://support.google.com/youtube/answer/2797466)
