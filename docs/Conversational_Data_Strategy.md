# Conversational Sinhala data strategy

## Goal

Reduce the formal news-reading style of existing Sinhala voices while improving
pronunciation coverage. This needs conversational prosody in the training data;
architecture changes alone cannot manufacture a style absent from the corpus.

## Source priority

1. **Record a consented speaker yourselves.** Record one male speaker reading a
   balanced mix of conversational dialogue and ordinary prose. This gives exact
   transcripts, consistent audio, explicit voice-cloning consent, and the lowest
   publication/legal risk.
2. **Partner with Sinhala creators.** Ask podcasters, educators, audiobook
   narrators, or YouTubers for written permission covering download, segmentation,
   ML/TTS training, publication of derived data/checkpoints, and voice synthesis.
   Prefer the creator's original audio files and scripts over platform downloads.
3. **Use clearly licensed speech corpora.** Verify that the license covers model
   training and redistribution of the intended artifacts. “Publicly accessible”
   is not the same as openly licensed.
4. **Commercial films only with explicit permission.** Movie copyright, music,
   actor/performer rights, and voice-likeness concerns make films unsuitable as
   an unlicensed training source.

YouTube's Terms restrict downloading, reproducing, or modifying content unless
the service expressly permits it or YouTube and the relevant rights holders give
permission. Therefore the project must not build an automated movie scraper as
its default data pipeline.

## Why movie subtitles are not enough

- subtitles paraphrase, shorten, censor, or translate speech;
- timing often covers a whole scene rather than one utterance;
- music, effects, reverberation, and overlapping actors contaminate the signal;
- character names do not reliably identify the active speaker;
- mixing actors without speaker labels can damage speaker identity and stability;
- emotional acting can improve expressiveness but reduce neutral TTS robustness.

## Pipeline for authorized long-form content

1. Store permission, license, source ID, speaker consent, and allowed outputs in
   a provenance manifest before processing.
2. Extract lossless mono audio from the rights-holder-provided master.
3. Apply speech/music detection and reject—not merely enhance—segments with music,
   overlap, clipping, heavy reverberation, or strong background effects.
4. Run speaker diarization and retain one consenting target speaker per model, or
   preserve reliable speaker IDs for a multi-speaker model.
5. Obtain an initial Sinhala ASR transcript, then force-align it to the audio.
6. Manually correct every retained transcript. TTS needs closer text/audio
   agreement than ASR training.
7. Split into roughly 2–12 second utterances at natural phrase boundaries.
8. Normalize loudness and sample rate without aggressive denoising artifacts.
9. Deduplicate text/audio and create speaker- and source-disjoint evaluation sets.
10. Manually audit a random sample and publish rejection statistics.

Recommended automatic filters include duration, SNR, clipping ratio, ASR/text CER,
speech probability, overlap probability, and speaker-embedding consistency. None
replaces listening.

## Feasible first collection

Record 5–10 hours from one consenting male speaker in a quiet treated room:

- 60% neutral conversational sentences;
- 15% questions, requests, and short responses;
- 10% numbers, dates, currency, names, and abbreviations;
- 10% Sinhala-English code-switching and common loanwords;
- 5% expressive but non-shouted speech.

Prompts should cover aspiration, prenasalized consonants, vowel length, conjuncts,
rare words, and the pronunciation errors found during baseline listening. Record
multiple natural versions of a smaller subset to model prosodic variation.

## Modeling recommendation

Do not continue training the current single-speaker VITS checkpoint on a mixture
of movie actors. For the proposed SiFi-TTS experiment:

- use the clean PathNirvana corpus for stable linguistic adaptation;
- add one licensed/consented conversational male speaker as a named speaker/style;
- use speaker/style conditioning or prompt-based F5-TTS rather than blending
  identities into one voice;
- balance conversational and read speech batches;
- compare read-only versus read-plus-conversational data at equal training steps;
- evaluate naturalness, pronunciation CER, speaker similarity, and style preference.

The existing Roshan male VITS model is a useful baseline voice, not evidence that
the proposed conversational-data method has already improved the model.
