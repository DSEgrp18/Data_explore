# Target-Speaker Recording Pack

The next naturalness model must be recorded by the same speakers as the current Roshan and Nipunika voices. Do not mix another podcast host into either voice.

## Recording conditions

- Record each prompt as a separate WAV file named exactly after its prompt ID.
- Use a quiet room, the same microphone, and the same distance for the whole session.
- Use mono, 16-bit PCM, 22050 Hz WAV.
- Leave a short natural pause before and after each sentence.
- Speak naturally; do not imitate a newsreader.
- Record neutral, conversational, empathetic, sad, happy, and questioning styles as labelled.
- Do not include music, room conversation, overlapping speakers, clipping, or aggressive noise reduction.

The starter prompt list is:

evaluation/target_speaker_recording_prompts.csv

Place recordings in a folder such as data/target_speaker/roshan_recordings/, then validate them:

```bash
/home/kusal/sportfit/.venv/bin/python scripts/prepare_target_speaker_vits_dataset.py \
  --prompts evaluation/target_speaker_recording_prompts.csv \
  --recordings-dir data/target_speaker/roshan_recordings \
  --output-root runs/target_vits_roshan \
  --romanizer ../tts_demo/model_male/romanizer.py \
  --base-config ../tts_demo/model_male/Roshan_config.json \
  --speaker roshan
```

For Nipunika, use the female romanizer/config and a separate output directory. The validator rejects missing files, wrong sample rates, stereo audio, non-16-bit WAVs, invalid model characters, duplicate text, and recordings outside the duration range.

The first fine-tuning stage will freeze the acoustic decoder and update only safe duration/prosody components. A candidate is accepted only after native-speaker A/B listening and intelligibility checks.
