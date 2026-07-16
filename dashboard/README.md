# Sinhala voice dashboard

Run from the repository root using the existing isolated Coqui environment:

```bash
../tts_demo/venv/bin/python dashboard/app.py
```

Then open <http://127.0.0.1:7861>. The defaults are `../tts_demo/model` for
Nipunika (female) and `../tts_demo/model_male` for Roshan (male). Override them
when needed:

```bash
SINHALA_TTS_FEMALE_MODEL_DIR=/path/to/female \
SINHALA_TTS_MALE_MODEL_DIR=/path/to/male \
../tts_demo/venv/bin/python dashboard/app.py
```

The dashboard offers male/female voices, removes common Markdown syntax,
optionally expands integer digits, splits long input into safe chunks, and
returns a single combined WAV. Each model is loaded lazily on first use.
