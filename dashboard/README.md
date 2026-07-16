# Sinhala voice dashboard

Run from the repository root using the existing isolated Coqui environment:

```bash
../tts_demo/venv/bin/python dashboard/app.py
```

Then open <http://127.0.0.1:7861>. The default model directory is
`../tts_demo/model`; override it when needed:

```bash
SINHALA_TTS_MODEL_DIR=/path/to/model ../tts_demo/venv/bin/python dashboard/app.py
```

The dashboard removes common Markdown syntax, optionally expands integer digits,
splits long input into safe chunks, and returns a single combined WAV. The model
is loaded lazily on the first synthesis request.
