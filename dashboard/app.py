"""Browser dashboard for the local Sinhala VITS baseline."""

from __future__ import annotations

import io
import json
import os
import re
import sys
import time
import base64
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_file


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT.parent / "tts_demo" / "model"
DEFAULT_MALE_MODEL_DIR = ROOT.parent / "tts_demo" / "model_male"
MAX_TEXT_LENGTH = 5_000

UNITS = ["බිංදුව", "එක", "දෙක", "තුන", "හතර", "පහ", "හය", "හත", "අට", "නවය"]
TEENS = ["දහය", "එකොළහ", "දොළහ", "දහතුන", "දාහතර", "පහළොව", "දාසය", "දාහත", "දහඅට", "දහනවය"]
TENS = {2: "විස්ස", 3: "තිහ", 4: "හතළිහ", 5: "පනහ", 6: "හැට", 7: "හැත්තෑව", 8: "අසූව", 9: "අනූව"}
TENS_JOINED = {2: "විසි", 3: "තිස්", 4: "හතළිස්", 5: "පනස්", 6: "හැට", 7: "හැත්තෑ", 8: "අසූ", 9: "අනූ"}
HUNDREDS = {1: "එක", 2: "දෙ", 3: "තුන්", 4: "හාර", 5: "පන්", 6: "හය", 7: "හත්", 8: "අට", 9: "නව"}


def integer_to_sinhala(number: int) -> str:
    if number < 10:
        return UNITS[number]
    if number < 20:
        return TEENS[number - 10]
    if number < 100:
        tens, units = divmod(number, 10)
        return TENS[tens] if units == 0 else TENS_JOINED[tens] + UNITS[units]
    if number < 1_000:
        hundreds, remainder = divmod(number, 100)
        head = "සියය" if hundreds == 1 else HUNDREDS[hundreds] + "සියය"
        if not remainder:
            return head
        head = "එකසිය" if hundreds == 1 else HUNDREDS[hundreds] + "සිය"
        return head + " " + integer_to_sinhala(remainder)
    if number < 1_000_000:
        thousands, remainder = divmod(number, 1_000)
        if thousands == 1:
            head = "දහස" if not remainder else "එක්දහස්"
        elif thousands < 10:
            head = HUNDREDS[thousands] + ("දහස" if not remainder else "දහස්")
        else:
            head = integer_to_sinhala(thousands) + (" දහස" if not remainder else " දහස්")
        return head if not remainder else head + " " + integer_to_sinhala(remainder)
    return " ".join(UNITS[int(digit)] for digit in str(number))


def clean_input(text: str, normalize_numbers: bool = True) -> str:
    """Remove common Markdown markup and normalize whitespace/numerals."""
    text = re.sub(r"\[([^]]+)]\([^)]+\)", r"\1", text)
    text = re.sub(r"[*_`#>]", "", text)
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = re.sub(r"\s+", " ", text).strip()
    if normalize_numbers:
        text = re.sub(r"\d+", lambda match: integer_to_sinhala(int(match.group())), text)
    return text


def split_text(text: str, max_chars: int = 180) -> list[str]:
    """Split at Sinhala/Latin sentence boundaries, then safely by word count."""
    sentences = [part.strip() for part in re.split(r"(?<=[.!?।])\s+", text) if part.strip()]
    chunks: list[str] = []
    for sentence in sentences or [text]:
        words = sentence.split()
        current: list[str] = []
        for word in words:
            candidate = " ".join(current + [word])
            if current and len(candidate) > max_chars:
                chunks.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            chunks.append(" ".join(current))
    return chunks


class TTSEngine:
    def __init__(self, model_dir: Path, checkpoint_name: str, config_name: str):
        from TTS.utils.synthesizer import Synthesizer

        romanizer_dir = model_dir
        if not (romanizer_dir / "romanizer.py").exists():
            romanizer_dir = Path(os.environ.get("SINHALA_TTS_ROMANIZER_DIR", DEFAULT_MALE_MODEL_DIR))
        sys.path.insert(0, str(romanizer_dir))
        from romanizer import sinhala_to_roman

        self.romanize = sinhala_to_roman
        self.sample_rate = 22_050
        started = time.perf_counter()
        self.synth = Synthesizer(
            tts_checkpoint=str(model_dir / checkpoint_name),
            tts_config_path=str(model_dir / config_name),
            use_cuda=False,
        )
        self.load_seconds = time.perf_counter() - started

    def synthesize(self, chunks: list[str]) -> tuple[np.ndarray, list[str], float]:
        audio = []
        romanized = []
        silence = np.zeros(int(self.sample_rate * 0.18), dtype=np.float32)
        started = time.perf_counter()
        for index, chunk in enumerate(chunks):
            roman = self.romanize(chunk)
            romanized.append(roman)
            audio.append(np.asarray(self.synth.tts(roman), dtype=np.float32))
            if index < len(chunks) - 1:
                audio.append(silence)
        return np.concatenate(audio), romanized, time.perf_counter() - started


PAGE = """<!doctype html>
<html lang="si"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SiFi-TTS Sinhala Voice Dashboard</title>
<style>
:root{color-scheme:light;--ink:#17202a;--muted:#607080;--brand:#8b2f45;--paper:#fffdf8;--line:#ddd4c7}
*{box-sizing:border-box} body{margin:0;background:#f3eee6;color:var(--ink);font-family:system-ui,"Noto Sans Sinhala",sans-serif}
main{max-width:900px;margin:3rem auto;padding:0 1rem}.card{background:var(--paper);border:1px solid var(--line);border-radius:18px;padding:clamp(1rem,4vw,2rem);box-shadow:0 12px 40px #4c342018}
h1{margin:0;font-size:clamp(1.6rem,5vw,2.5rem)} .sub{color:var(--muted);margin:.35rem 0 1.5rem}
textarea{width:100%;min-height:230px;resize:vertical;border:1px solid #b9ad9d;border-radius:12px;padding:1rem;background:#fff;font:1.08rem/1.8 system-ui,"Noto Sans Sinhala",sans-serif}
.row{display:flex;gap:1rem;align-items:center;flex-wrap:wrap;margin-top:1rem}button{border:0;border-radius:10px;padding:.75rem 1.3rem;background:var(--brand);color:#fff;font-weight:700;cursor:pointer}button:disabled{opacity:.55;cursor:wait}
label{color:var(--muted)}audio{width:100%;margin-top:1.2rem}.meta{white-space:pre-wrap;background:#f5f0e8;border-radius:10px;padding:.9rem;margin-top:1rem;color:#43505b;font-size:.9rem;display:none}.error{color:#a01919}
</style></head><body><main><section class="card">
<h1>සිංහල හඬ</h1><p class="sub">Group 18 · Local Sinhala TTS baseline dashboard</p>
<textarea id="text" maxlength="5000" placeholder="සිංහල පෙළ මෙහි ඇතුළත් කරන්න...">ශ්‍රී ලංකාවේ ප්‍රධාන ජාතිය වන සිංහල ජනයාගේ මව් බස සිංහල වෙයි.</textarea>
<div class="row"><button id="speak">හඬ අසන්න</button><label>හඬ: <select id="voice"><option value="male">පිරිමි — Roshan</option><option value="female">කාන්තා — Nipunika</option></select></label><label><input id="normalize" type="checkbox" checked> ඉලක්කම් වචන බවට පරිවර්තනය කරන්න</label></div>
<audio id="audio" controls hidden></audio><div id="meta" class="meta"></div>
</section></main><script>
const button=document.getElementById('speak'), meta=document.getElementById('meta'), audio=document.getElementById('audio');
button.onclick=async()=>{const text=document.getElementById('text').value.trim();if(!text)return;
 button.disabled=true;button.textContent='සකසමින්…';meta.style.display='block';meta.className='meta';meta.textContent='හඬ සකසමින් පවතී…';
 try{const response=await fetch('/api/tts',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text,voice:document.getElementById('voice').value,normalize_numbers:document.getElementById('normalize').checked})});
 if(!response.ok)throw new Error(await response.text());const encoded=response.headers.get('X-TTS-Meta-B64');const details=JSON.parse(new TextDecoder().decode(Uint8Array.from(atob(encoded),c=>c.charCodeAt(0))));const blob=await response.blob();
 audio.src=URL.createObjectURL(blob);audio.hidden=false;audio.play();meta.textContent=`සැකසූ පෙළ: ${details.normalized_text}\nකොටස්: ${details.chunk_count}\nහඬ දිග: ${details.audio_seconds}s · සැකසුම් කාලය: ${details.synthesis_seconds}s · RTF: ${details.rtf}`;
 }catch(error){meta.className='meta error';meta.textContent='දෝෂයක්: '+error.message}finally{button.disabled=false;button.textContent='හඬ අසන්න'}};
</script></body></html>"""


def create_app(model_dir: Path | None = None, male_model_dir: Path | None = None) -> Flask:
    app = Flask(__name__)
    voice_configs = {
        "female": {
            "model_dir": model_dir or Path(os.environ.get("SINHALA_TTS_FEMALE_MODEL_DIR", DEFAULT_MODEL_DIR)),
            "checkpoint": "Nipunika_210000.pth", "config": "Nipunika_config.json",
        },
        "male": {
            "model_dir": male_model_dir or Path(os.environ.get("SINHALA_TTS_MALE_MODEL_DIR", DEFAULT_MALE_MODEL_DIR)),
            "checkpoint": os.environ.get("SINHALA_TTS_MALE_CHECKPOINT", "Roshan_270000.pth"),
            "config": os.environ.get("SINHALA_TTS_MALE_CONFIG", "Roshan_config.json"),
        },
    }
    engines: dict[str, TTSEngine] = {}

    def get_engine(voice: str) -> TTSEngine:
        if voice not in voice_configs:
            raise ValueError(f"unknown voice: {voice}")
        if voice not in engines:
            selected = voice_configs[voice]
            engines[voice] = TTSEngine(selected["model_dir"], selected["checkpoint"], selected["config"])
        return engines[voice]

    @app.get("/")
    def index():
        return PAGE

    @app.get("/api/health")
    def health():
        return jsonify({
            "status": "ok", "loaded_voices": sorted(engines),
            "voices": {name: str(config["model_dir"]) for name, config in voice_configs.items()},
        })

    @app.post("/api/tts")
    def tts():
        payload = request.get_json(silent=True) or {}
        raw_text = str(payload.get("text", "")).strip()
        if not raw_text:
            return jsonify({"error": "text is required"}), 400
        if len(raw_text) > MAX_TEXT_LENGTH:
            return jsonify({"error": f"text exceeds {MAX_TEXT_LENGTH} characters"}), 400
        voice = str(payload.get("voice", "male"))
        if voice not in voice_configs:
            return jsonify({"error": f"unknown voice: {voice}"}), 400
        normalized = clean_input(raw_text, bool(payload.get("normalize_numbers", True)))
        chunks = split_text(normalized)
        if not chunks:
            return jsonify({"error": "text is empty after cleaning"}), 400
        active_engine = get_engine(voice)
        waveform, romanized, synthesis_seconds = active_engine.synthesize(chunks)
        audio_seconds = len(waveform) / active_engine.sample_rate
        buffer = io.BytesIO()
        active_engine.synth.save_wav(waveform, buffer)
        buffer.seek(0)
        response = send_file(buffer, mimetype="audio/wav", download_name="sinhala-tts.wav")
        metadata = json.dumps({
            "voice": voice,
            "normalized_text": normalized,
            "chunk_count": len(chunks),
            "romanized_chunks": romanized,
            "audio_seconds": round(audio_seconds, 3),
            "synthesis_seconds": round(synthesis_seconds, 3),
            "rtf": round(synthesis_seconds / audio_seconds, 3),
        }, ensure_ascii=False)
        response.headers["X-TTS-Meta-B64"] = base64.b64encode(metadata.encode("utf-8")).decode("ascii")
        return response

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=int(os.environ.get("PORT", "7861")), debug=False)
