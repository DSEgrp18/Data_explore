"""Generate baseline Sinhala TTS samples with dialoglk/SinhalaVITS-TTS-F1 (Coqui VITS)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/dialoglk")
from TTS.utils.synthesizer import Synthesizer
from romanizer import sinhala_to_roman

BASE = os.path.dirname(os.path.abspath(__file__)) + "/dialoglk"
OUT = os.environ.get("OUT_DIR", "samples")
os.makedirs(OUT, exist_ok=True)

synth = Synthesizer(
    tts_checkpoint=f"{BASE}/Nipunika_210000.pth",
    tts_config_path=f"{BASE}/Nipunika_config.json",
    use_cuda=False,
)

texts = [
    ("greeting", "ආයුබෝවන්, ඔබට කොහොමද?"),
    ("island", "ශ්‍රී ලංකාව ඉන්දියානු සාගරයේ පිහිටි දූපතකි."),
    ("weather", "අද කාලගුණය ඉතා හොඳයි."),
    ("numbers_words", "මට රුපියල් දෙසිය පනහක් ගෙවන්න තියෙනවා."),
    ("digits_raw", "මට රුපියල් 250ක් ගෙවන්න තියෙනවා."),
]

for name, t in texts:
    roman = sinhala_to_roman(t)
    print(f"{name}: {t} -> {roman}")
    wav = synth.tts(roman)
    path = os.path.join(OUT, f"dialoglk_F1_{name}.wav")
    synth.save_wav(wav, path)
    print("wrote", path)
