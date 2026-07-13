# Baseline TTS samples — dialoglk/SinhalaVITS-TTS-F1

Generated 2026-07-13 with `scripts/vits_baseline.py` (Coqui-TTS, CPU inference, female voice "Nipunika", checkpoint 210000). Pipeline: Sinhala text → bundled romanizer → VITS.

| File | Input text | Note |
|---|---|---|
| `dialoglk_F1_greeting.wav` | ආයුබෝවන්, ඔබට කොහොමද? | greeting |
| `dialoglk_F1_island.wav` | ශ්‍රී ලංකාව ඉන්දියානු සාගරයේ පිහිටි දූපතකි. | formal prose |
| `dialoglk_F1_weather.wav` | අද කාලගුණය ඉතා හොඳයි. | casual sentence |
| `dialoglk_F1_numbers_words.wav` | මට රුපියල් දෙසිය පනහක් ගෙවන්න තියෙනවා. | number written as words — works |
| `dialoglk_F1_digits_raw.wav` | මට රුපියල් 250ක් ගෙවන්න තියෙනවා. | **number as digits — model DISCARDS them** |

## Key finding

The baseline has **no text normalization**: digits are not in the model vocabulary and are silently dropped (`Character '2' not found in the vocabulary. Discarding it.`), so "රුපියල් 250ක්" is synthesized *without the amount*. Listen to `digits_raw` vs `numbers_words` back-to-back.

This is direct evidence for the text-frontend component of our proposal (adapting Google's `sinhala.grm` normalization): a demonstrable failure mode in the best available baseline that our system will fix, and a ready-made evaluation category for the paper.
