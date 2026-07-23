import csv
import sys
import tempfile
import unittest
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prepare_conversational_data import prepare, write_metadata


def make_wav(path: Path, seconds: float = 8.0) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\0\0" * int(16_000 * seconds))


class ConversationalManifestTests(unittest.TestCase):
    def test_rights_and_transcript_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "episode.wav"
            make_wav(audio)
            segments = [
                {
                    "utterance_id": "ok", "source_id": "source", "episode_id": "ep1",
                    "speaker_id": "host", "audio_path": "episode.wav", "start_sec": "1",
                    "end_sec": "5", "text": "සිංහල වාක්‍යයක්.", "style": "conversational",
                    "transcript_status": "manual_verified", "license_id": "lic",
                },
                {
                    "utterance_id": "bad", "source_id": "source", "episode_id": "ep1",
                    "speaker_id": "host", "audio_path": "episode.wav", "start_sec": "1",
                    "end_sec": "5", "text": "තවත් වාක්‍යයක්.", "style": "conversational",
                    "transcript_status": "auto_generated", "license_id": "lic",
                },
            ]
            rights = [{
                "license_id": "lic", "source_id": "source", "permission_to_train": "yes",
                "permission_to_release_model": "yes", "permission_to_release_audio": "no",
                "consent_scope": "training and model release",
            }]
            accepted, report = prepare(
                segments, rights, root=root, min_duration=2, max_duration=12,
                split_unit="episode", allowed_styles={"conversational"}, require_audio_files=True,
                require_model_release_permission=True,
            )
            self.assertEqual([row["id"] for row in accepted], ["ok"])
            self.assertEqual(report["rejections"], {"transcript_not_manually_verified": 1})

    def test_metadata_is_pipe_delimited_and_duplicate_text_is_rejected(self):
        segments = []
        for index in range(2):
            segments.append({
                "utterance_id": f"id{index}", "source_id": "source", "episode_id": f"ep{index}",
                "speaker_id": "host", "audio_path": f"audio{index}.wav", "start_sec": "0",
                "end_sec": "3", "text": "එකම වාක්‍යය", "style": "neutral",
                "transcript_status": "manual_verified", "license_id": "lic",
            })
        rights = [{
            "license_id": "lic", "source_id": "source", "permission_to_train": "yes",
            "permission_to_release_model": "yes", "permission_to_release_audio": "no",
            "consent_scope": "training",
        }]
        accepted, report = prepare(
            segments, rights, root=Path("."), min_duration=2, max_duration=12,
            split_unit="episode", allowed_styles={"neutral"}, require_audio_files=False,
            require_model_release_permission=True,
        )
        self.assertEqual(len(accepted), 1)
        self.assertEqual(report["rejections"], {"duplicate_text": 1})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "metadata.csv"
            write_metadata(accepted, output)
            with output.open(encoding="utf-8", newline="") as handle:
                self.assertEqual(next(csv.reader(handle, delimiter="|"))[0], "id")


if __name__ == "__main__":
    unittest.main()
