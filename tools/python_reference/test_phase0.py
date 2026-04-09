import hashlib
import importlib.util
import json
import struct
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import torch
from safetensors.torch import save_file as save_safetensors


ROOT = Path(__file__).resolve().parents[2]
EXPORT_REFERENCE_PATH = ROOT / "tools" / "python_reference" / "export_reference.py"
FREEZE_PHASE0_PATH = ROOT / "tools" / "python_reference" / "freeze_phase0.py"
FREEZE_PHASE6_STAGE0_PATH = ROOT / "tools" / "python_reference" / "freeze_phase6_stage0.py"
VALIDATE_REFERENCE_PATH = ROOT / "tools" / "python_reference" / "validate_reference.py"


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_wav(path: Path, sample_rate: int = 24000) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * 8)


def write_float_wav(path: Path, sample_rate: int = 24000) -> None:
    samples = [0.0, 0.25, -0.25, 0.0]
    data = b"".join(struct.pack("<f", sample) for sample in samples)
    fmt_chunk = struct.pack(
        "<IHHIIHH",
        16,
        3,
        1,
        sample_rate,
        sample_rate * 4,
        4,
        32,
    )
    riff_size = 4 + (8 + len(fmt_chunk)) + (8 + len(data))
    with path.open("wb") as handle:
        handle.write(b"RIFF")
        handle.write(struct.pack("<I", riff_size))
        handle.write(b"WAVE")
        handle.write(b"fmt ")
        handle.write(fmt_chunk)
        handle.write(b"data")
        handle.write(struct.pack("<I", len(data)))
        handle.write(data)


class ExportReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.export_reference = load_module("phase0_export_reference", EXPORT_REFERENCE_PATH)

    def test_parse_case_ids_parses_csv(self):
        self.assertIsNone(self.export_reference.parse_case_ids(None))
        self.assertEqual(
            self.export_reference.parse_case_ids(" auto_en_short , debug_auto_en_short "),
            ["auto_en_short", "debug_auto_en_short"],
        )

    def test_resolve_cases_filters_selected_ids(self):
        cases = self.export_reference.resolve_cases(
            ref_audio=Path("h:/omnivoice/ref.wav"),
            ref_text="reference text",
            case_ids=["auto_en_short", "debug_auto_en_short"],
        )
        self.assertEqual([case["id"] for case in cases], ["auto_en_short", "debug_auto_en_short"])

    def test_resolve_cases_rejects_unknown_case_ids(self):
        with self.assertRaises(ValueError):
            self.export_reference.resolve_cases(
                ref_audio=Path("h:/omnivoice/ref.wav"),
                ref_text="reference text",
                case_ids=["missing_case"],
            )


class FreezePhase0Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.freeze_phase0 = load_module("phase0_freeze_phase0", FREEZE_PHASE0_PATH)

    def test_build_reference_index_uses_relative_manifest_paths(self):
        index = self.freeze_phase0.build_reference_index(
            gpu_out_dir=Path("h:/omnivoice/artifacts/python_reference"),
            cpu_out_dir=Path("h:/omnivoice/artifacts/python_reference_cpu_strict"),
            model_dir=Path("h:/omnivoice/model"),
            ref_audio=Path("h:/omnivoice/ref.wav"),
            ref_text_file=Path("h:/omnivoice/ref_text.txt"),
            seed=1234,
        )
        self.assertEqual(index["baselines"]["gpu"]["manifest"], "python_reference/manifest.json")
        self.assertEqual(
            index["baselines"]["cpu_strict"]["manifest"],
            "python_reference_cpu_strict/manifest.json",
        )
        self.assertEqual(index["seed"], 1234)


class FreezePhase6Stage0Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.freeze_phase6_stage0 = load_module(
            "phase6_freeze_phase6_stage0",
            FREEZE_PHASE6_STAGE0_PATH,
        )

    def test_build_reference_index_uses_relative_manifest_paths(self):
        index = self.freeze_phase6_stage0.build_reference_index(
            gpu_out_dir=Path("h:/omnivoice/artifacts/python_reference_stage0_deterministic"),
            cpu_out_dir=Path("h:/omnivoice/artifacts/python_reference_stage0_deterministic_cpu_strict"),
            model_dir=Path("h:/omnivoice/model"),
            ref_audio=Path("h:/omnivoice/ref.wav"),
            ref_text_file=Path("h:/omnivoice/ref_text.txt"),
            seed=1234,
        )
        self.assertEqual(
            index["baselines"]["gpu"]["manifest"],
            "python_reference_stage0_deterministic/manifest.json",
        )
        self.assertEqual(
            index["baselines"]["cpu_strict"]["manifest"],
            "python_reference_stage0_deterministic_cpu_strict/manifest.json",
        )
        self.assertEqual(index["baselines"]["gpu"]["case_ids"][0], "det_auto_en_short")
        self.assertEqual(index["baselines"]["cpu_strict"]["case_ids"][1], "det_debug_clone_user_ref")
        self.assertEqual(index["seed"], 1234)


class ValidateReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.validate_reference = load_module("phase0_validate_reference", VALIDATE_REFERENCE_PATH)

    def _create_fake_manifest_tree(self, root: Path) -> Path:
        out_dir = root / "python_reference"
        out_dir.mkdir(parents=True)

        runtime_path = out_dir / "runtime.json"
        runtime_path.write_text('{"runtime": true}\n', encoding="utf-8")

        case_dir = out_dir / "debug_case"
        steps_dir = case_dir / "steps"
        case_dir.mkdir()
        steps_dir.mkdir()

        (case_dir / "case.json").write_text('{"id": "debug_case"}\n', encoding="utf-8")
        (case_dir / "prepared.json").write_text('{"chunk_plan": {"kind": "single"}}\n', encoding="utf-8")
        (case_dir / "debug.json").write_text('{"capture_steps": [0, 15, 31]}\n', encoding="utf-8")

        save_safetensors(
            {"prepared_input_ids": torch.zeros((1, 8, 1), dtype=torch.int64)},
            str(case_dir / "inputs.safetensors"),
        )
        save_safetensors(
            {"tokens": torch.zeros((8, 1), dtype=torch.int64)},
            str(case_dir / "final_tokens.safetensors"),
        )
        save_safetensors(
            {"c_logits": torch.zeros((1, 8, 1, 1025), dtype=torch.float32)},
            str(case_dir / "steps" / "step_00.safetensors"),
        )
        save_safetensors(
            {"c_logits": torch.zeros((1, 8, 1, 1025), dtype=torch.float32)},
            str(case_dir / "steps" / "step_15.safetensors"),
        )
        save_safetensors(
            {"c_logits": torch.zeros((1, 8, 1, 1025), dtype=torch.float32)},
            str(case_dir / "steps" / "step_31.safetensors"),
        )
        save_safetensors(
            {"final_hidden": torch.zeros((1, 1, 1), dtype=torch.float32)},
            str(case_dir / "forward_step_00.safetensors"),
        )
        write_wav(case_dir / "decoded_raw.wav")
        write_wav(case_dir / "final.wav")

        files = {}
        for relative in [
            "case.json",
            "prepared.json",
            "debug.json",
            "inputs.safetensors",
            "final_tokens.safetensors",
            "decoded_raw.wav",
            "final.wav",
            "steps/step_00.safetensors",
            "steps/step_15.safetensors",
            "steps/step_31.safetensors",
            "forward_step_00.safetensors",
        ]:
            files[relative] = sha256_file(case_dir / relative.replace("/", "\\"))

        manifest = {
            "out_dir": str(out_dir),
            "runtime": "runtime.json",
            "cases_requested": ["debug_case"],
            "cases": {
                "debug_case": {
                    "id": "debug_case",
                    "kind": "debug",
                    "path": "python_reference/debug_case",
                    "files": files,
                    "verification": {
                        "token_shapes": {"tokens": [8, 1]},
                        "audio_shape": [1, 8],
                        "sample_rate": 24000,
                    },
                    "determinism": {"matched": True},
                }
            },
        }
        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return manifest_path

    def test_validate_manifest_detects_missing_forward_debug_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = self._create_fake_manifest_tree(Path(tmp_dir))
            missing_path = manifest_path.parent / "debug_case" / "forward_step_00.safetensors"
            missing_path.unlink()
            errors = self.validate_reference.validate_manifest(manifest_path)
            self.assertTrue(any("forward_step_00.safetensors" in error for error in errors))

    def test_read_sample_rate_supports_ieee_float_wav(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = Path(tmp_dir) / "float.wav"
            write_float_wav(wav_path, sample_rate=24000)
            self.assertEqual(self.validate_reference.read_sample_rate(wav_path), 24000)


if __name__ == "__main__":
    unittest.main()
