#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXPORT_REFERENCE = ROOT / "tools" / "python_reference" / "export_reference.py"
VALIDATE_REFERENCE = ROOT / "tools" / "python_reference" / "validate_reference.py"
DEFAULT_GPU_OUT_DIR = ROOT / "artifacts" / "python_reference"
DEFAULT_CPU_OUT_DIR = ROOT / "artifacts" / "python_reference_cpu_strict"
DEFAULT_INDEX_PATH = ROOT / "artifacts" / "python_reference_index.json"
DEFAULT_MODEL_DIR = ROOT / "model"
DEFAULT_REF_AUDIO = ROOT / "ref.wav"
DEFAULT_REF_TEXT = ROOT / "ref_text.txt"
CPU_STRICT_CASE_IDS = ["debug_auto_en_short", "debug_clone_user_ref"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def build_reference_index(
    gpu_out_dir: Path,
    cpu_out_dir: Path,
    model_dir: Path,
    ref_audio: Path,
    ref_text_file: Path,
    seed: int,
) -> dict[str, Any]:
    artifacts_root = gpu_out_dir.parent
    return {
        "created_at": utc_now(),
        "root": str(ROOT),
        "model_dir": str(model_dir),
        "ref_audio": str(ref_audio),
        "ref_text_file": str(ref_text_file),
        "seed": seed,
        "baselines": {
            "gpu": {
                "manifest": str((gpu_out_dir / "manifest.json").relative_to(artifacts_root)).replace("\\", "/"),
                "device": "cuda:0",
                "dtype": "float16",
            },
            "cpu_strict": {
                "manifest": str((cpu_out_dir / "manifest.json").relative_to(artifacts_root)).replace("\\", "/"),
                "device": "cpu",
                "dtype": "float32",
                "case_ids": CPU_STRICT_CASE_IDS,
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze both Phase 0 OmniVoice baselines and validate them.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--ref-audio", default=str(DEFAULT_REF_AUDIO))
    parser.add_argument("--ref-text-file", default=str(DEFAULT_REF_TEXT))
    parser.add_argument("--gpu-out-dir", default=str(DEFAULT_GPU_OUT_DIR))
    parser.add_argument("--cpu-out-dir", default=str(DEFAULT_CPU_OUT_DIR))
    parser.add_argument("--index-path", default=str(DEFAULT_INDEX_PATH))
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    ref_audio = Path(args.ref_audio).resolve()
    ref_text_file = Path(args.ref_text_file).resolve()
    gpu_out_dir = Path(args.gpu_out_dir).resolve()
    cpu_out_dir = Path(args.cpu_out_dir).resolve()
    index_path = Path(args.index_path).resolve()

    run_command(
        [
            sys.executable,
            str(EXPORT_REFERENCE),
            "--model-dir",
            str(model_dir),
            "--ref-audio",
            str(ref_audio),
            "--ref-text-file",
            str(ref_text_file),
            "--out-dir",
            str(gpu_out_dir),
            "--device",
            "cuda:0",
            "--dtype",
            "float16",
            "--seed",
            str(args.seed),
        ]
    )

    run_command(
        [
            sys.executable,
            str(EXPORT_REFERENCE),
            "--model-dir",
            str(model_dir),
            "--ref-audio",
            str(ref_audio),
            "--ref-text-file",
            str(ref_text_file),
            "--out-dir",
            str(cpu_out_dir),
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--seed",
            str(args.seed),
            "--case-ids",
            ",".join(CPU_STRICT_CASE_IDS),
        ]
    )

    index = build_reference_index(
        gpu_out_dir=gpu_out_dir,
        cpu_out_dir=cpu_out_dir,
        model_dir=model_dir,
        ref_audio=ref_audio,
        ref_text_file=ref_text_file,
        seed=args.seed,
    )
    write_json(index_path, index)

    run_command([sys.executable, str(VALIDATE_REFERENCE), "--index", str(index_path)])
    print(f"froze phase0 baselines into {index_path}")


if __name__ == "__main__":
    main()
