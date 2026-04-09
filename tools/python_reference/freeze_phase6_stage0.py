#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXPORT_REFERENCE = ROOT / "tools" / "python_reference" / "export_reference.py"
VALIDATE_REFERENCE = ROOT / "tools" / "python_reference" / "validate_reference.py"
DEFAULT_GPU_OUT_DIR = ROOT / "artifacts" / "python_reference_stage0_deterministic"
DEFAULT_CPU_OUT_DIR = ROOT / "artifacts" / "python_reference_stage0_deterministic_cpu_strict"
DEFAULT_INDEX_PATH = ROOT / "artifacts" / "python_reference_stage0_deterministic_index.json"
DEFAULT_MODEL_DIR = ROOT / "model"
DEFAULT_REF_AUDIO = ROOT / "ref.wav"
DEFAULT_REF_TEXT = ROOT / "ref_text.txt"
GPU_CASE_IDS = [
    "det_auto_en_short",
    "det_design_en_british",
    "det_clone_user_ref",
    "det_auto_long_chunked",
]
CPU_STRICT_CASE_IDS = [
    "det_debug_auto_en_short",
    "det_debug_clone_user_ref",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
                "case_ids": GPU_CASE_IDS,
            },
            "cpu_strict": {
                "manifest": str((cpu_out_dir / "manifest.json").relative_to(artifacts_root)).replace("\\", "/"),
                "device": "cpu",
                "dtype": "float32",
                "case_ids": CPU_STRICT_CASE_IDS,
            },
        },
    }


def export_cases(
    *,
    model_dir: Path,
    ref_audio: Path,
    ref_text_file: Path,
    out_dir: Path,
    device: str,
    dtype: str,
    seed: int,
    case_ids: list[str],
) -> None:
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
            str(out_dir),
            "--device",
            device,
            "--dtype",
            dtype,
            "--seed",
            str(seed),
            "--case-ids",
            ",".join(case_ids),
        ]
    )


def merge_debug_exports(case_roots: list[Path], cpu_out_dir: Path, case_ids: list[str]) -> None:
    if cpu_out_dir.exists():
        shutil.rmtree(cpu_out_dir)
    cpu_out_dir.mkdir(parents=True, exist_ok=True)

    manifests = [read_json(root / "manifest.json") for root in case_roots]
    base_manifest = manifests[0]
    shutil.copy2(case_roots[0] / "runtime.json", cpu_out_dir / "runtime.json")

    merged_cases: dict[str, Any] = {}
    for root, manifest in zip(case_roots, manifests):
        for case_id, case_data in manifest["cases"].items():
            shutil.copytree(root / case_id, cpu_out_dir / case_id)
            normalized = json.loads(json.dumps(case_data))
            normalized["path"] = f"{cpu_out_dir.name}/{case_id}"
            merged_cases[case_id] = normalized

    merged_manifest = {
        "created_at": utc_now(),
        "root": str(ROOT),
        "out_dir": str(cpu_out_dir),
        "runtime": "runtime.json",
        "environment_check": base_manifest["environment_check"],
        "smoke_test": base_manifest["smoke_test"],
        "cases_requested": case_ids,
        "cases": merged_cases,
        "summary": {
            "case_count": len(merged_cases),
            "main_cases": [],
            "debug_cases": case_ids,
        },
        "runtime_snapshot": base_manifest["runtime_snapshot"],
    }
    write_json(cpu_out_dir / "manifest.json", merged_manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze deterministic Phase 6 stage0 OmniVoice baselines and validate them."
    )
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

    export_cases(
        model_dir=model_dir,
        ref_audio=ref_audio,
        ref_text_file=ref_text_file,
        out_dir=gpu_out_dir,
        device="cuda:0",
        dtype="float16",
        seed=args.seed,
        case_ids=GPU_CASE_IDS,
    )

    with tempfile.TemporaryDirectory(prefix="phase6-stage0-", dir=cpu_out_dir.parent) as tmp_dir:
        case_roots = []
        for case_id in CPU_STRICT_CASE_IDS:
            tmp_out_dir = Path(tmp_dir) / case_id
            export_cases(
                model_dir=model_dir,
                ref_audio=ref_audio,
                ref_text_file=ref_text_file,
                out_dir=tmp_out_dir,
                device="cpu",
                dtype="float32",
                seed=args.seed,
                case_ids=[case_id],
            )
            case_roots.append(tmp_out_dir)
        merge_debug_exports(case_roots, cpu_out_dir, CPU_STRICT_CASE_IDS)

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
    print(f"froze phase6 stage0 baselines into {index_path}")


if __name__ == "__main__":
    main()
