#!/usr/bin/env python3
import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any

from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[2]
AUDIO_MASK_ID = 1024
EXPECTED_SAMPLE_RATE = 24000
EXPECTED_DEBUG_INPUT_KEYS = {
    "batch_input_ids_before_step_00",
    "batch_audio_mask",
    "batch_attention_mask",
    "tokens_init",
}
EXPECTED_STAGE1_DEBUG_KEYS = {
    "project_out",
    "quantizer_output",
    "fc2_output",
    "decoder_input",
    "raw_waveform",
    "final_waveform",
}
def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_sample_rate(path: Path) -> int:
    with path.open("rb") as handle:
        if handle.read(4) != b"RIFF":
            raise ValueError(f"{path}: missing RIFF header")
        handle.read(4)
        if handle.read(4) != b"WAVE":
            raise ValueError(f"{path}: missing WAVE header")

        while True:
            chunk_id = handle.read(4)
            if len(chunk_id) < 4:
                break
            chunk_size_bytes = handle.read(4)
            if len(chunk_size_bytes) < 4:
                break
            chunk_size = struct.unpack("<I", chunk_size_bytes)[0]
            if chunk_id == b"fmt ":
                fmt_data = handle.read(chunk_size)
                if len(fmt_data) < 16:
                    raise ValueError(f"{path}: invalid fmt chunk")
                return int(struct.unpack("<I", fmt_data[4:8])[0])
            handle.seek(chunk_size + (chunk_size % 2), 1)
    raise ValueError(f"{path}: fmt chunk not found")


def validate_final_tokens(path: Path) -> list[str]:
    errors: list[str] = []
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            if tensor.dim() != 2 or tensor.shape[0] != 8:
                errors.append(f"{path}: invalid token tensor shape for {key}: {tuple(tensor.shape)}")
            if (tensor == AUDIO_MASK_ID).any():
                errors.append(f"{path}: audio_mask_id={AUDIO_MASK_ID} remained in {key}")
    return errors


def validate_debug_inputs(path: Path) -> list[str]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        missing = EXPECTED_DEBUG_INPUT_KEYS.difference(handle.keys())
    return [f"{path}: missing debug input keys {sorted(missing)}"] if missing else []


def validate_debug_forward(path: Path, capture_layers: list[Any]) -> list[str]:
    expected = {"inputs_embeds"}
    for layer in capture_layers:
        if layer == "final":
            expected.add("final_hidden")
        else:
            expected.add(f"hidden_layer_{int(layer):02d}")
    with safe_open(path, framework="pt", device="cpu") as handle:
        missing = expected.difference(handle.keys())
    return [f"{path}: missing debug forward keys {sorted(missing)}"] if missing else []


def validate_stage1_debug(path: Path) -> list[str]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        if EXPECTED_STAGE1_DEBUG_KEYS.issubset(keys):
            return []
        chunk_prefixes = {
            key.rsplit("_", 2)[0]
            for key in keys
            if key.startswith("chunk_") and key.endswith("_project_out")
        }
        if not chunk_prefixes:
            missing = sorted(EXPECTED_STAGE1_DEBUG_KEYS.difference(keys))
            return [f"{path}: missing stage1 debug keys {missing}"]
        for prefix in sorted(chunk_prefixes):
            required = {
                f"{prefix}_project_out",
                f"{prefix}_quantizer_output",
                f"{prefix}_fc2_output",
                f"{prefix}_decoder_input",
                f"{prefix}_raw_waveform",
            }
            missing = sorted(required.difference(keys))
            if missing:
                return [f"{path}: missing chunked stage1 debug keys {missing}"]
        if "raw_waveform" not in keys or "final_waveform" not in keys:
            return [f"{path}: missing combined chunked waveform keys"]
    return []


def validate_debug_json(path: Path) -> tuple[list[str], dict[str, Any] | None]:
    payload = read_json(path)
    required = {"mode", "capture_steps", "capture_layers", "c_lens", "target_lens", "max_c_len", "timesteps", "schedules"}
    missing = sorted(required.difference(payload))
    if missing:
        return [f"{path}: missing debug json fields {missing}"], None
    return [], payload


def case_dir_from_manifest(manifest_path: Path, case_data: dict[str, Any]) -> Path:
    artifacts_root = manifest_path.parent.parent
    return artifacts_root / case_data["path"]


def validate_case(manifest_path: Path, case_data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    case_dir = case_dir_from_manifest(manifest_path, case_data)
    if not case_dir.exists():
        return [f"{manifest_path}: missing case directory {case_dir}"]

    for relative, expected_hash in case_data["files"].items():
        file_path = case_dir / relative.replace("/", "\\")
        if not file_path.exists():
            errors.append(f"{manifest_path}: missing {file_path}")
            continue
        actual_hash = sha256_file(file_path)
        if actual_hash != expected_hash:
            errors.append(
                f"{manifest_path}: hash mismatch for {file_path}: {actual_hash} != {expected_hash}"
            )

    final_tokens_path = case_dir / "final_tokens.safetensors"
    if final_tokens_path.exists():
        errors.extend(validate_final_tokens(final_tokens_path))

    for wav_name in ("decoded_raw.wav", "final.wav"):
        wav_path = case_dir / wav_name
        if wav_path.exists():
            sample_rate = read_sample_rate(wav_path)
            if sample_rate != EXPECTED_SAMPLE_RATE:
                errors.append(
                    f"{manifest_path}: {wav_path} has sample rate {sample_rate}, expected {EXPECTED_SAMPLE_RATE}"
                )

    determinism = case_data.get("determinism", {})
    if not determinism.get("skipped") and determinism.get("matched") is not True:
        errors.append(f"{manifest_path}: determinism mismatch for {case_data['id']}")

    if "debug.json" in case_data.get("files", {}) or case_data.get("kind") == "debug":
        debug_json_path = case_dir / "debug.json"
        if debug_json_path.exists():
            debug_errors, debug_payload = validate_debug_json(debug_json_path)
            errors.extend(debug_errors)
        else:
            debug_payload = None
        inputs_path = case_dir / "inputs.safetensors"
        if inputs_path.exists():
            errors.extend(validate_debug_inputs(inputs_path))
        forward_path = case_dir / "forward_step_00.safetensors"
        if forward_path.exists() and debug_payload is not None:
            errors.extend(validate_debug_forward(forward_path, debug_payload["capture_layers"]))
        if debug_payload is not None:
            for step in debug_payload["capture_steps"]:
                step_path = case_dir / "steps" / f"step_{int(step):02d}.safetensors"
                if not step_path.exists():
                    errors.append(f"{manifest_path}: missing debug step file {step_path}")

    stage1_debug_path = case_dir / "stage1_debug.safetensors"
    if stage1_debug_path.exists():
        errors.extend(validate_stage1_debug(stage1_debug_path))

    return errors


def validate_manifest(manifest_path: Path) -> list[str]:
    errors: list[str] = []
    manifest = read_json(manifest_path)
    runtime_path = manifest_path.parent / manifest["runtime"]
    if not runtime_path.exists():
        errors.append(f"{manifest_path}: missing runtime file {runtime_path}")

    case_count = len(manifest["cases"])
    expected_count = manifest.get("summary", {}).get("case_count")
    if expected_count is not None and expected_count != case_count:
        errors.append(
            f"{manifest_path}: summary.case_count={expected_count} but actual={case_count}"
        )

    for case_data in manifest["cases"].values():
        errors.extend(validate_case(manifest_path, case_data))

    return errors


def validate_index(index_path: Path) -> list[str]:
    errors: list[str] = []
    index = read_json(index_path)
    artifacts_root = index_path.parent
    baselines = index.get("baselines", {})
    if not baselines:
        return [f"{index_path}: no baselines declared"]
    for baseline_name, baseline in baselines.items():
        manifest_path = artifacts_root / baseline["manifest"]
        if not manifest_path.exists():
            errors.append(f"{index_path}: missing manifest for {baseline_name}: {manifest_path}")
            continue
        errors.extend(validate_manifest(manifest_path))
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Phase 0 OmniVoice reference artifacts.")
    parser.add_argument("--index", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index_path = Path(args.index).resolve()
    errors = validate_index(index_path)
    if errors:
        for error in errors:
            print(error)
        raise SystemExit(1)
    print(f"validated {index_path}")


if __name__ == "__main__":
    main()
