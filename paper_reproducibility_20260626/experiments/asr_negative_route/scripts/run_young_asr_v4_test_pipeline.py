from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(sys.executable)
BASE = ROOT / "official_baseline" / "make_submission_forcodabench"
SAMPLE_BINARY = BASE / "binary_sample.csv"
TEST_AUDIO = ROOT / "data/privacy-constrained-raw-Young-test/audio"
ASR_JSONL = ROOT / "obs/asr/young_test_dashscope_asr_event1.jsonl"
ASR_FEATURE_DIR = ROOT / "obs/experiments/young_asr_phq9_evidence_test"
SUBMISSION_DIR = BASE / "young_asr_v4_submission"


def read_sample_ids() -> list[int]:
    with SAMPLE_BINARY.open("r", encoding="utf-8-sig", newline="") as f:
        return [int(row["id"]) for row in csv.DictReader(f)]


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("DASHSCOPE_API_KEY is not set; set it in the environment before running.", file=sys.stderr)
        return 2

    ids = [str(x) for x in read_sample_ids()]
    run(
        [
            str(PYTHON),
            "obs/scripts/transcribe_young_dashscope_asr.py",
            "--audio_root",
            str(TEST_AUDIO),
            "--out",
            str(ASR_JSONL),
            "--events",
            "event_1",
            "--ids",
            *ids,
            "--input_mode",
            "base64_compatible",
            "--chunk_sec",
            "0",
            "--max_payload_mb",
            "25",
            "--fallback_target_sr",
            "0",
            "--sleep_sec",
            "0.3",
            "--max_retries",
            "3",
        ]
    )
    run(
        [
            str(PYTHON),
            "obs/scripts/extract_young_asr_phq9_evidence_features.py",
            "--asr",
            str(ASR_JSONL),
            "--out_dir",
            str(ASR_FEATURE_DIR),
        ]
    )
    run(
        [
            str(PYTHON),
            "obs/scripts/build_young_asr_v4_test_submission.py",
            "--asr_features",
            str(ASR_FEATURE_DIR / "young_asr_phq9_evidence_features.csv"),
            "--asr_audit",
            str(ASR_FEATURE_DIR / "young_asr_phq9_evidence_audit.jsonl"),
            "--out_dir",
            str(SUBMISSION_DIR),
        ]
    )
    print(f"validated submission: {SUBMISSION_DIR}_validated/submission.zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
