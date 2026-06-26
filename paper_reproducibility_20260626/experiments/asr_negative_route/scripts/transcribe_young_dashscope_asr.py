# -*- coding: utf-8 -*-
"""
Batch transcribe Young audio files with DashScope Qwen-ASR.

The script intentionally does not store API keys. Set DASHSCOPE_API_KEY in the
environment before running.

Example:
    python obs/scripts/transcribe_young_dashscope_asr.py --events event_1 --limit 3
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import math
import os
import sys
import time
import wave
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import requests
from scipy.io import wavfile
from scipy.signal import resample_poly


ROOT = Path(__file__).resolve().parents[2]
YOUNG_TRAIN_AUDIO = ROOT / "data" / "privacy-constrained-raw-Young-train" / "audio"
YOUNG_LABEL_CSV = ROOT / "data" / "Train-MPDD-Young" / "Young" / "split_labels_train.csv"
DEFAULT_OUT = ROOT / "obs" / "asr" / "young_dashscope_asr.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe Young audio with DashScope Qwen-ASR.")
    parser.add_argument("--audio_root", type=Path, default=YOUNG_TRAIN_AUDIO)
    parser.add_argument("--label_csv", type=Path, default=YOUNG_LABEL_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--events", nargs="+", default=["event_1"], choices=["event_1", "event_2", "event_3"])
    parser.add_argument("--ids", nargs="*", type=int, default=None, help="Optional explicit sample IDs.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap for quick tests.")
    parser.add_argument("--model", default="qwen3-asr-flash")
    parser.add_argument("--base_url", default="https://dashscope.aliyuncs.com/api/v1")
    parser.add_argument("--compatible_base_url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--enable_itn", action="store_true", help="Enable inverse text normalization.")
    parser.add_argument(
        "--input_mode",
        default="base64_compatible",
        choices=["base64_compatible", "dashscope_path", "dashscope_file_uri"],
        help=(
            "base64_compatible avoids DashScope SDK's temporary OSS upload by sending "
            "resampled WAV chunks as Base64 Data URLs."
        ),
    )
    parser.add_argument("--chunk_sec", type=float, default=0.0, help="0 means send the whole audio segment.")
    parser.add_argument("--target_sr", type=int, default=16000)
    parser.add_argument("--fallback_target_sr", type=int, default=8000)
    parser.add_argument("--max_payload_mb", type=float, default=9.5)
    parser.add_argument("--sleep_sec", type=float, default=0.2)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run already present sample/event pairs.",
    )
    return parser.parse_args()


def load_ids(label_csv: Path, explicit_ids: Optional[List[int]]) -> List[int]:
    if explicit_ids:
        return sorted(set(explicit_ids))
    ids: List[int] = []
    with label_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            ids.append(int(row["ID"]))
    return sorted(ids)


def existing_keys(path: Path) -> set[tuple[int, str]]:
    keys: set[tuple[int, str]] = set()
    if not path.exists():
        return keys
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("ok") and "sample_id" in row and "event" in row:
                keys.add((int(row["sample_id"]), str(row["event"])))
    return keys


def response_to_dict(response: Any) -> Dict[str, Any]:
    if isinstance(response, dict):
        return response
    if hasattr(response, "to_dict"):
        try:
            obj = response.to_dict()
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    data: Dict[str, Any] = {}
    for key in ["status_code", "request_id", "code", "message", "output", "usage"]:
        if hasattr(response, key):
            try:
                data[key] = getattr(response, key)
            except Exception:
                pass
    return data


def extract_text_and_annotations(obj: Any) -> tuple[str, list[dict[str, Any]]]:
    texts: List[str] = []
    annotations: List[dict[str, Any]] = []

    def walk(x: Any) -> None:
        if isinstance(x, str):
            if x.strip():
                texts.append(x.strip())
            return
        if isinstance(x, list):
            for item in x:
                walk(item)
            return
        if not isinstance(x, dict):
            return

        if isinstance(x.get("annotations"), list):
            annotations.extend([a for a in x["annotations"] if isinstance(a, dict)])

        # DashScope result_format="message" often nests text in message.content.
        if "text" in x and isinstance(x["text"], str):
            texts.append(x["text"].strip())
        content = x.get("content")
        if isinstance(content, str):
            if content.strip():
                texts.append(content.strip())
        elif isinstance(content, list):
            walk(content)

        for key in ["output", "choices", "message", "delta"]:
            if key in x:
                walk(x[key])

    walk(obj)
    # Keep order but remove exact duplicates introduced by recursive traversal.
    deduped: List[str] = []
    seen = set()
    for text in texts:
        if text and text not in seen:
            deduped.append(text)
            seen.add(text)
    return "".join(deduped), annotations


def load_resampled_mono(audio_path: Path, target_sr: int = 16000) -> tuple[int, np.ndarray]:
    sr, audio = wavfile.read(audio_path)
    if audio.ndim > 1:
        audio = audio.astype(np.float32).mean(axis=1)
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / max(1, np.iinfo(audio.dtype).max)
    else:
        audio = audio.astype(np.float32)
        max_abs = float(np.max(np.abs(audio))) if len(audio) else 1.0
        if max_abs > 1.5:
            audio = audio / 32768.0
    if sr != target_sr:
        from math import gcd

        g = gcd(sr, target_sr)
        audio = resample_poly(audio, target_sr // g, sr // g).astype(np.float32)
        sr = target_sr
    audio = np.clip(audio, -1.0, 1.0)
    return sr, audio


def wav_data_uri(audio: np.ndarray, sr: int) -> str:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:audio/wav;base64,{data}"


def estimate_wav_data_url_mb(n_samples: int) -> float:
    wav_bytes = 44 + n_samples * 2
    base64_bytes = math.ceil(wav_bytes / 3) * 4
    prefix_bytes = len("data:audio/wav;base64,")
    return (base64_bytes + prefix_bytes) / (1024 * 1024)


def extract_text_from_compatible_response(payload: Dict[str, Any]) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except Exception:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"].strip())
            elif isinstance(item, str):
                parts.append(item.strip())
        return "".join(parts)
    return ""


def call_compatible_base64(audio_path: Path, args: argparse.Namespace, api_key: str) -> Dict[str, Any]:
    sr, audio = load_resampled_mono(audio_path, target_sr=args.target_sr)
    payload_mb = estimate_wav_data_url_mb(len(audio))
    used_fallback_sr = False
    if args.chunk_sec <= 0 and payload_mb > args.max_payload_mb and args.fallback_target_sr:
        sr, audio = load_resampled_mono(audio_path, target_sr=args.fallback_target_sr)
        payload_mb = estimate_wav_data_url_mb(len(audio))
        used_fallback_sr = True
    if args.chunk_sec <= 0 and payload_mb > args.max_payload_mb:
        return {
            "ok": False,
            "text": "",
            "annotations": [],
            "chunks": [],
            "resampled_sr": sr,
            "duration_sec": round(len(audio) / sr, 3),
            "estimated_payload_mb": round(payload_mb, 3),
            "response": {
                "mode": "base64_compatible",
                "error": "estimated payload exceeds max_payload_mb and chunking is disabled",
                "max_payload_mb": args.max_payload_mb,
            },
        }
    chunk_len = len(audio) if args.chunk_sec <= 0 else max(1, int(args.chunk_sec * sr))
    chunks = []
    endpoint = args.compatible_base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for chunk_idx, start in enumerate(range(0, len(audio), chunk_len), 1):
        chunk = audio[start : start + chunk_len]
        if len(chunk) < sr * 0.2:
            continue
        data_uri = wav_data_uri(chunk, sr)
        payload: Dict[str, Any] = {
            "model": args.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": data_uri},
                        }
                    ],
                }
            ],
            "stream": False,
            "asr_options": {"enable_itn": bool(args.enable_itn)},
        }
        if args.language:
            payload["asr_options"]["language"] = args.language
        response = None
        response_json: Dict[str, Any]
        for attempt in range(1, args.max_retries + 1):
            try:
                response = requests.post(endpoint, headers=headers, json=payload, timeout=180)
                try:
                    response_json = response.json()
                except Exception:
                    response_json = {"raw_text": response.text}
                break
            except Exception as exc:
                response_json = {"error": repr(exc), "attempt": attempt}
                if attempt < args.max_retries:
                    time.sleep(args.sleep_sec * attempt)
        text = extract_text_from_compatible_response(response_json)
        chunks.append(
            {
                "chunk_index": chunk_idx,
                "start_sec": round(start / sr, 3),
                "end_sec": round(min(start + len(chunk), len(audio)) / sr, 3),
                "status_code": response.status_code if response is not None else None,
                "ok": bool(response is not None and response.ok and text),
                "text": text,
                "response": response_json,
            }
        )
        if args.sleep_sec > 0:
            time.sleep(args.sleep_sec)

    full_text = "".join(c["text"] for c in chunks if c.get("text"))
    ok = bool(full_text) and all(c.get("status_code") == 200 for c in chunks)
    return {
        "ok": ok,
        "text": full_text,
        "annotations": [],
        "chunks": chunks,
        "resampled_sr": sr,
        "used_fallback_sr": used_fallback_sr,
        "duration_sec": round(len(audio) / sr, 3),
        "estimated_payload_mb": round(payload_mb, 3),
        "response": {"mode": "base64_compatible", "chunk_count": len(chunks)},
    }


def call_dashscope_sdk(
    audio_path: Path,
    args: argparse.Namespace,
    api_key: str,
) -> Dict[str, Any]:
    import dashscope

    dashscope.base_http_api_url = args.base_url

    abs_path = audio_path.resolve()
    audio_value = f"file://{abs_path}" if args.input_mode == "dashscope_file_uri" else str(abs_path)
    messages = [{"role": "user", "content": [{"audio": audio_value}]}]
    asr_options: Dict[str, Any] = {"enable_itn": bool(args.enable_itn)}
    if args.language:
        asr_options["language"] = args.language

    response = dashscope.MultiModalConversation.call(
        api_key=api_key,
        model=args.model,
        messages=messages,
        result_format="message",
        asr_options=asr_options,
    )
    response_dict = response_to_dict(response)
    text, annotations = extract_text_and_annotations(response_dict)
    ok = not response_dict.get("code") and bool(text)
    return {
        "ok": ok,
        "text": text,
        "annotations": annotations,
        "response": response_dict,
    }


def call_asr(audio_path: Path, args: argparse.Namespace, api_key: str) -> Dict[str, Any]:
    if args.input_mode == "base64_compatible":
        return call_compatible_base64(audio_path, args, api_key)
    return call_dashscope_sdk(audio_path, args, api_key)


def iter_jobs(args: argparse.Namespace) -> Iterable[tuple[int, str, Path]]:
    ids = load_ids(args.label_csv, args.ids)
    count = 0
    for sample_id in ids:
        for event in args.events:
            path = args.audio_root / str(sample_id) / f"{event}.wav"
            if not path.exists():
                continue
            yield sample_id, event, path
            count += 1
            if args.limit is not None and count >= args.limit:
                return


def main() -> int:
    args = parse_args()
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("DASHSCOPE_API_KEY is not set.", file=sys.stderr)
        return 2

    if args.input_mode != "base64_compatible":
        try:
            import dashscope  # noqa: F401
        except ImportError:
            print(
                "dashscope is not installed. Install it with: pip install dashscope",
                file=sys.stderr,
            )
            return 2
    elif requests is None:
        print(
            "requests is not installed.",
            file=sys.stderr,
        )
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = existing_keys(args.out) if not args.overwrite else set()
    jobs = list(iter_jobs(args))
    if not args.overwrite:
        jobs = [(sid, ev, p) for sid, ev, p in jobs if (sid, ev) not in done]

    print(f"jobs={len(jobs)} out={args.out}")
    with args.out.open("a", encoding="utf-8") as f:
        for idx, (sample_id, event, audio_path) in enumerate(jobs, 1):
            row: Dict[str, Any] = {
                "sample_id": sample_id,
                "event": event,
                "audio_path": str(audio_path),
                "model": args.model,
                "base_url": args.base_url,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            for attempt in range(1, args.max_retries + 1):
                try:
                    result = call_asr(audio_path, args, api_key)
                    row.pop("error", None)
                    row.pop("attempt", None)
                    row.update(result)
                    break
                except Exception as exc:
                    row.update({"ok": False, "error": repr(exc), "attempt": attempt})
                    if attempt < args.max_retries:
                        time.sleep(args.sleep_sec * attempt)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            text_preview = row.get("text", "")[:80].replace("\n", " ")
            print(f"[{idx}/{len(jobs)}] ID={sample_id} {event} ok={row.get('ok')} {text_preview}")
            if args.sleep_sec > 0:
                time.sleep(args.sleep_sec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
