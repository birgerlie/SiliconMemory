#!/usr/bin/env python3
"""Run a combined extraction + reflection evaluation and emit one scorecard JSON."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


def _run_cmd(cmd: list[str]) -> tuple[bool, str]:
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    out = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
    return proc.returncode == 0, out.strip()


def _safe_load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _summary_reflection(bench: dict[str, Any] | None) -> dict[str, Any]:
    if not bench:
        return {}
    runs = bench.get("runs", []) or []
    elapsed = [float(r.get("elapsed_s", 0.0)) for r in runs if r.get("elapsed_s") is not None]
    processed = [int(r.get("experiences_processed", 0)) for r in runs]
    updated = [int(r.get("updated_beliefs", 0)) for r in runs]
    if not elapsed:
        return {
            "cycles": len(runs),
            "avg_elapsed_s": 0.0,
            "p95_elapsed_s": 0.0,
            "total_experiences_processed": sum(processed),
            "total_updated_beliefs": sum(updated),
        }
    sorted_elapsed = sorted(elapsed)
    p95_idx = min(len(sorted_elapsed) - 1, max(0, int(round(len(sorted_elapsed) * 0.95)) - 1))
    return {
        "cycles": len(runs),
        "avg_elapsed_s": round(statistics.mean(elapsed), 4),
        "p95_elapsed_s": round(sorted_elapsed[p95_idx], 4),
        "total_experiences_processed": sum(processed),
        "total_updated_beliefs": sum(updated),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gold-jsonl",
        type=Path,
        default=Path("/Users/birger/code/epstein/SiliconMemory/export/gold/ner_gold_from_images.jsonl"),
    )
    parser.add_argument(
        "--rules-json",
        type=Path,
        default=Path("/Users/birger/code/epstein/SiliconMemory/export/entity_rules.json"),
    )
    parser.add_argument(
        "--pred-output",
        type=Path,
        default=Path("export/metrics/ner_predictions.scorecard.jsonl"),
    )
    parser.add_argument(
        "--ner-report-output",
        type=Path,
        default=Path("export/metrics/ner_eval_report.scorecard.json"),
    )
    parser.add_argument(
        "--reflection-report-output",
        type=Path,
        default=Path("export/metrics/reflection_benchmark.scorecard.json"),
    )
    parser.add_argument(
        "--scorecard-output",
        type=Path,
        default=Path("export/metrics/extraction_reflection_scorecard.json"),
    )
    parser.add_argument("--db-path", default="silicon_memory.db")
    parser.add_argument("--db-grpc-host", default="127.0.0.1")
    parser.add_argument("--db-grpc-port", type=int, default=8643)
    parser.add_argument("--db-retry-attempts", type=int, default=2)
    parser.add_argument("--db-request-timeout-s", type=float, default=10.0)
    parser.add_argument("--user-id", default="benchmark-user")
    parser.add_argument("--tenant-id", default="benchmark-tenant")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--max-experiences", type=int, default=100)
    parser.add_argument("--llm-url", default="http://localhost:8000/v1")
    parser.add_argument("--llm-model", default="qwen3-30b")
    parser.add_argument("--llm-api-key", default="not-needed")
    parser.add_argument("--allow-weak-extraction-model", action="store_true")
    parser.add_argument("--skip-reflection-benchmark", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    predict_script = root / "predict_ner_from_gold.py"
    eval_script = root / "eval_ner_against_gold.py"
    bench_script = root / "benchmark_reflection.py"

    steps: list[dict[str, Any]] = []

    ok, out = _run_cmd(
        [
            sys.executable,
            str(predict_script),
            "--gold-jsonl",
            str(args.gold_jsonl),
            "--rules-json",
            str(args.rules_json),
            "--output",
            str(args.pred_output),
        ]
    )
    steps.append({"step": "predict_ner", "ok": ok, "log": out})

    ok2, out2 = _run_cmd(
        [
            sys.executable,
            str(eval_script),
            "--gold-jsonl",
            str(args.gold_jsonl),
            "--pred-jsonl",
            str(args.pred_output),
            "--output",
            str(args.ner_report_output),
        ]
    )
    steps.append({"step": "eval_ner", "ok": ok2, "log": out2})

    if args.skip_reflection_benchmark:
        steps.append({"step": "benchmark_reflection", "ok": False, "log": "skipped by flag"})
    else:
        ok3, out3 = _run_cmd(
            [
                sys.executable,
                str(bench_script),
                "--db-path",
                str(args.db_path),
                "--db-grpc-host",
                str(args.db_grpc_host),
                "--db-grpc-port",
                str(args.db_grpc_port),
                "--db-retry-attempts",
                str(args.db_retry_attempts),
                "--db-request-timeout-s",
                str(args.db_request_timeout_s),
                "--user-id",
                str(args.user_id),
                "--tenant-id",
                str(args.tenant_id),
                "--cycles",
                str(args.cycles),
                "--max-experiences",
                str(args.max_experiences),
                "--llm-url",
                str(args.llm_url),
                "--llm-model",
                str(args.llm_model),
                "--llm-api-key",
                str(args.llm_api_key),
                "--output",
                str(args.reflection_report_output),
            ]
            + (["--allow-weak-extraction-model"] if args.allow_weak_extraction_model else [])
        )
        steps.append({"step": "benchmark_reflection", "ok": ok3, "log": out3})

    ner_report = _safe_load_json(args.ner_report_output)
    reflection_report = _safe_load_json(args.reflection_report_output)

    scorecard = {
        "inputs": {
            "gold_jsonl": str(args.gold_jsonl),
            "rules_json": str(args.rules_json),
            "pred_output": str(args.pred_output),
            "ner_report_output": str(args.ner_report_output),
            "reflection_report_output": str(args.reflection_report_output),
        },
        "steps": steps,
        "extraction_quality": {
            "micro": (ner_report or {}).get("micro", {}),
            "by_type": (ner_report or {}).get("by_type", {}),
            "gold_docs": (ner_report or {}).get("gold_docs", 0),
            "pred_docs": (ner_report or {}).get("pred_docs", 0),
            "scored_docs": (ner_report or {}).get("scored_docs", 0),
        },
        "reflection_performance": _summary_reflection(reflection_report),
    }

    args.scorecard_output.parent.mkdir(parents=True, exist_ok=True)
    args.scorecard_output.write_text(json.dumps(scorecard, indent=2), encoding="utf-8")
    print(json.dumps(scorecard, indent=2))


if __name__ == "__main__":
    main()
