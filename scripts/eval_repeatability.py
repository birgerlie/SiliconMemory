#!/usr/bin/env python3
"""Run two isolated full cycles and report repeatability deltas."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MetricThreshold:
    path: str
    max_abs_delta: float


DEFAULT_THRESHOLDS: tuple[MetricThreshold, ...] = (
    MetricThreshold("patterns_total", 8.0),
    MetricThreshold("committed_triplet_beliefs", 8.0),
    MetricThreshold("reflection.experiences_processed", 12.0),
    MetricThreshold("dream.procedures_created", 4.0),
    MetricThreshold("dream.questions_generated", 4.0),
)


def _run_cmd(cmd: list[str]) -> tuple[bool, str]:
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    out = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
    return proc.returncode == 0, out.strip()


def _safe_load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _metric_value(data: dict[str, Any], dotted_path: str, default: float = 0.0) -> float:
    cur: Any = data
    for part in dotted_path.split("."):
        if not isinstance(cur, dict):
            return default
        cur = cur.get(part)
    try:
        return float(cur)
    except Exception:
        return default


def _build_gate_snapshot(batch: dict[str, Any], dream: dict[str, Any]) -> dict[str, Any]:
    sanity = (batch or {}).get("retrieval_sanity", {}) or {}
    progress = sanity.get("extraction_progress", {}) or {}
    doc_count = int((batch or {}).get("doc_count", 0) or 0)
    expected_visibility = int((sanity.get("direct_experience_visibility", {}) or {}).get("expected", doc_count) or doc_count)
    visible = int((sanity.get("direct_experience_visibility", {}) or {}).get("visible", 0) or 0)
    extracted = int(progress.get("extracted", 0) or 0)
    unprocessed = int(sanity.get("unprocessed_experiences_count", 0) or 0)
    unextracted = int(sanity.get("unextracted_experiences_count", 0) or 0)
    procedures_count = int((dream or {}).get("retrieval_sanity", {}).get("procedures_count", 0) or 0)
    open_questions = int((dream or {}).get("retrieval_sanity", {}).get("open_questions_count", 0) or 0)

    checks = {
        "direct_visibility_complete": visible >= expected_visibility and expected_visibility > 0,
        "all_extracted": extracted >= doc_count and doc_count > 0,
        "none_unprocessed": unprocessed == 0,
        "none_unextracted": unextracted == 0,
    }
    return {
        "doc_count": doc_count,
        "direct_visibility": {"expected": expected_visibility, "visible": visible},
        "extraction_progress": {
            "extracted": extracted,
            "unprocessed": unprocessed,
            "unextracted": unextracted,
        },
        "procedures_count": procedures_count,
        "open_questions_count": open_questions,
        "checks": checks,
        "checks_passed": all(checks.values()),
    }


def evaluate_repeatability(
    batch_a: dict[str, Any],
    batch_b: dict[str, Any],
    dream_a: dict[str, Any],
    dream_b: dict[str, Any],
    thresholds: tuple[MetricThreshold, ...] = DEFAULT_THRESHOLDS,
    require_open_questions_min: int = 0,
    require_procedures_min: int = 0,
) -> dict[str, Any]:
    merged_a = dict(batch_a or {})
    merged_b = dict(batch_b or {})
    merged_a["dream"] = (dream_a or {}).get("dream", {}) or {}
    merged_b["dream"] = (dream_b or {}).get("dream", {}) or {}

    metrics: list[dict[str, Any]] = []
    metrics_ok = True
    for threshold in thresholds:
        a_val = _metric_value(merged_a, threshold.path, default=0.0)
        b_val = _metric_value(merged_b, threshold.path, default=0.0)
        abs_delta = abs(a_val - b_val)
        passed = abs_delta <= threshold.max_abs_delta
        metrics_ok = metrics_ok and passed
        metrics.append(
            {
                "metric": threshold.path,
                "run_a": a_val,
                "run_b": b_val,
                "abs_delta": round(abs_delta, 4),
                "max_abs_delta": threshold.max_abs_delta,
                "passed": passed,
            },
        )

    gate_a = _build_gate_snapshot(batch_a, dream_a)
    gate_b = _build_gate_snapshot(batch_b, dream_b)

    for gate in (gate_a, gate_b):
        checks = dict(gate.get("checks", {}))
        if require_open_questions_min > 0:
            checks["min_open_questions"] = int(gate.get("open_questions_count", 0)) >= require_open_questions_min
        if require_procedures_min > 0:
            checks["min_procedures"] = int(gate.get("procedures_count", 0)) >= require_procedures_min
        gate["checks"] = checks
        gate["checks_passed"] = all(checks.values())
    gate_ok = bool(gate_a.get("checks_passed")) and bool(gate_b.get("checks_passed"))

    return {
        "gate_a": gate_a,
        "gate_b": gate_b,
        "metric_deltas": metrics,
        "metrics_passed": metrics_ok,
        "gates_passed": gate_ok,
        "overall_passed": metrics_ok and gate_ok,
    }


def _run_single_cycle(
    python_exe: str,
    root: Path,
    args: argparse.Namespace,
    suffix: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    steps: list[dict[str, Any]] = []
    tenant = f"{args.tenant_prefix}-{suffix}"
    user = f"{args.user_prefix}-{suffix}"

    batch_out = args.output_dir / f"repeatability_batch_{suffix}.json"
    dream_out = args.output_dir / f"repeatability_dream_{suffix}.json"

    batch_cmd = [
        python_exe,
        str(root / "run_batch_extract_reflect.py"),
        "--docs-glob",
        args.docs_glob,
        "--max-docs",
        str(args.max_docs),
        "--db-path",
        args.db_path,
        "--db-grpc-host",
        args.db_grpc_host,
        "--db-grpc-port",
        str(args.db_grpc_port),
        "--tenant-id",
        tenant,
        "--user-id",
        user,
        "--rules-json",
        args.rules_json,
        "--llm-mode",
        args.llm_mode,
        "--llm-model",
        args.llm_model,
        "--llm-temperature",
        str(args.llm_temperature),
        "--llm-url",
        args.llm_url,
        "--llm-api-key",
        args.llm_api_key,
        "--llm-cache-mode",
        args.llm_cache_mode,
        "--reflect-limit",
        str(args.reflect_limit),
        "--extraction-max-chars",
        str(args.extraction_max_chars),
        "--extraction-max-items",
        str(args.extraction_max_items),
        "--extraction-max-tokens",
        str(args.extraction_max_tokens),
        "--extraction-cache-dir",
        args.extraction_cache_dir,
        "--output",
        str(batch_out),
    ]
    if args.allow_weak_extraction_model:
        batch_cmd.append("--allow-weak-extraction-model")
    ok_batch, out_batch = _run_cmd(batch_cmd)
    steps.append({"step": f"batch_{suffix}", "ok": ok_batch, "log": out_batch, "output": str(batch_out)})
    batch = _safe_load_json(batch_out) if ok_batch else {}

    dream_cmd = [
        python_exe,
        str(root / "benchmark_reflection.py"),
        "--db-path",
        args.db_path,
        "--db-grpc-host",
        args.db_grpc_host,
        "--db-grpc-port",
        str(args.db_grpc_port),
        "--db-retry-attempts",
        str(args.db_retry_attempts),
        "--db-request-timeout-s",
        str(args.db_request_timeout_s),
        "--tenant-id",
        tenant,
        "--user-id",
        user,
        "--llm-mode",
        args.llm_mode,
        "--llm-model",
        args.llm_model,
        "--llm-url",
        args.llm_url,
        "--llm-api-key",
        args.llm_api_key,
        "--llm-cache-mode",
        args.llm_cache_mode,
        "--cycles",
        "0",
        "--max-experiences",
        str(args.reflect_limit),
        "--include-dream",
        "--dream-enable-procedure-detection",
        "--dream-enable-question-generation",
        "--disable-hypothesis-generation",
        "--disable-hypothesis-validation",
        "--output",
        str(dream_out),
    ]
    if args.allow_weak_extraction_model:
        dream_cmd.append("--allow-weak-extraction-model")
    ok_dream, out_dream = _run_cmd(dream_cmd)
    steps.append({"step": f"dream_{suffix}", "ok": ok_dream, "log": out_dream, "output": str(dream_out)})
    dream = _safe_load_json(dream_out) if ok_dream else {}

    return batch, dream, steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs-glob", default="/Users/birger/code/epstein/epstein-docs/results/IMAGES*/DOJ-OGR-*.json")
    parser.add_argument("--max-docs", type=int, default=5)
    parser.add_argument("--db-path", default="silicon_memory.db")
    parser.add_argument("--db-grpc-host", default="127.0.0.1")
    parser.add_argument("--db-grpc-port", type=int, default=8643)
    parser.add_argument("--db-retry-attempts", type=int, default=2)
    parser.add_argument("--db-request-timeout-s", type=float, default=10.0)
    parser.add_argument("--tenant-prefix", default="repeatability-tenant")
    parser.add_argument("--user-prefix", default="repeatability-user")
    parser.add_argument("--rules-json", default="/Users/birger/code/epstein/SiliconMemory/bootstrap_rules.json")
    parser.add_argument("--llm-mode", choices=["local", "mock", "anthropic"], default="local")
    parser.add_argument("--llm-model", default="qwen3-30b")
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument("--llm-url", default="http://localhost:8000/v1")
    parser.add_argument("--llm-api-key", default="not-needed")
    parser.add_argument("--allow-weak-extraction-model", action="store_true")
    parser.add_argument("--llm-cache-mode", choices=["off", "record_replay", "replay_only"], default="off")
    parser.add_argument("--reflect-limit", type=int, default=120)
    parser.add_argument("--extraction-max-chars", type=int, default=22000)
    parser.add_argument("--extraction-max-items", type=int, default=8)
    parser.add_argument("--extraction-max-tokens", type=int, default=1800)
    parser.add_argument("--extraction-cache-dir", default="")
    parser.add_argument("--output-dir", type=Path, default=Path("export/metrics/stabilization"))
    parser.add_argument("--output", type=Path, default=Path("export/metrics/stabilization/repeatability_report.json"))
    parser.add_argument("--require-open-questions-min", type=int, default=0)
    parser.add_argument("--require-procedures-min", type=int, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    root = Path(__file__).resolve().parent
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix_a = f"{run_stamp}_a"
    suffix_b = f"{run_stamp}_b"

    batch_a, dream_a, steps_a = _run_single_cycle(sys.executable, root, args, suffix_a)
    batch_b, dream_b, steps_b = _run_single_cycle(sys.executable, root, args, suffix_b)

    eval_report = evaluate_repeatability(
        batch_a,
        batch_b,
        dream_a,
        dream_b,
        require_open_questions_min=max(0, int(args.require_open_questions_min)),
        require_procedures_min=max(0, int(args.require_procedures_min)),
    )
    report = {
        "run_stamp_utc": run_stamp,
        "inputs": {
            "docs_glob": args.docs_glob,
            "max_docs": args.max_docs,
            "llm_mode": args.llm_mode,
            "llm_model": args.llm_model,
            "llm_temperature": args.llm_temperature,
            "llm_cache_mode": args.llm_cache_mode,
            "tenant_prefix": args.tenant_prefix,
            "user_prefix": args.user_prefix,
        },
        "steps": steps_a + steps_b,
        "run_a": {"batch": batch_a, "dream": dream_a},
        "run_b": {"batch": batch_b, "dream": dream_b},
        "evaluation": eval_report,
    }

    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
