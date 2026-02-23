#!/usr/bin/env python3
"""Evaluate NER predictions against Epstein-derived gold JSONL."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

ENTITY_TYPES = ("people", "organizations", "locations")


def _normalize(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = text.casefold()
    text = re.sub(r"[\"'`]", "", text)
    return text


def _entity_set(values: Iterable[str]) -> set[str]:
    out: set[str] = set()
    for value in values:
        norm = _normalize(str(value))
        if norm:
            out.add(norm)
    return out


def _extract_entities(record: dict, key_hint: str | None = None) -> dict[str, set[str]]:
    container = None
    keys = [key_hint] if key_hint else []
    keys.extend(["gold_entities", "predicted_entities", "entities"])
    for key in keys:
        if isinstance(record.get(key), dict):
            container = record[key]
            break
    if container is None:
        container = {}
    return {et: _entity_set(container.get(et, [])) for et in ENTITY_TYPES}


def _load_gold(path: Path) -> dict[str, dict[str, set[str]]]:
    out: dict[str, dict[str, set[str]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        doc_id = row.get("doc_id")
        if doc_id:
            out[doc_id] = _extract_entities(row, key_hint="gold_entities")
    return out


def _load_pred_jsonl(path: Path) -> dict[str, dict[str, set[str]]]:
    out: dict[str, dict[str, set[str]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        doc_id = row.get("doc_id")
        if doc_id:
            out[doc_id] = _extract_entities(row, key_hint="predicted_entities")
    return out


def _load_pred_results_root(root: Path) -> dict[str, dict[str, set[str]]]:
    out: dict[str, dict[str, set[str]]] = {}
    for fp in sorted(root.glob("IMAGES*/DOJ-OGR-*.json")):
        try:
            row = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        out[fp.stem] = _extract_entities(row, key_hint="entities")
    return out


def _safe_div(a: int, b: int) -> float:
    return a / b if b else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gold-jsonl",
        type=Path,
        default=Path("/Users/birger/code/epstein/SiliconMemory/export/gold/ner_gold_from_images.jsonl"),
    )
    parser.add_argument(
        "--pred-jsonl",
        type=Path,
        default=None,
        help="JSONL containing doc_id + predicted_entities",
    )
    parser.add_argument(
        "--pred-results-root",
        type=Path,
        default=Path("/Users/birger/code/epstein/epstein-docs/results"),
        help="Directory containing IMAGES*/DOJ-OGR-*.json predictions",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/Users/birger/code/SiliconMemory/export/metrics/ner_eval_report.json"),
    )
    args = parser.parse_args()

    if not args.pred_jsonl and not args.pred_results_root:
        raise SystemExit("Provide --pred-jsonl or --pred-results-root")

    gold = _load_gold(args.gold_jsonl)
    pred = _load_pred_jsonl(args.pred_jsonl) if args.pred_jsonl else _load_pred_results_root(args.pred_results_root)

    totals = {et: {"tp": 0, "fp": 0, "fn": 0} for et in ENTITY_TYPES}
    per_doc = []
    for doc_id, gold_entities in gold.items():
        pred_entities = pred.get(doc_id, {et: set() for et in ENTITY_TYPES})
        doc_stats = {"doc_id": doc_id, "by_type": {}}
        for et in ENTITY_TYPES:
            g = gold_entities.get(et, set())
            p = pred_entities.get(et, set())
            tp = len(g & p)
            fp = len(p - g)
            fn = len(g - p)
            totals[et]["tp"] += tp
            totals[et]["fp"] += fp
            totals[et]["fn"] += fn
            doc_stats["by_type"][et] = {"tp": tp, "fp": fp, "fn": fn}
        per_doc.append(doc_stats)

    report = {"gold_docs": len(gold), "pred_docs": len(pred), "scored_docs": len(per_doc), "by_type": {}}
    micro_tp = micro_fp = micro_fn = 0
    for et in ENTITY_TYPES:
        tp, fp, fn = totals[et]["tp"], totals[et]["fp"], totals[et]["fn"]
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        report["by_type"][et] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
    micro_precision = _safe_div(micro_tp, micro_tp + micro_fp)
    micro_recall = _safe_div(micro_tp, micro_tp + micro_fn)
    micro_f1 = _safe_div(2 * micro_precision * micro_recall, micro_precision + micro_recall)
    report["micro"] = {
        "tp": micro_tp,
        "fp": micro_fp,
        "fn": micro_fn,
        "precision": round(micro_precision, 4),
        "recall": round(micro_recall, 4),
        "f1": round(micro_f1, 4),
    }
    ranked = sorted(
        per_doc,
        key=lambda d: sum(v["fp"] + v["fn"] for v in d["by_type"].values()),
        reverse=True,
    )
    report["worst_docs"] = ranked[:20]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

