#!/usr/bin/env python3
"""Evaluate named-entity extraction against JSON gold sets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _extract_entities(payload: Any) -> set[tuple[str, str]]:
    """Extract (text,label) tuples from flexible JSON shapes."""
    entities: list[Any] = []
    if isinstance(payload, dict):
        if isinstance(payload.get("entities"), list):
            entities = payload["entities"]
        elif isinstance(payload.get("named_entities"), list):
            entities = payload["named_entities"]
    elif isinstance(payload, list):
        entities = payload

    normalized: set[tuple[str, str]] = set()
    for item in entities:
        if not isinstance(item, dict):
            continue
        text = (
            item.get("text")
            or item.get("name")
            or item.get("entity")
            or item.get("value")
            or ""
        )
        label = (
            item.get("label")
            or item.get("type")
            or item.get("category")
            or "UNKNOWN"
        )
        text_norm = str(text).strip().lower()
        label_norm = str(label).strip().upper()
        if text_norm:
            normalized.add((text_norm, label_norm))
    return normalized


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate named entities against gold JSON files")
    p.add_argument("--gold-dir", type=Path, required=True, help="Directory with gold JSON files")
    p.add_argument("--pred-dir", type=Path, required=True, help="Directory with prediction JSON files")
    args = p.parse_args()

    gold_files = sorted(args.gold_dir.glob("*.json"))
    if not gold_files:
        raise SystemExit(f"No gold JSON files found in {args.gold_dir}")

    tp = fp = fn = 0
    compared = 0
    missing = []

    for gold_path in gold_files:
        pred_path = args.pred_dir / gold_path.name
        if not pred_path.exists():
            missing.append(gold_path.name)
            continue

        gold_entities = _extract_entities(_load_json(gold_path))
        pred_entities = _extract_entities(_load_json(pred_path))
        tp += len(gold_entities & pred_entities)
        fp += len(pred_entities - gold_entities)
        fn += len(gold_entities - pred_entities)
        compared += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    print(json.dumps(
        {
            "files_compared": compared,
            "missing_predictions": len(missing),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "missing_files": missing[:20],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()

