#!/usr/bin/env python3
"""Generate NER predictions from gold JSONL full_text using extractor regexes."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from silicon_memory.entities.types import DetectorRule, ExtractorRule


ENTITY_TYPE_MAP = {
    "person": "people",
    "people": "people",
    "organization": "organizations",
    "organisation": "organizations",
    "org": "organizations",
    "organizations": "organizations",
    "location": "locations",
    "locations": "locations",
    "place": "locations",
}

PEOPLE_TITLE_RE = re.compile(
    r"\b(?:Mr|Ms|Mrs|Miss|Dr|Judge|Honorable)\.?\s+([A-Z][a-z]+(?:\s+[A-Z]\.)?(?:\s+[A-Z][a-z]+)+)\b"
)
PEOPLE_NAME_RE = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z]\.)?(?:\s+[A-Z][a-z]+){1,2})\b"
)
ORG_RE = re.compile(
    r"\b([A-Z][A-Za-z&.,'\\-]*(?:\s+[A-Z][A-Za-z&.,'\\-]*){0,5}\s+(?:Court|Services|Department|District|Herald|Reporters|University|Inc\\.?|LLC|Corp\\.?|P\\.C\\.))\b"
)
LOCATION_RE = re.compile(
    r"\b(?:New York|Palm Beach|Florida|Manhattan|Brooklyn|Queens|Bronx|Staten Island|Washington|Miami|Chicago|Los Angeles|California|Texas|Virginia|Maryland)\b"
)


def _load_rules(path: Path) -> tuple[list[DetectorRule], list[ExtractorRule]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    detectors: list[DetectorRule] = []
    extractors: list[ExtractorRule] = []
    for d in data.get("detectors", []):
        if not d.get("id") or not d.get("pattern"):
            continue
        detectors.append(
            DetectorRule(
                id=d["id"],
                pattern=d["pattern"],
                description=d.get("description", d["id"]),
            )
        )
    for e in data.get("extractors", []):
        if not e.get("id") or not e.get("pattern") or not e.get("entity_type"):
            continue
        extractors.append(
            ExtractorRule(
                id=e["id"],
                entity_type=e["entity_type"],
                detector_ids=e.get("detector_ids", []),
                pattern=e["pattern"],
                normalize_template=e.get("normalize_template", "{match}"),
                confidence=float(e.get("confidence", 1.0)),
            )
        )
    return detectors, extractors


def _normalise_bucket(entity_type: str) -> str | None:
    return ENTITY_TYPE_MAP.get(entity_type.strip().lower())


def _predict_record(extractors: list[tuple[ExtractorRule, re.Pattern[str]]], row: dict[str, Any]) -> dict[str, Any]:
    text = row.get("full_text", "") or ""
    buckets: dict[str, set[str]] = {
        "people": set(),
        "organizations": set(),
        "locations": set(),
    }
    for rule, pattern in extractors:
        bucket = _normalise_bucket(rule.entity_type)
        if not bucket:
            continue
        for m in pattern.finditer(text):
            cleaned = m.group(0).strip()
            if cleaned:
                buckets[bucket].add(cleaned)

    # Heuristic fallback for coarse NER buckets.
    for m in PEOPLE_TITLE_RE.finditer(text):
        buckets["people"].add(m.group(1).strip())
    for m in PEOPLE_NAME_RE.finditer(text):
        name = m.group(1).strip()
        # Skip obvious organization-like matches.
        if any(tok in name for tok in ("Court", "District", "Services", "Department", "Inc", "LLC", "P.C")):
            continue
        buckets["people"].add(name)
    for m in ORG_RE.finditer(text):
        buckets["organizations"].add(m.group(1).strip())
    for m in LOCATION_RE.finditer(text):
        buckets["locations"].add(m.group(0).strip())

    return {
        "doc_id": row.get("doc_id"),
        "predicted_entities": {
            "people": sorted(buckets["people"]),
            "organizations": sorted(buckets["organizations"]),
            "locations": sorted(buckets["locations"]),
        },
    }


def _run(args: argparse.Namespace) -> None:
    detectors, extractors = _load_rules(args.rules_json)
    compiled_extractors: list[tuple[ExtractorRule, re.Pattern[str]]] = []
    for rule in extractors:
        try:
            compiled_extractors.append((rule, re.compile(rule.pattern)))
        except re.error:
            continue

    lines = args.gold_jsonl.read_text(encoding="utf-8").splitlines()
    outputs: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        pred = _predict_record(compiled_extractors, row)
        outputs.append(json.dumps(pred, ensure_ascii=True))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(outputs) + ("\n" if outputs else ""), encoding="utf-8")
    print(
        json.dumps(
            {
                "gold_rows": len([ln for ln in lines if ln.strip()]),
                "pred_rows": len(outputs),
                "rules_detectors": len(detectors),
                "rules_extractors": len(extractors),
                "compiled_extractors": len(compiled_extractors),
                "output": str(args.output),
            },
            indent=2,
        )
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--gold-jsonl",
        type=Path,
        default=Path("/Users/birger/code/epstein/SiliconMemory/export/gold/ner_gold_from_images.jsonl"),
    )
    p.add_argument(
        "--rules-json",
        type=Path,
        default=Path("/Users/birger/code/epstein/SiliconMemory/export/entity_rules.json"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("/Users/birger/code/SiliconMemory/export/metrics/ner_predictions.from_rules.jsonl"),
    )
    args = p.parse_args()
    _run(args)


if __name__ == "__main__":
    main()
