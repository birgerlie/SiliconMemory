#!/usr/bin/env python3
"""Clean legacy temporal hypothesis artifacts (e.g., 1994 -> current year)."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from silicon_memory.memory.silicondb_router import SiliconMemory
from silicon_memory.reflection.hypothesis import HypothesisGenerator
from silicon_memory.security.types import UserContext


def _norm(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _key(subject: str, object_value: str) -> str:
    return f"{_norm(subject)}|timeline_progresses_from_to|{_norm(object_value)}"


def _as_tags(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, set | tuple):
        return [str(v) for v in value]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _extract_year(text: str) -> int | None:
    m = re.fullmatch(r"(19|20)\d{2}", text.strip())
    if not m:
        return None
    return int(text.strip())


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    user = UserContext(user_id=args.user_id, tenant_id=args.tenant_id)
    memory = SiliconMemory(
        path=Path(args.db_path),
        user_context=user,
        db_grpc_host=args.db_grpc_host,
        db_grpc_port=args.db_grpc_port,
    )
    backend = memory._backend
    current_year = datetime.now(timezone.utc).year

    try:
        # Recompute expected temporal hypotheses with patched logic.
        generator = HypothesisGenerator(memory=memory, llm=object())
        expected = await generator._generate_global_temporal_hypotheses(limit=5000)  # noqa: SLF001
        expected_keys = {
            _key(c.subject or "", c.object or "")
            for c in expected
            if c.subject and c.object and c.predicate == "timeline_progresses_from_to"
        }

        triples = backend._query_triples(k=50000)
        scanned = 0
        cleaned = 0
        candidates: list[dict[str, Any]] = []

        for t in triples:
            metadata = getattr(t, "metadata", {}) or {}
            external_id = getattr(t, "external_id", "")
            if not backend._can_access(metadata, _external_id=external_id):
                continue

            predicate = str(getattr(t, "predicate", "") or "").strip()
            if predicate != "timeline_progresses_from_to":
                continue

            tags = {tag.lower() for tag in _as_tags(metadata.get("tags"))}
            if "hypothesis" not in tags:
                continue

            scanned += 1
            subject = str(getattr(t, "subject", "") or "")
            object_value = str(getattr(t, "object_value", "") or "")
            sm = metadata.get("source_metadata") if isinstance(metadata, dict) else {}
            sm = sm if isinstance(sm, dict) else {}

            temporal_to = str(sm.get("temporal_to") or "").strip()
            to_year = _extract_year(temporal_to)
            support_predicates = [str(p).lower() for p in (sm.get("support_predicates") or [])]

            is_current_year_tail = (to_year == current_year) or (f"-> {current_year}" in object_value)
            uses_event_support = "event" in support_predicates
            still_expected = _key(subject, object_value) in expected_keys

            # Legacy artifact signature:
            # - timeline endpoint equals current ingestion year
            # - evidence includes generic "event" predicate
            # - no longer produced by patched global temporal generator
            artifact = is_current_year_tail and uses_event_support and not still_expected
            if not artifact:
                continue

            entry = {
                "external_id": external_id,
                "subject": subject,
                "object": object_value,
                "temporal_from": sm.get("temporal_from"),
                "temporal_to": sm.get("temporal_to"),
                "support_predicates": support_predicates,
            }
            candidates.append(entry)

            if args.apply:
                doc = backend._db.get(external_id)
                if not isinstance(doc, dict):
                    continue
                current_meta = dict(doc.get("metadata", {}) or {})
                raw_tags = _as_tags(current_meta.get("tags"))
                next_tags = []
                for tag in raw_tags:
                    norm = tag.strip().lower()
                    if norm == "hypothesis":
                        continue
                    next_tags.append(tag)
                if "temporal_artifact" not in {t.lower() for t in next_tags}:
                    next_tags.append("temporal_artifact")

                current_meta["tags"] = next_tags
                current_meta["status"] = "rejected"
                current_meta["status_reason"] = (
                    "Removed by temporal cleanup: legacy relative-date timeline artifact"
                )
                current_meta["temporal_cleanup"] = {
                    "applied_at": datetime.now(timezone.utc).isoformat(),
                    "method": "legacy_relative_year_filter",
                    "current_year": current_year,
                }
                backend._db.update(external_id, metadata=current_meta)
                cleaned += 1

        return {
            "tenant_id": args.tenant_id,
            "user_id": args.user_id,
            "current_year": current_year,
            "expected_temporal_candidates": len(expected_keys),
            "timeline_hypotheses_scanned": scanned,
            "artifacts_detected": len(candidates),
            "artifacts_cleaned": cleaned,
            "apply": bool(args.apply),
            "candidates": candidates,
        }
    finally:
        memory.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db-path", default="silicon_memory.db")
    p.add_argument("--db-grpc-host", default="127.0.0.1")
    p.add_argument("--db-grpc-port", type=int, default=8643)
    p.add_argument("--user-id", required=True)
    p.add_argument("--tenant-id", required=True)
    p.add_argument("--apply", action="store_true", help="Apply cleanup updates")
    p.add_argument("--output", default="")
    args = p.parse_args()

    out = asyncio.run(_run(args))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
