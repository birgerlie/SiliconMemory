"""EntityRuleStore — persist entity rules and aliases in SiliconDB."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from silicon_memory.entities.types import DetectorRule, ExtractorRule

logger = logging.getLogger(__name__)

# Prefix for all system-scoped entity rule documents.
_PREFIX = "_system/_rules/"

# Node types stored in SiliconDB.
NODE_TYPE_DETECTOR = "entity_detector"
NODE_TYPE_EXTRACTOR = "entity_extractor"
NODE_TYPE_ALIAS = "entity_alias"

# External ID for the manifest document that tracks all rule/alias IDs.
_MANIFEST_ID = f"{_PREFIX}manifest"


def _detector_ext_id(rule_id: str) -> str:
    return f"{_PREFIX}detector-{rule_id}"


def _extractor_ext_id(rule_id: str) -> str:
    return f"{_PREFIX}extractor-{rule_id}"


def _alias_ext_id(alias: str) -> str:
    normalized = re.sub(r"\s+", " ", alias.strip().lower())
    return f"{_PREFIX}alias-{normalized}"


def _detector_text(rule: DetectorRule) -> str:
    return f"Entity detector: {rule.description}. Pattern: {rule.pattern}"


def _extractor_text(rule: ExtractorRule) -> str:
    parts = [f"Entity extractor for {rule.entity_type}. Pattern: {rule.pattern}"]
    if rule.examples:
        parts.append(f"Examples: {', '.join(rule.examples)}")
    if rule.context_examples:
        parts.append(f"Context: {', '.join(rule.context_examples)}")
    return ". ".join(parts)


def _alias_text(alias: str, canonical_id: str, entity_type: str) -> str:
    return f"Entity alias: {alias} maps to {canonical_id} (type: {entity_type})"


class EntityRuleStore:
    """Persist entity rules (DetectorRule, ExtractorRule) and alias mappings in SiliconDB.

    Uses a system-level namespace (``_system/_rules/``) outside per-user data.
    A manifest document tracks all IDs for reliable enumeration
    (workaround for SiliconDB #125).
    """

    def __init__(
        self,
        db_path: str | Path,
        language: str = "english",
        auto_embedder: bool = True,
        embedder_model: str = "base",
    ) -> None:
        try:
            from silicondb import SiliconDB
        except ImportError as e:
            raise ImportError(
                "SiliconDB is required. Install with: pip install silicondb"
            ) from e

        self._db = SiliconDB(
            path=str(db_path),
            language=language,
            auto_embedder=auto_embedder,
            embedder_model=embedder_model,
        )
        self._manifest: dict[str, list[str]] = {
            "detectors": [],
            "extractors": [],
            "aliases": [],
        }
        self._load_manifest()

    # ------------------------------------------------------------------
    # Manifest helpers
    # ------------------------------------------------------------------

    def _load_manifest(self) -> None:
        """Load the manifest from SiliconDB (or initialise empty)."""
        try:
            doc = self._db.get(_MANIFEST_ID)
            if doc:
                raw = (doc.get("metadata") or {}).get("manifest")
                if raw:
                    self._manifest = json.loads(raw) if isinstance(raw, str) else raw
                    return
        except Exception:
            pass
        # First run or missing — start empty
        self._manifest = {"detectors": [], "extractors": [], "aliases": []}

    def _save_manifest(self) -> None:
        """Persist the manifest to SiliconDB."""
        metadata = {"manifest": json.dumps(self._manifest)}
        try:
            self._db.update(external_id=_MANIFEST_ID, text="manifest", metadata=metadata)
        except Exception:
            self._db.ingest(
                external_id=_MANIFEST_ID,
                text="manifest",
                metadata=metadata,
                node_type="manifest",
            )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_detector(self, rule: DetectorRule) -> None:
        """Persist a DetectorRule (upsert)."""
        ext_id = _detector_ext_id(rule.id)
        text = _detector_text(rule)
        metadata = {
            "rule_id": rule.id,
            "pattern": rule.pattern,
            "description": rule.description,
            "created_at": rule.created_at.isoformat(),
        }
        try:
            self._db.update(external_id=ext_id, text=text, metadata=metadata)
        except Exception:
            self._db.ingest(
                external_id=ext_id,
                text=text,
                metadata=metadata,
                node_type=NODE_TYPE_DETECTOR,
            )
        if rule.id not in self._manifest["detectors"]:
            self._manifest["detectors"].append(rule.id)
            self._save_manifest()

    def save_extractor(self, rule: ExtractorRule) -> None:
        """Persist an ExtractorRule (upsert)."""
        ext_id = _extractor_ext_id(rule.id)
        text = _extractor_text(rule)
        metadata = {
            "rule_id": rule.id,
            "entity_type": rule.entity_type,
            "detector_ids": json.dumps(rule.detector_ids),
            "pattern": rule.pattern,
            "normalize_template": rule.normalize_template,
            "examples": json.dumps(rule.examples),
            "context_examples": json.dumps(rule.context_examples),
            "confidence": rule.confidence,
            "context_threshold": rule.context_threshold,
            "created_at": rule.created_at.isoformat(),
        }
        if rule.context_embedding is not None:
            metadata["context_embedding"] = json.dumps(rule.context_embedding)
        try:
            self._db.update(external_id=ext_id, text=text, metadata=metadata)
        except Exception:
            self._db.ingest(
                external_id=ext_id,
                text=text,
                metadata=metadata,
                node_type=NODE_TYPE_EXTRACTOR,
            )
        if rule.id not in self._manifest["extractors"]:
            self._manifest["extractors"].append(rule.id)
            self._save_manifest()

    def save_alias(self, alias: str, canonical_id: str, entity_type: str) -> None:
        """Persist an alias mapping (upsert)."""
        ext_id = _alias_ext_id(alias)
        text = _alias_text(alias, canonical_id, entity_type)
        normalized = re.sub(r"\s+", " ", alias.strip().lower())
        metadata = {
            "alias": alias,
            "normalized": normalized,
            "canonical_id": canonical_id,
            "entity_type": entity_type,
        }
        try:
            self._db.update(external_id=ext_id, text=text, metadata=metadata)
        except Exception:
            self._db.ingest(
                external_id=ext_id,
                text=text,
                metadata=metadata,
                node_type=NODE_TYPE_ALIAS,
            )
        if normalized not in self._manifest["aliases"]:
            self._manifest["aliases"].append(normalized)
            self._save_manifest()

    # ------------------------------------------------------------------
    # Load all (via manifest)
    # ------------------------------------------------------------------

    def load_all_detectors(self) -> list[DetectorRule]:
        """Load every persisted DetectorRule."""
        rules: list[DetectorRule] = []
        stale_ids: list[str] = []
        for rule_id in list(self._manifest["detectors"]):
            ext_id = _detector_ext_id(rule_id)
            try:
                doc = self._db.get(ext_id)
                if not doc:
                    stale_ids.append(rule_id)
                    continue
                meta = doc.get("metadata") or {}
                rules.append(DetectorRule(
                    id=meta["rule_id"],
                    pattern=meta["pattern"],
                    description=meta.get("description", ""),
                    created_at=datetime.fromisoformat(meta["created_at"])
                    if meta.get("created_at") else datetime.now(timezone.utc),
                ))
            except Exception:
                stale_ids.append(rule_id)
        if stale_ids:
            for sid in stale_ids:
                self._manifest["detectors"].remove(sid)
            self._save_manifest()
        return rules

    def load_all_extractors(self) -> list[ExtractorRule]:
        """Load every persisted ExtractorRule."""
        rules: list[ExtractorRule] = []
        stale_ids: list[str] = []
        for rule_id in list(self._manifest["extractors"]):
            ext_id = _extractor_ext_id(rule_id)
            try:
                doc = self._db.get(ext_id)
                if not doc:
                    stale_ids.append(rule_id)
                    continue
                meta = doc.get("metadata") or {}
                context_embedding = None
                if meta.get("context_embedding"):
                    context_embedding = json.loads(meta["context_embedding"])
                rules.append(ExtractorRule(
                    id=meta["rule_id"],
                    entity_type=meta.get("entity_type", ""),
                    detector_ids=json.loads(meta.get("detector_ids", "[]")),
                    pattern=meta["pattern"],
                    normalize_template=meta.get("normalize_template", "{match}"),
                    examples=json.loads(meta.get("examples", "[]")),
                    context_examples=json.loads(meta.get("context_examples", "[]")),
                    context_embedding=context_embedding,
                    confidence=float(meta.get("confidence", 1.0)),
                    context_threshold=float(meta.get("context_threshold", 0.6)),
                    created_at=datetime.fromisoformat(meta["created_at"])
                    if meta.get("created_at") else datetime.now(timezone.utc),
                ))
            except Exception:
                stale_ids.append(rule_id)
        if stale_ids:
            for sid in stale_ids:
                self._manifest["extractors"].remove(sid)
            self._save_manifest()
        return rules

    def load_all_aliases(self) -> list[tuple[str, str, str]]:
        """Load every persisted alias → (alias, canonical_id, entity_type)."""
        aliases: list[tuple[str, str, str]] = []
        stale_keys: list[str] = []
        for normalized in list(self._manifest["aliases"]):
            ext_id = f"{_PREFIX}alias-{normalized}"
            try:
                doc = self._db.get(ext_id)
                if not doc:
                    stale_keys.append(normalized)
                    continue
                meta = doc.get("metadata") or {}
                aliases.append((
                    meta.get("alias", normalized),
                    meta["canonical_id"],
                    meta.get("entity_type", "unknown"),
                ))
            except Exception:
                stale_keys.append(normalized)
        if stale_keys:
            for sk in stale_keys:
                self._manifest["aliases"].remove(sk)
            self._save_manifest()
        return aliases

    # ------------------------------------------------------------------
    # Vector search
    # ------------------------------------------------------------------

    def find_similar_detectors(self, description: str, k: int = 5) -> list[DetectorRule]:
        """Find detector rules similar to *description* via vector search."""
        results = self._db.search(
            query=f"Entity detector: {description}",
            k=k,
            filter={"node_type": NODE_TYPE_DETECTOR},
        )
        rules: list[DetectorRule] = []
        for r in results:
            meta = (r.metadata if hasattr(r, "metadata") else r.get("metadata")) or {}
            try:
                rules.append(DetectorRule(
                    id=meta["rule_id"],
                    pattern=meta["pattern"],
                    description=meta.get("description", ""),
                    created_at=datetime.fromisoformat(meta["created_at"])
                    if meta.get("created_at") else datetime.now(timezone.utc),
                ))
            except Exception:
                continue
        return rules

    def find_similar_extractors(self, description: str, k: int = 5) -> list[ExtractorRule]:
        """Find extractor rules similar to *description* via vector search."""
        results = self._db.search(
            query=f"Entity extractor: {description}",
            k=k,
            filter={"node_type": NODE_TYPE_EXTRACTOR},
        )
        rules: list[ExtractorRule] = []
        for r in results:
            meta = (r.metadata if hasattr(r, "metadata") else r.get("metadata")) or {}
            try:
                rules.append(ExtractorRule(
                    id=meta["rule_id"],
                    entity_type=meta.get("entity_type", ""),
                    detector_ids=json.loads(meta.get("detector_ids", "[]")),
                    pattern=meta["pattern"],
                    normalize_template=meta.get("normalize_template", "{match}"),
                    examples=json.loads(meta.get("examples", "[]")),
                    context_examples=json.loads(meta.get("context_examples", "[]")),
                    confidence=float(meta.get("confidence", 1.0)),
                    context_threshold=float(meta.get("context_threshold", 0.6)),
                ))
            except Exception:
                continue
        return rules

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_detector(self, rule_id: str) -> bool:
        """Delete a persisted DetectorRule. Returns True if deleted."""
        ext_id = _detector_ext_id(rule_id)
        try:
            self._db.delete(ext_id)
        except Exception:
            pass
        if rule_id in self._manifest["detectors"]:
            self._manifest["detectors"].remove(rule_id)
            self._save_manifest()
            return True
        return False

    def delete_extractor(self, rule_id: str) -> bool:
        """Delete a persisted ExtractorRule. Returns True if deleted."""
        ext_id = _extractor_ext_id(rule_id)
        try:
            self._db.delete(ext_id)
        except Exception:
            pass
        if rule_id in self._manifest["extractors"]:
            self._manifest["extractors"].remove(rule_id)
            self._save_manifest()
            return True
        return False

    def delete_alias(self, alias: str) -> bool:
        """Delete a persisted alias mapping. Returns True if deleted."""
        ext_id = _alias_ext_id(alias)
        normalized = re.sub(r"\s+", " ", alias.strip().lower())
        try:
            self._db.delete(ext_id)
        except Exception:
            pass
        if normalized in self._manifest["aliases"]:
            self._manifest["aliases"].remove(normalized)
            self._save_manifest()
            return True
        return False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying SiliconDB handle."""
        self._db.close()
