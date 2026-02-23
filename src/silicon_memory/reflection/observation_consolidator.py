"""Observation consolidation pipeline — deduplicate extracted observations.

Takes raw per-document observations (facts, relationships, arguments, events)
and consolidates them into canonical beliefs using existing infrastructure:

1. Normalize entity names via EntityResolver (replaces custom AliasTable)
2. Ingest observations into SiliconDB as triples
3. Find near-duplicates via SiliconDB find_similar_triples (replaces SequenceMatcher)
4. Cluster duplicates via union-find (replaces O(n^3) agglomerative)
5. Merge clusters → consolidated beliefs (simple or LLM)
6. Commit via SiliconDBBackend.commit_belief + Bayesian updates

Observations are preserved as evidence; consolidated beliefs link back via
evidence_for. Confidence grows with corroboration count.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING
from uuid import UUID, uuid4

from silicon_memory.core.types import (
    Belief,
    BeliefStatus,
    Source,
    SourceType,
    Triplet,
)
from silicon_memory.entities.date_normalizer import normalize_date

if TYPE_CHECKING:
    from silicon_memory.entities.resolver import EntityResolver
    from silicon_memory.storage.silicondb_backend import SiliconDBBackend

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    """A single extracted observation (fact, relationship, argument, event)."""
    id: UUID = field(default_factory=uuid4)
    kind: str = "fact"  # fact | relationship | argument | event
    subject: str = ""
    predicate: str = ""
    object: str = ""
    confidence: float = 0.7
    source_doc: str = ""  # doc_id or cache file
    raw: dict[str, Any] = field(default_factory=dict)

    def as_text(self) -> str:
        """Human-readable representation."""
        return f"{self.subject} {self.predicate} {self.object}".strip()


@dataclass
class Cluster:
    """A cluster of similar observations about the same topic."""
    id: UUID = field(default_factory=uuid4)
    subject: str = ""
    observations: list[Observation] = field(default_factory=list)
    kind: str = "fact"

    @property
    def size(self) -> int:
        return len(self.observations)

    @property
    def best(self) -> Observation:
        """Observation with highest confidence."""
        return max(self.observations, key=lambda o: o.confidence)


@dataclass
class ConsolidatedBelief:
    """A merged belief from a cluster of observations."""
    id: UUID = field(default_factory=uuid4)
    content: str = ""
    triplet: Triplet | None = None
    confidence: float = 0.5
    corroboration_count: int = 1
    observation_ids: list[UUID] = field(default_factory=list)
    kind: str = "fact"

    def to_belief(self) -> Belief:
        return Belief(
            id=self.id,
            content=self.content,
            triplet=self.triplet,
            confidence=self.confidence,
            source=Source(
                id="observation_consolidation",
                type=SourceType.REFLECTION,
                reliability=self.confidence,
                metadata={
                    "corroboration_count": self.corroboration_count,
                    "kind": self.kind,
                },
            ),
            status=BeliefStatus.PROVISIONAL,
            evidence_for=self.observation_ids,
            tags={"consolidated", self.kind},
        )


@dataclass
class ObservationConsolidationResult:
    """Result of running the observation consolidation pipeline."""
    observations_loaded: int = 0
    entities_normalized: int = 0
    observations_ingested: int = 0
    similar_pairs_found: int = 0
    clusters_found: int = 0
    singletons: int = 0
    beliefs_committed: int = 0
    corroboration_updates: int = 0
    consolidated_beliefs: list[ConsolidatedBelief] = field(default_factory=list)

    @property
    def merged_count(self) -> int:
        return len([c for c in self.consolidated_beliefs if c.corroboration_count > 1])


# ---------------------------------------------------------------------------
# Cache loader
# ---------------------------------------------------------------------------

def load_from_cache(cache_dir: str | Path, max_docs: int = 0) -> list[Observation]:
    """Load observations from extraction cache JSON files.

    Each cache file has structure:
      { "extraction": { "facts": [...], "relationships": [...],
                         "arguments": [...], "events": [...] },
        "metadata": { "doc_id": "..." } }

    Args:
        cache_dir: Path to the extraction cache directory.
        max_docs: Maximum number of cache files to load (0 = unlimited).
    """
    cache_path = Path(cache_dir)
    if not cache_path.is_dir():
        logger.warning("Cache directory does not exist: %s", cache_dir)
        return []

    observations: list[Observation] = []

    files = sorted(cache_path.glob("extraction_*.json"))
    if max_docs > 0:
        files = files[:max_docs]

    for fp in files:
        try:
            data = json.loads(fp.read_text())
        except Exception as e:
            logger.warning("Failed to read cache file %s: %s", fp, e)
            continue

        ext = data.get("extraction", {})
        doc_id = data.get("metadata", {}).get("doc_id", fp.stem)
        source_lookup = data.get("source_lookup", {})

        def _resolve_source_doc(source_label: str) -> str:
            label = str(source_label or "").strip()
            if not label:
                return doc_id
            if isinstance(source_lookup, dict):
                meta = source_lookup.get(label)
                if not isinstance(meta, dict):
                    lowered = label.lower()
                    for key, value in source_lookup.items():
                        key_text = str(key)
                        if lowered in key_text.lower() or key_text.lower() in lowered:
                            if isinstance(value, dict):
                                meta = value
                                break
                if isinstance(meta, dict):
                    candidate = (
                        str(meta.get("document_id") or "")
                        or str(meta.get("title") or "")
                        or str(meta.get("experience_id") or "")
                    )
                    if candidate.strip():
                        return candidate.strip()
            return label

        for fact in ext.get("facts", []):
            observations.append(Observation(
                kind="fact",
                subject=fact.get("subject", ""),
                predicate=fact.get("predicate", ""),
                object=fact.get("object", ""),
                confidence=fact.get("confidence", 0.7),
                source_doc=_resolve_source_doc(fact.get("source", "")),
                raw=fact,
            ))

        for rel in ext.get("relationships", []):
            observations.append(Observation(
                kind="relationship",
                subject=rel.get("person1", ""),
                predicate=rel.get("relationship", ""),
                object=rel.get("person2", ""),
                confidence=rel.get("confidence", 0.7),
                source_doc=_resolve_source_doc(rel.get("source", "")),
                raw=rel,
            ))

        for arg in ext.get("arguments", []):
            observations.append(Observation(
                kind="argument",
                subject=arg.get("actor", ""),
                predicate="argues",
                object=arg.get("claim", ""),
                confidence=arg.get("confidence", 0.7),
                source_doc=_resolve_source_doc(arg.get("source", "")),
                raw=arg,
            ))

        for ev in ext.get("events", []):
            actors = ", ".join(ev.get("actors", [])) or "unknown"
            source_doc = _resolve_source_doc(ev.get("source", ""))
            observations.append(Observation(
                kind="event",
                subject=actors,
                predicate="event",
                object=ev.get("event", ""),
                confidence=ev.get("confidence", 0.7),
                source_doc=source_doc,
                raw=ev,
            ))
            raw_date = str(ev.get("date", "")).strip()
            date_norm = normalize_date(raw_date) if raw_date else None
            if date_norm:
                observations.append(Observation(
                    kind="event",
                    subject=actors,
                    predicate="occurred_on",
                    object=date_norm.canonical,
                    confidence=ev.get("confidence", 0.7),
                    source_doc=source_doc,
                    raw={
                        **ev,
                        "_date_normalized": {
                            "raw": date_norm.raw,
                            "canonical": date_norm.canonical,
                            "precision": date_norm.precision,
                            "ambiguous": date_norm.ambiguous,
                            "confidence": date_norm.confidence,
                            "relative": date_norm.relative,
                            "reference_date": date_norm.reference_date,
                        },
                    },
                ))

    logger.info("Loaded %d observations from %s", len(observations), cache_dir)
    return observations


# ---------------------------------------------------------------------------
# Union-Find (O(n·α(n)) instead of O(n³) agglomerative)
# ---------------------------------------------------------------------------

class _UnionFind:
    """Weighted union-find with path compression."""

    def __init__(self) -> None:
        self._parent: dict[UUID, UUID] = {}
        self._rank: dict[UUID, int] = {}

    def make_set(self, x: UUID) -> None:
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0

    def find(self, x: UUID) -> UUID:
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        # Path compression
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: UUID, b: UUID) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1


# ---------------------------------------------------------------------------
# Core consolidator — uses EntityResolver + SiliconDB
# ---------------------------------------------------------------------------

class ObservationConsolidator:
    """Consolidates observations into deduplicated beliefs.

    Uses EntityResolver for entity normalization and SiliconDB's
    embedding-based similarity search for duplicate detection.

    Usage:
        consolidator = ObservationConsolidator(backend, resolver, llm=scheduler)
        observations = load_from_cache("extraction_cache/")
        result = await consolidator.consolidate(observations)
    """

    def __init__(
        self,
        backend: "SiliconDBBackend",
        resolver: "EntityResolver",
        llm: Any = None,
        similarity_threshold: float = 0.65,
        min_cluster_for_llm: int = 2,
    ) -> None:
        self._backend = backend
        self._resolver = resolver
        self._llm = llm
        self._threshold = similarity_threshold
        self._min_cluster_for_llm = min_cluster_for_llm

    # ------------------------------------------------------------------
    # Step 1: Normalize entities via EntityResolver
    # ------------------------------------------------------------------

    async def _normalize_entities(
        self, observations: list[Observation],
    ) -> int:
        """Resolve entity names via EntityResolver.

        Registers multi-word names as entities, then normalizes each
        observation's subject/object to canonical form. Reuses the
        auto-last-name-aliasing from EntityResolver.register_extracted_entities.

        Returns number of name changes made.
        """
        # Collect unique multi-word names for bulk registration
        entities: list[tuple[str, str, str]] = []
        seen: set[str] = set()
        for obs in observations:
            for name in (obs.subject, obs.object):
                if not name or name in seen:
                    continue
                seen.add(name)
                if " " in name:
                    # Use "person" for names that look like person names
                    # (multi-word, each part capitalized) to enable auto-last-name
                    # aliasing in EntityResolver.  Fall back to "entity" otherwise.
                    parts = name.split()
                    looks_like_person = all(p[0].isupper() for p in parts if p)
                    etype = "person" if looks_like_person else "entity"
                    entities.append((name, etype, name))

        if entities:
            try:
                registered = await self._resolver.register_extracted_entities(entities)
                logger.info("Registered %d entity aliases for normalization", registered)
            except Exception as e:
                logger.warning("Entity registration partially failed (cache still usable): %s", e)

        # Two-pass normalization:
        # Pass 1: resolve full/unambiguous names and build context
        # Pass 2: resolve bare/ambiguous names using document context
        changes = 0

        # Group observations by source_doc for document-level context
        from collections import defaultdict
        by_doc: dict[str, list[Observation]] = defaultdict(list)
        for obs in observations:
            by_doc[obs.source_doc].append(obs)

        for doc_id, doc_obs in by_doc.items():
            # Pass 1: resolve unambiguous names, collect as context
            doc_context: set[str] = set()
            for obs in doc_obs:
                for attr in ("subject", "object"):
                    original = getattr(obs, attr)
                    if not original:
                        continue
                    canonical = self._resolver.cache.lookup(original)
                    if canonical and canonical != original:
                        setattr(obs, attr, canonical)
                        changes += 1
                        doc_context.add(canonical)
                    elif canonical:
                        # Already canonical — still useful as context
                        doc_context.add(canonical)

            # Pass 2: retry unresolved names with document context
            if doc_context:
                for obs in doc_obs:
                    for attr in ("subject", "object"):
                        original = getattr(obs, attr)
                        if not original:
                            continue
                        # Skip if already resolved in pass 1
                        if self._resolver.cache.lookup(original):
                            continue
                        canonical = self._resolver.cache.lookup_with_context(
                            original, doc_context,
                        )
                        if canonical and canonical != original:
                            setattr(obs, attr, canonical)
                            changes += 1

        return changes

    # ------------------------------------------------------------------
    # Step 2: Ingest observations — observe existing or insert new
    # ------------------------------------------------------------------

    def _find_existing_triple(
        self, subject: str, predicate: str, object_value: str,
    ) -> str | None:
        """Return the external_id of an existing triple matching (s, p, o), or None."""
        try:
            existing = self._backend._db.query_triples(
                subject=subject, predicate=predicate, k=50,
            )
            for t in existing:
                if getattr(t, "object_value", None) == object_value:
                    return t.external_id
        except Exception:
            pass
        return None

    async def _ingest_observations(
        self, observations: list[Observation],
    ) -> tuple[dict[UUID, str], int]:
        """Ingest observations using Monte Carlo observe-or-insert.

        For each observation:
        - If the (subject, predicate, object) triple already exists,
          record_observation() on it (Bayesian update) instead of
          creating a duplicate.
        - If new, insert_triple() to create it.

        Returns (obs_id→external_id mapping, observation_count).
        """
        obs_to_ext: dict[UUID, str] = {}
        prefix = self._backend._get_user_prefix()
        mc_observations = 0

        for obs in observations:
            # Check for existing triple with same (s, p, o)
            existing_ext = self._find_existing_triple(
                obs.subject, obs.predicate, obs.object,
            )

            if existing_ext:
                # Triple already known — record observation (MC update)
                try:
                    self._backend._db.record_observation(
                        external_id=existing_ext,
                        confirmed=True,
                        source=f"cache:{obs.source_doc}",
                    )
                    obs_to_ext[obs.id] = existing_ext
                    mc_observations += 1
                except Exception as e:
                    logger.debug("MC observe failed for %s: %s", existing_ext, e)
            else:
                # New triple — insert it
                ext_id = f"{prefix}obs-{obs.id}"
                metadata: dict[str, Any] = {
                    "kind": obs.kind,
                    "observation_id": str(obs.id),
                    "source_doc": obs.source_doc,
                }
                source_meta: dict[str, Any] = {}
                if obs.source_doc:
                    source_meta["source_document"] = {"document_id": obs.source_doc}
                raw_date_norm = None
                if isinstance(obs.raw, dict):
                    raw_date_blob = obs.raw.get("_date_normalized")
                    if isinstance(raw_date_blob, dict):
                        raw_date_norm = {
                            "raw": str(raw_date_blob.get("raw") or ""),
                            "canonical": str(raw_date_blob.get("canonical") or ""),
                            "precision": str(raw_date_blob.get("precision") or ""),
                            "ambiguous": bool(raw_date_blob.get("ambiguous", False)),
                            "confidence": float(raw_date_blob.get("confidence", 0.0) or 0.0),
                            "relative": bool(raw_date_blob.get("relative", False)),
                            "reference_date": raw_date_blob.get("reference_date"),
                        }
                if raw_date_norm is None:
                    date_guess = normalize_date(obs.object)
                    if date_guess:
                        raw_date_norm = {
                            "raw": date_guess.raw,
                            "canonical": date_guess.canonical,
                            "precision": date_guess.precision,
                            "ambiguous": date_guess.ambiguous,
                            "confidence": date_guess.confidence,
                            "relative": date_guess.relative,
                            "reference_date": date_guess.reference_date,
                        }
                if raw_date_norm and raw_date_norm.get("canonical"):
                    source_meta["object_date"] = raw_date_norm
                    source_meta["date_normalized"] = raw_date_norm
                if source_meta:
                    metadata["source_metadata"] = source_meta
                try:
                    self._backend._db.insert_triple(
                        external_id=ext_id,
                        subject=obs.subject,
                        predicate=obs.predicate,
                        object_value=obs.object,
                        probability=obs.confidence,
                        metadata=metadata,
                    )
                    obs_to_ext[obs.id] = ext_id
                except Exception as e:
                    logger.debug("Failed to ingest observation %s: %s", obs.id, e)

        return obs_to_ext, mc_observations

    # ------------------------------------------------------------------
    # Step 3: Find similar pairs by subject grouping + text overlap
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_text(value: Any) -> str:
        """Coerce mixed LLM values into a compact string."""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list | tuple | set):
            return ", ".join(str(item).strip() for item in value if str(item).strip())
        if isinstance(value, dict):
            return ", ".join(
                f"{k}:{v}" for k, v in value.items() if str(v).strip()
            )
        return str(value or "").strip()

    @staticmethod
    def _text_overlap(a: str, b: str) -> float:
        """Token-level Jaccard similarity between two strings."""
        ta = set(a.lower().split())
        tb = set(b.lower().split())
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    async def _find_similar_pairs(
        self,
        observations: list[Observation],
        obs_to_ext: dict[UUID, str],
    ) -> list[tuple[UUID, UUID, float]]:
        """Find near-duplicate observation pairs.

        Groups observations by (normalized_subject, kind), then compares
        predicate+object text within each group using token overlap.
        This avoids reliance on embedding-based similarity search
        which requires async indexing to complete first.
        """
        # Group by (subject_lower, kind)
        from collections import defaultdict
        groups: dict[tuple[str, str], list[Observation]] = defaultdict(list)
        for obs in observations:
            key = (obs.subject.lower(), obs.kind)
            groups[key].append(obs)

        pairs: list[tuple[UUID, UUID, float]] = []
        seen_pairs: set[tuple[UUID, UUID]] = set()

        for (_subj, _kind), group in groups.items():
            if len(group) < 2:
                continue
            # Compare all pairs within the group
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a, b = group[i], group[j]
                    # Compare predicate+object text
                    text_a = f"{a.predicate} {a.object}"
                    text_b = f"{b.predicate} {b.object}"
                    score = self._text_overlap(text_a, text_b)
                    if score < self._threshold:
                        continue
                    pair_key = (min(a.id, b.id), max(a.id, b.id))
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)
                    pairs.append((a.id, b.id, score))

        return pairs

    # ------------------------------------------------------------------
    # Step 4: Build clusters (union-find)
    # ------------------------------------------------------------------

    def _build_clusters(
        self,
        observations: list[Observation],
        pairs: list[tuple[UUID, UUID, float]],
    ) -> list[Cluster]:
        """Cluster observations using union-find.

        Constrains clusters to same kind and same normalized subject.
        O(n·α(n)) instead of O(n³) agglomerative clustering.
        """
        obs_by_id = {o.id: o for o in observations}
        uf = _UnionFind()

        for obs in observations:
            uf.make_set(obs.id)

        for id_a, id_b, _score in pairs:
            obs_a = obs_by_id.get(id_a)
            obs_b = obs_by_id.get(id_b)
            if not obs_a or not obs_b:
                continue
            # Same kind + same normalized subject
            if obs_a.kind != obs_b.kind:
                continue
            if obs_a.subject.lower() != obs_b.subject.lower():
                continue
            uf.union(id_a, id_b)

        # Group by root
        root_to_members: dict[UUID, list[Observation]] = {}
        for obs in observations:
            root = uf.find(obs.id)
            if root not in root_to_members:
                root_to_members[root] = []
            root_to_members[root].append(obs)

        clusters = []
        for members in root_to_members.values():
            clusters.append(Cluster(
                subject=members[0].subject,
                observations=members,
                kind=members[0].kind,
            ))

        return clusters

    # ------------------------------------------------------------------
    # Step 5a: Simple merge (no LLM)
    # ------------------------------------------------------------------

    @staticmethod
    def _simple_merge(cluster: Cluster) -> ConsolidatedBelief:
        """Merge a cluster without LLM — pick the most confident observation."""
        best = cluster.best
        subject = ObservationConsolidator._coerce_text(best.subject)
        predicate = ObservationConsolidator._coerce_text(best.predicate)
        object_value = ObservationConsolidator._coerce_text(best.object)
        # Confidence boosts: more corroboration = higher confidence
        base_conf = best.confidence
        boost = min(0.15, 0.05 * (cluster.size - 1))
        final_conf = min(0.95, base_conf + boost)

        return ConsolidatedBelief(
            content=f"{subject} {predicate} {object_value}".strip(),
            triplet=Triplet(
                subject=subject,
                predicate=predicate,
                object=object_value,
            ),
            confidence=final_conf,
            corroboration_count=cluster.size,
            observation_ids=[o.id for o in cluster.observations],
            kind=cluster.kind,
        )

    # ------------------------------------------------------------------
    # Step 5b: LLM merge
    # ------------------------------------------------------------------

    async def _llm_merge(self, cluster: Cluster) -> ConsolidatedBelief:
        """Use LLM to merge a cluster into one canonical belief."""
        obs_texts = []
        for i, obs in enumerate(cluster.observations[:10]):
            obs_texts.append(f"{i+1}. {obs.as_text()} [conf={obs.confidence:.2f}]")

        prompt = (
            "You are merging duplicate or near-duplicate knowledge observations "
            "into a single canonical fact.\n\n"
            f"Subject entity: {cluster.subject}\n"
            f"Observation type: {cluster.kind}\n\n"
            "Observations to merge:\n" + "\n".join(obs_texts) + "\n\n"
            "Produce ONE merged fact as JSON with fields:\n"
            '{"subject": "...", "predicate": "...", "object": "...", "confidence": 0.X}\n\n'
            "Rules:\n"
            "- Use the most specific and accurate version of each field\n"
            "- Use the full canonical name for the subject\n"
            "- Merge overlapping info (e.g. '71st Street' and '9 East 71st Street' → '9 East 71st Street')\n"
            "- Confidence should reflect the combined strength (more corroboration = higher)\n"
            "- Return ONLY the JSON object, no explanation\n"
        )

        try:
            raw = await self._llm.generate(prompt, max_tokens=256, temperature=0.2)
            raw = raw.strip()
            if raw.startswith("```"):
                lines = raw.split("\n")
                raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

            match = re.search(r"\{[\s\S]*\}", raw)
            if match:
                parsed = json.loads(match.group())
                try:
                    parsed_conf = float(parsed.get("confidence", 0.7))
                except Exception:
                    parsed_conf = 0.7
                base_conf = min(0.95, max(0.0, parsed_conf))
                boost = min(0.15, 0.05 * (cluster.size - 1))
                final_conf = min(0.95, base_conf + boost)
                fallback = cluster.best
                subject = self._coerce_text(parsed.get("subject")) or self._coerce_text(fallback.subject)
                predicate = self._coerce_text(parsed.get("predicate")) or self._coerce_text(fallback.predicate)
                object_value = self._coerce_text(parsed.get("object")) or self._coerce_text(fallback.object)

                return ConsolidatedBelief(
                    content=f"{subject} {predicate} {object_value}".strip(),
                    triplet=Triplet(
                        subject=subject,
                        predicate=predicate,
                        object=object_value,
                    ),
                    confidence=final_conf,
                    corroboration_count=cluster.size,
                    observation_ids=[o.id for o in cluster.observations],
                    kind=cluster.kind,
                )
        except Exception as e:
            logger.warning("LLM merge failed for cluster %s: %s", cluster.subject, e)

        # Fallback to simple merge
        return self._simple_merge(cluster)

    # ------------------------------------------------------------------
    # Step 6: Commit consolidated beliefs + Bayesian corroboration
    # ------------------------------------------------------------------

    async def _merge_one(self, cluster: Cluster) -> tuple[Cluster, ConsolidatedBelief]:
        """Merge a single cluster (LLM or simple). Safe for concurrent use."""
        if self._llm and cluster.size >= self._min_cluster_for_llm:
            return cluster, await self._llm_merge(cluster)
        return cluster, self._simple_merge(cluster)

    async def _commit_consolidated(
        self,
        clusters: list[Cluster],
        obs_to_ext: dict[UUID, str],
    ) -> tuple[list[ConsolidatedBelief], int]:
        """Merge clusters and commit as beliefs via observe-or-insert.

        For each cluster:
        - Merge (simple or LLM) into a single canonical triple
        - Check if an identical triple already exists
          - Yes: record_observation on it (MC corroboration)
          - No: commit as new belief
        - Record N-1 additional corroboration observations
        - Link cluster members via add_cooccurrences

        All LLM calls go through the shared scheduler which controls
        concurrency — no local semaphore needed.

        Returns (consolidated_beliefs, corroboration_update_count).
        """
        import asyncio

        consolidated: list[ConsolidatedBelief] = []
        corroboration_count = 0

        # Phase 1: Merge all clusters concurrently via asyncio.gather.
        # The shared LLM scheduler controls concurrency — no local semaphore.
        tasks = [self._merge_one(c) for c in clusters]
        merged_pairs = await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("Merged %d clusters via shared scheduler", len(clusters))

        # Phase 2: Commit merged results sequentially (DB writes)
        for item in merged_pairs:
            if isinstance(item, Exception):
                logger.warning("Cluster merge failed: %s", item)
                continue

            cluster, merged = item
            triplet = merged.triplet
            if not triplet:
                continue

            try:
                # Check if this merged triple already exists
                existing_ext = self._find_existing_triple(
                    triplet.subject, triplet.predicate, triplet.object,
                )

                if existing_ext:
                    # Already known — record N observations (whole cluster)
                    for obs in cluster.observations:
                        try:
                            self._backend._db.record_observation(
                                external_id=existing_ext,
                                confirmed=True,
                                source=f"consolidation:{obs.source_doc}",
                            )
                            corroboration_count += 1
                        except Exception:
                            pass
                    # Try to extract the belief UUID from the external ID
                    if "belief-" in existing_ext:
                        try:
                            uuid_str = existing_ext.split("belief-", 1)[1]
                            merged.id = UUID(uuid_str)
                        except (ValueError, IndexError):
                            pass  # Keep generated ID
                    consolidated.append(merged)
                else:
                    # New canonical belief — commit it
                    belief = merged.to_belief()
                    await self._backend.commit_belief(belief)
                    consolidated.append(merged)

                    # Record N-1 corroboration observations on the new belief
                    ext_id = self._backend._build_external_id("belief", belief.id)
                    for obs in cluster.observations[1:]:
                        try:
                            self._backend._db.record_observation(
                                external_id=ext_id,
                                confirmed=True,
                                source=f"corroboration:{obs.source_doc}",
                            )
                            corroboration_count += 1
                        except Exception:
                            pass

                # Link cluster members via co-occurrences
                member_ext_ids = [
                    obs_to_ext[o.id]
                    for o in cluster.observations
                    if o.id in obs_to_ext
                ]

                # Merge cluster member triples into the canonical target when
                # the backend supports it. This preserves one canonical edge
                # while folding corroborating duplicates.
                target_ext = existing_ext if existing_ext else (
                    self._backend._build_external_id("belief", merged.id)
                )
                merge_sources = [
                    ext for ext in member_ext_ids
                    if ext and ext != target_ext
                ]
                if merge_sources:
                    try:
                        unique_sources = list(dict.fromkeys(merge_sources))[:100]
                        self._backend._db.merge_triples(
                            source_ids=unique_sources,
                            target_id=target_ext,
                        )
                    except Exception:
                        # Merge support may not exist in older backends.
                        pass

                if len(member_ext_ids) >= 2:
                    try:
                        self._backend._db.add_cooccurrences(
                            member_ext_ids[:20],
                            session_id=f"consolidation-{cluster.id}",
                        )
                    except Exception:
                        pass

            except Exception as e:
                logger.warning("Failed to commit consolidated belief: %s", e)

        return consolidated, corroboration_count

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def consolidate(
        self, observations: list[Observation],
    ) -> ObservationConsolidationResult:
        """Run the full consolidation pipeline."""
        result = ObservationConsolidationResult(observations_loaded=len(observations))

        if not observations:
            return result

        # Step 1: Normalize entities via EntityResolver
        result.entities_normalized = await self._normalize_entities(observations)
        logger.info(
            "Entity normalization: %d name changes across %d observations",
            result.entities_normalized, len(observations),
        )

        # Step 2: Observe-or-insert into SiliconDB (Monte Carlo)
        obs_to_ext, mc_observations = await self._ingest_observations(observations)
        result.observations_ingested = len(obs_to_ext)
        result.corroboration_updates = mc_observations
        logger.info(
            "Ingested %d observations: %d new triples, %d MC observations on existing",
            len(obs_to_ext),
            len(obs_to_ext) - mc_observations,
            mc_observations,
        )

        # Step 3: Find similar pairs via embedding search
        pairs = await self._find_similar_pairs(observations, obs_to_ext)
        result.similar_pairs_found = len(pairs)
        logger.info("Found %d similar observation pairs", len(pairs))

        # Step 4: Cluster via union-find
        clusters = self._build_clusters(observations, pairs)
        result.clusters_found = len(clusters)
        multi_clusters = [c for c in clusters if c.size >= 2]
        result.singletons = sum(1 for c in clusters if c.size == 1)
        logger.info(
            "Formed %d clusters (%d multi-obs, %d singletons)",
            len(clusters), len(multi_clusters), result.singletons,
        )

        # Step 5-6: Merge and commit (only multi-observation clusters)
        consolidated, commit_corroborations = await self._commit_consolidated(
            multi_clusters, obs_to_ext,
        )
        result.beliefs_committed = len(consolidated)
        result.corroboration_updates += commit_corroborations
        result.consolidated_beliefs = consolidated
        logger.info(
            "Committed %d consolidated beliefs (%d MC observations total: %d ingest + %d commit)",
            len(consolidated), result.corroboration_updates,
            mc_observations, commit_corroborations,
        )

        return result

    async def consolidate_from_cache(
        self, cache_dir: str | Path, max_docs: int = 0,
    ) -> ObservationConsolidationResult:
        """Convenience: load from extraction cache and consolidate.

        Args:
            cache_dir: Path to the extraction cache directory.
            max_docs: Maximum number of cache files to process (0 = unlimited).
        """
        observations = load_from_cache(cache_dir, max_docs=max_docs)
        return await self.consolidate(observations)
