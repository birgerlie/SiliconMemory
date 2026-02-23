"""Main reflection engine orchestrating the cognitive pipeline.

Implements a multi-phase cognitive process:
1. Extract — LLM extracts structured knowledge from experiences
2. Generate — Convert patterns into belief candidates
3. Commit — Store validated beliefs
4. Dream Forward — Transitive inference discovers hidden connections
5. Dream Backward — New evidence updates existing beliefs
6. Hypothesize — Clustering + LLM generates hypotheses
7. Consolidate — Monte Carlo, propagation, generalization, decay

Operates on the knowledge graph, NOT on raw document text. Extraction
(raw text → structured knowledge) is handled separately by the
ExtractionWorker. This engine reasons over already-extracted beliefs:

1. Transitive inference — A→B + B→C = A→C
2. Contradiction detection — conflicting beliefs
3. Hypothesis generation — graph clustering + LLM
4. Monte Carlo consolidation — probability propagation, decay
5. Decision assumption drift review
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any, TYPE_CHECKING
from uuid import UUID, uuid4

from silicon_memory.core.types import Belief, BeliefStatus, Source, SourceType, Triplet
from silicon_memory.core.decision import DecisionStatus
from silicon_memory.core.utils import utc_now
import logging

from silicon_memory.reflection.types import (
    BeliefCandidate,
    ConsolidationResult,
    Pattern,
    ReflectionConfig,
    ReflectionResult,
)

logger = logging.getLogger(__name__)
from silicon_memory.reflection.processor import ExperienceProcessor
from silicon_memory.reflection.llm_extractor import LLMPatternExtractor
from silicon_memory.reflection.generator import BeliefGenerator
from silicon_memory.reflection.inference import TransitiveInferenceEngine
from silicon_memory.reflection.hypothesis import HypothesisGenerator
from silicon_memory.reflection.consolidation import MemoryConsolidator
from silicon_memory.reflection.lifecycle import BeliefLifecycleManager
from silicon_memory.reflection.procedure_detector import ProcedureDetector
from silicon_memory.reflection.question_generator import QuestionGenerator
from silicon_memory.reflection.observation_consolidator import ObservationConsolidator
from silicon_memory.reflection.predicate_consolidator import PredicateConsolidator

if TYPE_CHECKING:
    from silicon_memory.entities.resolver import EntityResolver
    from silicon_memory.memory.silicondb_router import SiliconMemory


def _parse_json(text: str) -> dict | None:
    """Extract JSON from LLM response text."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for marker in ("```json", "```"):
        if marker in text:
            start = text.index(marker) + len(marker)
            end = text.index("```", start) if "```" in text[start:] else len(text)
            try:
                return json.loads(text[start:end].strip())
            except json.JSONDecodeError:
                pass
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start:brace_end + 1])
        except json.JSONDecodeError:
            pass
    return None


def _as_bool(value: Any, default: bool = False) -> bool:
    """Parse bool-like values from metadata payloads."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in {"true", "1", "yes", "y", "on"}:
            return True
        if norm in {"false", "0", "no", "n", "off", ""}:
            return False
    return default


class ReflectionEngine:
    """Reflection engine — reasons over the knowledge graph.

    Extraction of raw text into structured knowledge is now handled by
    the ExtractionWorker (Phase 2). This engine is Phase 3: it operates
    on already-extracted beliefs and graph structure.

    The cognitive pipeline:
    1. Fetch unprocessed experiences and extract patterns (legacy path)
       OR query recently-added beliefs from the graph (new path)
    2. Generate belief candidates from patterns
    3. Validate against existing knowledge
    4. Commit valid beliefs
    5. Dream forward — transitive inference
    6. Dream backward — new evidence updates existing beliefs
    7. Monte Carlo consolidation
    8. Mark experiences as processed

    Extended operations (run periodically):
    - Hypothesis generation from graph clustering
    - Memory consolidation (generalization, decay, cross-modal linking)

    Example:
        >>> engine = ReflectionEngine(memory, llm=scheduler)
        >>> result = await engine.reflect(auto_commit=True)
        >>> print(result.summary())
        >>>
        >>> # Deep dreaming: inference + hypotheses + consolidation
        >>> await engine.dream()
    """

    def __init__(
        self,
        memory: "SiliconMemory",
        llm: Any,
        config: ReflectionConfig | None = None,
        extraction_cache_dir: str | None = None,
        resolver: "EntityResolver | None" = None,
    ) -> None:
        self._memory = memory
        self._llm = llm
        self._config = config or ReflectionConfig()
        self._extraction_cache_dir = extraction_cache_dir
        self._last_empty_extracted_scan: float | None = None

        # Legacy extraction pipeline (still available for direct use)
        self._processor = ExperienceProcessor(memory, self._config)
        self._extractor = LLMPatternExtractor(
            memory,
            llm,
            self._config,
            cache_dir=extraction_cache_dir,
            resolver=resolver,
        )
        self._generator = BeliefGenerator(memory, self._config, llm=llm)

        # Cognitive extensions — operate on graph, not raw text
        self._inferencer = TransitiveInferenceEngine(memory, self._config)
        self._hypothesizer = HypothesisGenerator(memory, llm, self._config)
        self._consolidator = MemoryConsolidator(memory, self._config)

        # LLM-based belief lifecycle evaluation
        self._lifecycle = BeliefLifecycleManager(memory, llm)

        # Observation consolidation (deduplication of extracted observations)
        self._observation_consolidator: ObservationConsolidator | None = None
        if resolver is not None:
            self._observation_consolidator = ObservationConsolidator(
                backend=memory._backend,
                resolver=resolver,
                llm=llm,
            )

    @property
    def config(self) -> ReflectionConfig:
        """Get the current configuration."""
        return self._config

    async def reflect(
        self,
        max_experiences: int | None = None,
        auto_commit: bool | None = None,
    ) -> ReflectionResult:
        """Run a reflection cycle — graph-first, with legacy fallback.

        The primary path queries recently-extracted beliefs from the graph
        and runs inference/consolidation on them (no raw text reading).

        Legacy fallback: if no extracted beliefs exist but unprocessed
        experiences do, falls back to LLM extraction (for backward compat).

        Args:
            max_experiences: Override max experiences to process
            auto_commit: Override auto-commit setting

        Returns:
            ReflectionResult with new beliefs, patterns, and stats
        """
        import time as _time

        run_id = str(uuid4())
        reflect_start = _time.monotonic()
        timings: dict[str, float] = {}
        result = ReflectionResult()
        should_commit = auto_commit if auto_commit is not None else self._config.auto_commit_beliefs
        reflection_status = "success"
        reflection_source = "none"

        try:
            # ---- Primary path: graph-based reflection ----
            # Query unreflected extraction journal items first.
            t0 = _time.monotonic()
            recently_extracted, extraction_item_external_ids, extracted_external_ids = (
                await self._query_recent_extraction_items(limit=max_experiences)
            )
            if not recently_extracted:
                # Remote DB may be briefly eventually-consistent after ingest.
                for _ in range(3):
                    await asyncio.sleep(0.2)
                    recently_extracted, extraction_item_external_ids, extracted_external_ids = (
                        await self._query_recent_extraction_items(limit=max_experiences)
                    )
                    if recently_extracted:
                        break
            timings["query_extraction_journal"] = _time.monotonic() - t0

            # Fallback to belief-doc scan if journal is empty.
            if not recently_extracted:
                t0 = _time.monotonic()
                recently_extracted, extracted_external_ids = await self._query_recent_extracted_beliefs(
                    limit=max_experiences,
                )
                timings["query_recent_extracted"] = _time.monotonic() - t0
                extraction_item_external_ids = []
                if recently_extracted:
                    reflection_source = "belief_docs"
            else:
                reflection_source = "extraction_journal"

            if recently_extracted:
                logger.info(
                    "Reflecting on %d recently extracted beliefs (graph-only)",
                    len(recently_extracted),
                )

                # Observation consolidation: deduplicate before inference
                if (
                    self._config.enable_observation_consolidation
                    and self._observation_consolidator
                    and self._extraction_cache_dir
                ):
                    t0 = _time.monotonic()
                    try:
                        consol_result = await self._observation_consolidator.consolidate_from_cache(
                            self._extraction_cache_dir,
                            max_docs=max_experiences or 0,
                        )
                        logger.info(
                            "Observation consolidation: %d obs → %d clusters → %d beliefs committed",
                            consol_result.observations_loaded,
                            consol_result.clusters_found,
                            consol_result.beliefs_committed,
                        )
                    except Exception as e:
                        logger.warning("Observation consolidation failed: %s", e)
                    timings["observation_consolidation"] = _time.monotonic() - t0
                else:
                    timings["observation_consolidation"] = 0.0

                # Run transitive inference on extracted beliefs
                committed_beliefs = recently_extracted
                committed_ids = [b.id for b in committed_beliefs]
                inference_beliefs = recently_extracted

                # Yield event loop before heavy inference
                await asyncio.sleep(0)

                # Dream forward
                t0 = _time.monotonic()
                try:
                    inference_result = await self._inferencer.infer_forward(
                        inference_beliefs, max_hops=2, min_confidence=0.3
                    )
                    for inferred in inference_result.inferred_beliefs:
                        try:
                            await self._memory.commit_belief(inferred)
                            committed_ids.append(inferred.id)
                            result.updated_beliefs.append(
                                (inferred.id, inferred.confidence)
                            )
                        except Exception as e:
                            logger.debug("Failed to commit inferred belief: %s", e)

                    logger.info(
                        "Dream forward: %d chains → %d inferred beliefs",
                        inference_result.chains_found,
                        len(inference_result.inferred_beliefs),
                    )
                except Exception as e:
                    logger.warning("Forward inference failed: %s", e)
                timings["infer_forward"] = _time.monotonic() - t0

                # Yield between stages
                await asyncio.sleep(0)

                # Dream backward
                t0 = _time.monotonic()
                try:
                    backward_result = await self._inferencer.infer_backward(
                        inference_beliefs, min_confidence=0.3
                    )
                    logger.info(
                        "Dream backward: %d beliefs updated",
                        backward_result.backward_updates,
                    )
                except Exception as e:
                    logger.warning("Backward dreaming failed: %s", e)
                timings["infer_backward"] = _time.monotonic() - t0

                # Yield after backward inference
                await asyncio.sleep(0)
                logger.info("Post-inference: %d committed IDs", len(committed_ids))

                # Lightweight consolidation (lifecycle only — heavy MC ops deferred)
                if committed_ids:
                    t0 = _time.monotonic()
                    result.consolidation = await self._consolidate_beliefs(committed_ids)
                    timings["consolidate_beliefs"] = _time.monotonic() - t0

                # Review decisions
                t0 = _time.monotonic()
                await self._review_active_decisions()
                timings["review_decisions"] = _time.monotonic() - t0

                # Mark sources as processed by reflection so future cycles only process deltas.
                t0 = _time.monotonic()
                try:
                    if extraction_item_external_ids:
                        await self._memory._backend.mark_extraction_items_processed(
                            extraction_item_external_ids,
                            run_id=run_id,
                        )
                    if extracted_external_ids:
                        await self._memory._backend.mark_beliefs_reflection_processed(
                            extracted_external_ids,
                        )
                except Exception as e:
                    logger.warning("Failed to mark reflected beliefs: %s", e)
                timings["mark_reflection_processed"] = _time.monotonic() - t0

                result.experiences_processed = len(recently_extracted)
                timings["total"] = _time.monotonic() - reflect_start
                result.timings = {k: round(v, 4) for k, v in timings.items()}
                return result

            # ---- Legacy fallback: LLM extraction from raw text ----
            reflection_source = "legacy_experiences"
            limit = max_experiences or self._config.max_experiences_per_batch or 10000
            t0 = _time.monotonic()
            experiences = await self._processor.fetch_unprocessed(limit)
            timings["fetch_unprocessed"] = _time.monotonic() - t0

            if not experiences:
                timings["total"] = _time.monotonic() - reflect_start
                result.timings = {k: round(v, 4) for k, v in timings.items()}
                return result

            result.experiences_processed = len(experiences)

            # Flat char-based chunking — bypasses the overlapping grouping
            # strategies (session/entity/time) which created 3-5x more groups
            # than documents and wasted LLM calls on duplicate content.
            t0 = _time.monotonic()
            patterns = await self._extractor.extract_patterns_flat(experiences)
            timings["extract_patterns_flat"] = _time.monotonic() - t0
            result.patterns_found = patterns

            if not patterns:
                t0 = _time.monotonic()
                await self._processor.mark_processed([e.id for e in experiences])
                timings["mark_processed"] = _time.monotonic() - t0
                timings["total"] = _time.monotonic() - reflect_start
                result.timings = {k: round(v, 4) for k, v in timings.items()}
                return result

            t0 = _time.monotonic()
            candidates = await self._generator.generate_beliefs(patterns)
            timings["generate_beliefs"] = _time.monotonic() - t0
            result.new_beliefs = candidates

            for candidate in candidates:
                for contra_id in candidate.contradicts:
                    result.contradictions.append((candidate.id, contra_id))

            t0 = _time.monotonic()
            await self._review_active_decisions()
            timings["review_decisions"] = _time.monotonic() - t0

            committed_beliefs: list[Belief] = []
            committed_ids: list[UUID] = []
            if should_commit:
                t0 = _time.monotonic()
                for candidate in candidates:
                    if not candidate.is_contested:
                        belief = await self._generator.commit_belief(candidate)
                        if belief:
                            committed_beliefs.append(belief)
                            committed_ids.append(belief.id)
                            result.updated_beliefs.append((belief.id, belief.confidence))
                timings["commit_candidates"] = _time.monotonic() - t0

            if committed_beliefs:
                inference_input = committed_beliefs
                t0 = _time.monotonic()
                try:
                    inference_result = await self._inferencer.infer_forward(
                        inference_input, max_hops=2, min_confidence=0.3
                    )
                    for inferred in inference_result.inferred_beliefs:
                        try:
                            await self._memory.commit_belief(inferred)
                            committed_ids.append(inferred.id)
                            result.updated_beliefs.append(
                                (inferred.id, inferred.confidence)
                            )
                        except Exception as e:
                            logger.debug("Failed to commit inferred belief: %s", e)

                    logger.info(
                        "Dream forward: %d chains → %d inferred beliefs",
                        inference_result.chains_found,
                        len(inference_result.inferred_beliefs),
                    )
                except Exception as e:
                    logger.warning("Forward inference failed: %s", e)
                timings["infer_forward"] = _time.monotonic() - t0

            if committed_beliefs:
                t0 = _time.monotonic()
                try:
                    backward_result = await self._inferencer.infer_backward(
                        inference_input, min_confidence=0.3
                    )
                    logger.info(
                        "Dream backward: %d beliefs updated",
                        backward_result.backward_updates,
                    )
                except Exception as e:
                    logger.warning("Backward dreaming failed: %s", e)
                timings["infer_backward"] = _time.monotonic() - t0

            # Consolidation (MC update, contradiction detection, etc.) is
            # deferred to the dedicated /dream and /consolidate endpoints.
            # Running it inline here blocked the gRPC transport for 10+ min.

            t0 = _time.monotonic()
            await self._processor.mark_processed([e.id for e in experiences])
            timings["mark_processed"] = _time.monotonic() - t0
            timings["total"] = _time.monotonic() - reflect_start
            result.timings = {k: round(v, 4) for k, v in timings.items()}
            return result
        except Exception:
            reflection_status = "failed"
            raise
        finally:
            if "total" not in timings:
                timings["total"] = _time.monotonic() - reflect_start
                result.timings = {k: round(v, 4) for k, v in timings.items()}
            try:
                metrics = {
                    "source": reflection_source,
                    "experiences_processed": result.experiences_processed,
                    "patterns_found": len(result.patterns_found),
                    "new_beliefs": len(result.new_beliefs),
                    "updated_beliefs": len(result.updated_beliefs),
                    "contradictions": len(result.contradictions),
                    "timings": result.timings,
                }
                await self._memory._backend.record_reflection_run(
                    run_id=run_id,
                    status=reflection_status,
                    metrics=metrics,
                )
            except Exception:
                logger.debug("Failed to persist reflection run journal")

    async def _query_recent_extraction_items(
        self,
        limit: int | None = None,
    ) -> tuple[list[Belief], list[str], list[str]]:
        """Read unprocessed extraction journal items and resolve beliefs."""
        backend = self._memory._backend
        effective_limit = limit or self._config.max_experiences_per_batch or 100_000
        try:
            items = await backend.get_unprocessed_extraction_items(limit=effective_limit)
        except Exception as e:
            logger.debug("Failed to query extraction journal: %s", e)
            return [], [], []

        beliefs: list[Belief] = []
        extraction_external_ids: list[str] = []
        belief_external_ids: list[str] = []
        seen_beliefs: set[str] = set()

        for item in items:
            metadata = item.get("metadata") or {}
            belief_id_raw = str(metadata.get("belief_id") or "").strip()
            if not belief_id_raw or belief_id_raw in seen_beliefs:
                continue

            belief: Belief | None = None
            try:
                belief = await self._memory.get_belief(UUID(belief_id_raw))
            except Exception:
                belief = None

            if belief is None:
                belief = self._belief_from_extraction_metadata(metadata)
            if belief is None:
                continue

            beliefs.append(belief)
            seen_beliefs.add(str(belief.id))

            ext_id = str(item.get("external_id") or "")
            if ext_id:
                extraction_external_ids.append(ext_id)
            belief_ext_id = str(metadata.get("belief_external_id") or "")
            if belief_ext_id:
                belief_external_ids.append(belief_ext_id)

            if len(beliefs) >= effective_limit:
                break

        logger.info("Found %d extracted beliefs via extraction journal", len(beliefs))
        return beliefs, extraction_external_ids, belief_external_ids

    def _belief_from_extraction_metadata(self, metadata: dict[str, Any]) -> Belief | None:
        """Rehydrate a belief from extraction journal metadata when lookup fails."""
        belief_id_raw = str(metadata.get("belief_id") or "").strip()
        if not belief_id_raw:
            return None
        try:
            belief_id = UUID(belief_id_raw)
        except Exception:
            return None

        triplet: Triplet | None = None
        raw_triplet = metadata.get("triplet")
        if isinstance(raw_triplet, dict):
            tdict = raw_triplet
        elif isinstance(raw_triplet, str) and raw_triplet.strip():
            try:
                parsed = json.loads(raw_triplet)
                tdict = parsed if isinstance(parsed, dict) else {}
            except Exception:
                tdict = {}
        else:
            tdict = {}
        if tdict.get("subject") and tdict.get("predicate") and tdict.get("object"):
            triplet = Triplet(
                subject=str(tdict["subject"]),
                predicate=str(tdict["predicate"]),
                object=str(tdict["object"]),
            )

        source: Source | None = None
        source_doc_id = str(metadata.get("source_doc_id") or "").strip()
        if source_doc_id:
            source = Source(
                id=source_doc_id,
                type=SourceType.OBSERVATION,
                reliability=0.5,
                retrieved_at=utc_now(),
            )

        confidence_raw = metadata.get("confidence", 0.5)
        try:
            confidence = max(0.0, min(1.0, float(confidence_raw)))
        except Exception:
            confidence = 0.5

        tags_raw = metadata.get("tags", [])
        if isinstance(tags_raw, str):
            try:
                parsed_tags = json.loads(tags_raw)
                tags = set(parsed_tags) if isinstance(parsed_tags, list) else {tags_raw}
            except Exception:
                tags = {tags_raw}
        elif isinstance(tags_raw, list | set | tuple):
            tags = {str(tag) for tag in tags_raw}
        else:
            tags = set()

        status_raw = str(metadata.get("status") or BeliefStatus.PROVISIONAL.value)
        try:
            status = BeliefStatus(status_raw)
        except Exception:
            status = BeliefStatus.PROVISIONAL

        return Belief(
            id=belief_id,
            content=str(metadata.get("content") or ""),
            triplet=triplet,
            confidence=confidence,
            source=source,
            status=status,
            tags=tags,
        )

    async def _query_recent_extracted_beliefs(
        self,
        limit: int | None = None,
    ) -> tuple[list[Belief], list[str]]:
        """Query beliefs recently created by the extraction worker.

        Returns beliefs with the 'extracted' tag.  Uses document search
        (BM25) which preserves full metadata including tags.  Triple
        queries (``query_triples``) lose metadata in the CAPI, so the
        old entity-iteration approach never found the "extracted" tag.
        """
        import time as _time

        cooldown = max(0.0, self._config.empty_extracted_cooldown_s)
        if cooldown > 0 and self._last_empty_extracted_scan is not None:
            if (_time.monotonic() - self._last_empty_extracted_scan) < cooldown:
                return [], []

        try:
            backend = self._memory._backend
            effective_limit = limit or self._config.max_experiences_per_batch or 100_000

            # Search belief documents via BM25 — these carry full metadata.
            # Try multiple query terms in case a sparse index misses one.
            query_terms = ["the", "a", "is", "of", "and", "to", "in", "was"]
            extracted: list[Belief] = []
            seen_ids: set[str] = set()
            source_external_ids: list[str] = []

            for query_term in query_terms:
                if extracted:
                    break  # Found results with a previous term

                results = backend._search_by_type(
                    query_term,
                    {backend.NODE_TYPE_BELIEF},
                    target=effective_limit * 4,  # Over-fetch since we filter by metadata
                )

                for r in results:
                    external_id = backend._rget(r, "external_id", "")
                    if not external_id:
                        continue
                    metadata = backend._rget(r, "metadata") or {}
                    if not backend._can_access(metadata, _external_id=external_id):
                        continue
                    belief = backend._search_result_to_belief(r)
                    if not belief:
                        continue
                    if str(belief.id) in seen_ids:
                        continue
                    if "extracted" not in (belief.tags or []):
                        continue
                    if _as_bool(metadata.get("reflection_processed"), default=False):
                        continue
                    extracted.append(belief)
                    source_external_ids.append(external_id)
                    seen_ids.add(str(belief.id))
                    if len(extracted) >= effective_limit:
                        break

            logger.info("Found %d extracted beliefs via document search", len(extracted))
            if extracted:
                self._last_empty_extracted_scan = None
            else:
                self._last_empty_extracted_scan = _time.monotonic()
            return extracted, source_external_ids
        except Exception as e:
            logger.debug("Could not query extracted beliefs: %s", e)
            self._last_empty_extracted_scan = _time.monotonic()
            return [], []

    async def _consolidate_beliefs(
        self, belief_ids: list[UUID],
    ) -> ConsolidationResult:
        """Run consolidation on a list of belief IDs (no candidates needed).

        NOTE: Monte Carlo operations (MC update, MC contradiction detection)
        are intentionally skipped here because they trigger global SiliconDB
        computations that block the gRPC transport for minutes and freeze
        the event loop.  Use the dedicated ``/consolidate`` endpoint for
        heavy MC-based consolidation instead.
        """
        result = ConsolidationResult()
        backend = self._memory._backend

        # All SiliconDB global operations (MC update, MC contradiction
        # detection, triple contradiction detection, uncertain belief queries)
        # are skipped here.  These trigger gRPC calls that hold the GIL and
        # block the async event loop for minutes.  Use /consolidate or /dream
        # for heavy-weight operations.
        logger.info(
            "Consolidating %d beliefs (lifecycle-only, SiliconDB ops deferred)",
            len(belief_ids),
        )

        # Lifecycle evaluation: evaluate all beliefs
        try:
            beliefs_to_evaluate = []
            for bid in belief_ids:
                belief = await self._memory.get_belief(bid)
                if belief:
                    beliefs_to_evaluate.append(belief)

            logger.info(
                "Lifecycle: evaluating %d/%d beliefs",
                len(beliefs_to_evaluate), len(belief_ids),
            )

            if beliefs_to_evaluate:
                try:
                    transitions = await self._lifecycle.evaluate_batch(beliefs_to_evaluate)
                except Exception as llm_err:
                    logger.info("LLM lifecycle failed (%s), using rule-based fallback", llm_err)
                    transitions = await self._lifecycle.evaluate_rule_based(beliefs_to_evaluate)

                for bid, (new_status, reason) in transitions.items():
                    await backend.update_belief_status(bid, new_status, reason)
                    logger.info("Lifecycle: belief %s → %s (%s)", bid, new_status.value, reason)
        except Exception as e:
            logger.warning("Lifecycle evaluation failed: %s", e)

        return result

    async def dream(self) -> dict[str, Any]:
        """Run deep dreaming: hypothesis generation + memory consolidation.

        This is a heavier operation than reflect() and should be run
        periodically (e.g., after processing many batches) rather than
        every cycle.

        Phases are parallelized where possible:
          Batch 1 (parallel): hypotheses, procedures, questions, entity aliases
          Batch 2 (sequential): predicate consolidation (must precede generalization)
          Batch 3 (parallel): memory consolidation + hypothesis validation

        Returns:
            Dict with hypothesis and consolidation statistics + per-phase timing
        """
        import time as _time

        run_id = str(uuid4())
        dream_status = "success"
        stats: dict[str, Any] = {}
        timings: dict[str, float] = {}
        dream_start = _time.monotonic()
        # ------------------------------------------------------------------
        # Batch 1 (parallel): independent phases
        # ------------------------------------------------------------------

        async def _phase_hypotheses() -> None:
            t0 = _time.monotonic()
            try:
                if not self._config.dream_enable_hypothesis_generation:
                    stats["hypothesis_generation_skipped"] = True
                    stats["hypotheses_generated"] = 0
                    stats["hypotheses_committed"] = 0
                    return
                hyp_result = await self._hypothesizer.generate(
                    max_communities=self._config.dream_max_communities,
                    min_community_size=self._config.dream_min_community_size,
                )
                stats["hypotheses_generated"] = len(hyp_result.hypotheses)
                stats["communities_found"] = hyp_result.communities_found
                stats["pagerank_computed"] = hyp_result.pagerank_computed

                committed_hyp = 0
                for h in hyp_result.hypotheses:
                    if h.confidence >= 0.3:
                        belief = await self._generator.commit_belief(h)
                        if belief:
                            committed_hyp += 1
                stats["hypotheses_committed"] = committed_hyp
                logger.info(
                    "Hypothesis generation: %d generated, %d committed",
                    len(hyp_result.hypotheses), committed_hyp,
                )
            except Exception as e:
                logger.warning("Hypothesis generation failed: %s", e)
                stats["hypothesis_error"] = str(e)
            timings["hypotheses"] = round(_time.monotonic() - t0, 1)

        async def _phase_procedures() -> None:
            t0 = _time.monotonic()
            try:
                if not self._config.dream_enable_procedure_detection:
                    stats["procedure_detection_skipped"] = True
                    stats["procedures_created"] = 0
                    return
                detector = ProcedureDetector(self._memory, self._llm)
                procedures_created = await detector.detect_and_commit()
                stats["procedures_created"] = procedures_created
                logger.info("Procedure detection: %d created", procedures_created)
            except Exception as e:
                logger.warning("Procedure detection failed: %s", e)
                stats["procedure_error"] = str(e)
            timings["procedures"] = round(_time.monotonic() - t0, 1)

        async def _phase_questions() -> None:
            t0 = _time.monotonic()
            try:
                if not self._config.dream_enable_question_generation:
                    stats["question_generation_skipped"] = True
                    stats["questions_generated"] = 0
                    return
                qgen = QuestionGenerator(self._memory, self._llm)
                questions_stored = await qgen.generate_and_store()
                stats["questions_generated"] = questions_stored
                if questions_stored > 0:
                    logger.info("Dream: generated %d questions", questions_stored)
            except Exception as e:
                logger.warning("Question generation in dream failed: %s", e)
                stats["question_error"] = str(e)
            timings["questions"] = round(_time.monotonic() - t0, 1)

        async def _phase_entity_consolidation() -> None:
            t0 = _time.monotonic()
            try:
                if not self._config.dream_enable_entity_consolidation:
                    stats["entity_consolidation_skipped"] = True
                    stats["entities_consolidated"] = 0
                    return
                entities_consolidated = await self._consolidate_entities()
                stats["entities_consolidated"] = entities_consolidated
            except Exception as e:
                logger.warning("Entity consolidation failed: %s", e)
                stats["entity_consolidation_error"] = str(e)
            timings["entity_consolidation"] = round(_time.monotonic() - t0, 1)

        # ------------------------------------------------------------------
        # Batch 3 (parallel): memory consolidation + hypothesis validation
        # ------------------------------------------------------------------

        async def _phase_memory_consolidation() -> None:
            t0 = _time.monotonic()
            try:
                consol_stats = await self._consolidator.consolidate()
                stats["generalizations_created"] = consol_stats.generalizations_created
                stats["beliefs_decayed"] = consol_stats.beliefs_decayed
                stats["beliefs_strengthened"] = consol_stats.beliefs_strengthened
                stats["clusters_found"] = consol_stats.clusters_found
                stats["cross_links_created"] = consol_stats.cross_links_created
                logger.info(
                    "Consolidation: %d generalizations, %d decayed, %d strengthened",
                    consol_stats.generalizations_created,
                    consol_stats.beliefs_decayed,
                    consol_stats.beliefs_strengthened,
                )
            except Exception as e:
                logger.warning("Memory consolidation failed: %s", e)
                stats["consolidation_error"] = str(e)
            timings["memory_consolidation"] = round(_time.monotonic() - t0, 1)

        async def _phase_hypothesis_validation() -> None:
            t0 = _time.monotonic()
            try:
                if not self._config.dream_enable_hypothesis_validation:
                    stats["hypothesis_validation_skipped"] = True
                    stats["hypotheses_validated"] = 0
                    return
                validated_count = 0
                if "hypotheses_generated" in stats:
                    backend = self._memory._backend
                    if hasattr(backend, "get_beliefs_by_tag"):
                        hyp_beliefs = await backend.get_beliefs_by_tag(
                            "hypothesis",
                            limit=200,
                            min_confidence=0.0,
                        )
                    else:
                        hyp_beliefs = await self._memory.query_beliefs(
                            query="hypothesis",
                            limit=50,
                            min_confidence=0.0,
                        )
                    max_validations = max(0, int(self._config.dream_max_hypothesis_validations))
                    for hyp in hyp_beliefs:
                        if max_validations and validated_count >= max_validations:
                            break
                        if "hypothesis" not in (hyp.tags or set()):
                            continue
                        if not hyp.triplet:
                            continue
                        query = f"{hyp.triplet.subject} {hyp.triplet.object}"
                        supporting = await self._memory.query_beliefs(
                            query=query, limit=10, min_confidence=0.3,
                        )
                        support_count = sum(
                            1 for b in supporting
                            if b.id != hyp.id and "hypothesis" not in (b.tags or set())
                        )
                        if support_count >= 2:
                            await backend.update_belief_confidence(hyp.id, delta=0.15)
                            validated_count += 1
                        elif support_count == 0:
                            await backend.update_belief_confidence(hyp.id, delta=-0.05)
                stats["hypotheses_validated"] = validated_count
            except Exception as e:
                logger.warning("Hypothesis validation failed: %s", e)
                stats["hypothesis_validation_error"] = str(e)
            timings["hypothesis_validation"] = round(_time.monotonic() - t0, 1)

        try:
            logger.info("Dream batch 1: hypotheses + procedures + questions + entities (parallel)")
            await asyncio.gather(
                _phase_hypotheses(),
                _phase_procedures(),
                _phase_questions(),
                _phase_entity_consolidation(),
            )

            # ------------------------------------------------------------------
            # Batch 2 (sequential): predicate consolidation must precede generalization
            # ------------------------------------------------------------------
            t0 = _time.monotonic()
            try:
                if not self._config.dream_enable_predicate_consolidation:
                    stats["predicate_consolidation_skipped"] = True
                    stats["predicates_merged"] = 0
                    stats["predicates_refitted"] = 0
                else:
                    pred_consolidator = PredicateConsolidator(self._memory, self._llm)
                    pred_result = await pred_consolidator.consolidate_predicates()
                    stats["predicate_consolidation_skipped"] = pred_result.skipped
                    stats["predicates_merged"] = pred_result.merged
                    stats["predicates_refitted"] = pred_result.refitted
                    if not pred_result.skipped:
                        logger.info(
                            "Predicate consolidation: %d merged, %d triples refitted",
                            pred_result.merged, pred_result.refitted,
                        )
            except Exception as e:
                logger.warning("Predicate consolidation failed: %s", e)
                stats["predicate_consolidation_error"] = str(e)
            timings["predicate_consolidation"] = round(_time.monotonic() - t0, 1)

            logger.info("Dream batch 3: memory consolidation + hypothesis validation (parallel)")
            await asyncio.gather(
                _phase_memory_consolidation(),
                _phase_hypothesis_validation(),
            )

            stats["timings"] = timings
            stats["total_seconds"] = round(_time.monotonic() - dream_start, 1)
            logger.info("Dream complete in %.1fs — timings: %s", stats["total_seconds"], timings)
            return stats
        except Exception:
            dream_status = "failed"
            raise
        finally:
            if "timings" not in stats:
                stats["timings"] = timings
            if "total_seconds" not in stats:
                stats["total_seconds"] = round(_time.monotonic() - dream_start, 1)
            try:
                await self._memory._backend.record_dream_run(
                    run_id=run_id,
                    status=dream_status,
                    metrics=stats,
                )
            except Exception:
                logger.debug("Failed to persist dream run journal")

    async def _consolidate_entities(self) -> int:
        """Use LLM to identify entity aliases and register them.

        Queries all entity nodes, asks LLM to group aliases, and
        registers via EntityResolver if available.
        """
        backend = self._memory._backend
        db = backend._db

        # Get all unique subjects and objects
        try:
            subjects = db.all_subjects()
        except Exception:
            return 0

        if len(subjects) < 5:
            return 0

        # Build entity list for LLM
        entity_list = "\n".join(f"- {s}" for s in sorted(subjects)[:200])

        system = (
            "You identify entity aliases in a knowledge graph. "
            "Below are entity names. Group any that refer to the same real-world entity.\n\n"
            "Rules:\n"
            "- Only group entities you are CERTAIN are the same\n"
            "- Pick the most complete name as canonical\n"
            "- Common patterns: abbreviations, first/last name vs full name, acronyms\n\n"
            'Respond with JSON: {"aliases": [{"canonical": "Full Name", '
            '"variants": ["Short Name", "Abbreviation"]}, ...]}\n'
            "Return empty list if no aliases found."
        )
        user = f"Entities:\n{entity_list}"

        try:
            prompt = f"{system}\n\n{user}"
            if hasattr(self._llm, "complete"):
                response = await self._llm.complete(
                    prompt, system=system, temperature=0.3, max_tokens=2048,
                )
            elif hasattr(self._llm, "generate"):
                response = await self._llm.generate(
                    prompt, max_tokens=2048, temperature=0.3,
                )
            else:
                return 0

            parsed = _parse_json(response)
            if not parsed or "aliases" not in parsed:
                return 0

            registered = 0
            for group in parsed["aliases"]:
                canonical = group.get("canonical", "")
                variants = group.get("variants", [])
                if not canonical or not variants:
                    continue
                for variant in variants:
                    try:
                        # Store alias mapping in working memory for now
                        key = f"entity_alias_{variant}"
                        await self._memory.set_context(
                            key,
                            {"alias": variant, "canonical": canonical},
                            ttl_seconds=86400 * 7,  # 7 day TTL
                        )
                        registered += 1
                    except Exception:
                        pass

            logger.info("Entity consolidation: %d aliases registered", registered)
            return registered
        except Exception as e:
            logger.debug("Entity consolidation LLM call failed: %s", e)
            return 0

    async def _review_active_decisions(self) -> None:
        """Review active decisions for assumption confidence drift."""
        try:
            decisions = await self._memory.recall_decisions(
                query="*", k=50, min_confidence=0.0
            )
        except Exception:
            return

        for decision in decisions:
            if decision.status != DecisionStatus.ACTIVE:
                continue

            for assumption in decision.assumptions:
                if not assumption.is_critical:
                    continue

                try:
                    belief = await self._memory.get_belief(assumption.belief_id)
                except Exception:
                    continue

                if belief is None:
                    continue

                drift = abs(belief.confidence - assumption.confidence_at_decision)
                if drift > 0.2:
                    decision.status = DecisionStatus.REVISIT_SUGGESTED
                    reason = (
                        f"Auto-flagged: assumption '{assumption.description}' "
                        f"drifted {drift:.2f} from {assumption.confidence_at_decision:.2f} "
                        f"to {belief.confidence:.2f}"
                    )
                    try:
                        await self._memory._backend.update_decision_status(
                            decision.id,
                            DecisionStatus.REVISIT_SUGGESTED,
                            reason=reason,
                        )
                    except Exception:
                        pass
                    break

    async def reflect_incremental(
        self,
        batch_size: int = 20,
        dream_every: int = 5,
    ) -> ReflectionResult:
        """Run reflection in incremental batches with periodic dreaming.

        Processes the full backlog of unprocessed experiences, running
        hypothesis generation and memory consolidation every N batches.

        Args:
            batch_size: Number of experiences per batch
            dream_every: Run dream() every N batches (0 = never)

        Returns:
            Combined ReflectionResult from all batches
        """
        combined = ReflectionResult()
        batch_count = 0

        while True:
            result = await self.reflect(
                max_experiences=batch_size, auto_commit=True
            )

            if result.experiences_processed == 0:
                break

            batch_count += 1

            # Combine results
            combined.experiences_processed += result.experiences_processed
            combined.patterns_found.extend(result.patterns_found)
            combined.new_beliefs.extend(result.new_beliefs)
            combined.updated_beliefs.extend(result.updated_beliefs)
            combined.contradictions.extend(result.contradictions)

            logger.info(
                "Batch %d: %d experiences, %d patterns, %d beliefs",
                batch_count, result.experiences_processed,
                len(result.patterns_found), len(result.new_beliefs),
            )

            # Periodic deep dreaming
            if dream_every > 0 and batch_count % dream_every == 0:
                logger.info("Running periodic dream at batch %d", batch_count)
                try:
                    await self.dream()
                except Exception as e:
                    logger.warning("Periodic dream failed: %s", e)

        # Final dream after all batches
        if batch_count > 0 and dream_every > 0:
            logger.info("Running final dream after %d batches", batch_count)
            try:
                await self.dream()
            except Exception as e:
                logger.warning("Final dream failed: %s", e)

        return combined

    async def commit_belief(
        self,
        candidate: BeliefCandidate,
    ) -> Belief | None:
        """Commit a specific belief candidate."""
        return await self._generator.commit_belief(candidate, require_approval=False)

    async def commit_all_valid(
        self,
        result: ReflectionResult,
    ) -> list[Belief]:
        """Commit all non-contested beliefs from a reflection result."""
        return await self._generator.commit_all_valid(result.new_beliefs)

    async def get_pending_candidates(
        self,
        min_confidence: float = 0.5,
    ) -> list[BeliefCandidate]:
        """Get belief candidates waiting for approval."""
        original = self._config.auto_commit_beliefs
        self._config.auto_commit_beliefs = False

        try:
            result = await self.reflect()
            return [c for c in result.new_beliefs if c.confidence >= min_confidence]
        finally:
            self._config.auto_commit_beliefs = original

    async def analyze_patterns(
        self,
        max_experiences: int = 100,
    ) -> list[Pattern]:
        """Analyze experiences for patterns without generating beliefs."""
        experiences = await self._processor.fetch_unprocessed(max_experiences)
        if not experiences:
            return []

        groups = await self._processor.process_batch(experiences)
        if not groups:
            return []

        return await self._extractor.extract_patterns(groups)

    def update_config(self, **kwargs) -> None:
        """Update configuration settings."""
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)

        # Propagate to components
        self._processor._config = self._config
        if self._extractor:
            self._extractor._config = self._config
        self._generator._config = self._config
