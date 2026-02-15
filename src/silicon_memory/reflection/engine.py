"""Main reflection engine orchestrating the cognitive pipeline.

Implements a multi-phase cognitive process:
1. Extract — LLM extracts structured knowledge from experiences
2. Generate — Convert patterns into belief candidates
3. Commit — Store validated beliefs
4. Dream Forward — Transitive inference discovers hidden connections
5. Dream Backward — New evidence updates existing beliefs
6. Hypothesize — Clustering + LLM generates hypotheses
7. Consolidate — Monte Carlo, propagation, generalization, decay
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, TYPE_CHECKING
from uuid import UUID

from silicon_memory.core.types import Belief
from silicon_memory.core.decision import DecisionStatus
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

if TYPE_CHECKING:
    from silicon_memory.memory.silicondb_router import SiliconMemory


class ReflectionEngine:
    """Main reflection engine that orchestrates memory consolidation.

    The reflection engine processes episodic memories (experiences) and
    extracts semantic knowledge (beliefs), mimicking human memory
    consolidation during sleep.

    The full cognitive pipeline:
    1. Fetch unprocessed experiences
    2. Group related experiences
    3. Extract patterns via LLM (facts, relationships, arguments, events)
    4. Generate belief candidates (entity-aware, with source grounding)
    5. Validate against existing knowledge
    6. Commit valid beliefs
    7. Dream forward — transitive inference discovers indirect connections
    8. Dream backward — new evidence updates existing beliefs
    9. Monte Carlo consolidation (co-occurrence, propagation, contradictions)
    10. Mark experiences as processed

    Extended operations (run periodically, not every cycle):
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
    ) -> None:
        self._memory = memory
        self._llm = llm
        self._config = config or ReflectionConfig()

        # Core extraction pipeline
        self._processor = ExperienceProcessor(memory, self._config)
        self._extractor = LLMPatternExtractor(memory, llm, self._config)
        self._generator = BeliefGenerator(memory, self._config)

        # Cognitive extensions
        self._inferencer = TransitiveInferenceEngine(memory, self._config)
        self._hypothesizer = HypothesisGenerator(memory, llm, self._config)
        self._consolidator = MemoryConsolidator(memory, self._config)

    @property
    def config(self) -> ReflectionConfig:
        """Get the current configuration."""
        return self._config

    async def reflect(
        self,
        max_experiences: int | None = None,
        auto_commit: bool | None = None,
    ) -> ReflectionResult:
        """Run a reflection cycle with full cognitive pipeline.

        Args:
            max_experiences: Override max experiences to process
            auto_commit: Override auto-commit setting

        Returns:
            ReflectionResult with new beliefs, patterns, and stats
        """
        result = ReflectionResult()

        # Step 1: Fetch unprocessed experiences
        limit = max_experiences or self._config.max_experiences_per_batch
        experiences = await self._processor.fetch_unprocessed(limit)

        if not experiences:
            return result

        result.experiences_processed = len(experiences)

        # Step 2: Group related experiences
        groups = await self._processor.process_batch(experiences)

        if not groups:
            await self._processor.mark_processed([e.id for e in experiences])
            return result

        # Step 3: Extract patterns via LLM (entity-aware, source-grounded)
        patterns = await self._extractor.extract_patterns(groups)
        result.patterns_found = patterns

        if not patterns:
            await self._processor.mark_processed([e.id for e in experiences])
            return result

        # Step 4: Generate belief candidates
        candidates = await self._generator.generate_beliefs(patterns)
        result.new_beliefs = candidates

        # Step 5: Track contradictions
        for candidate in candidates:
            for contra_id in candidate.contradicts:
                result.contradictions.append((candidate.id, contra_id))

        # Step 5b: Review active decisions for assumption drift
        await self._review_active_decisions()

        # Step 6: Commit beliefs
        committed_beliefs: list[Belief] = []
        committed_ids: list[UUID] = []
        should_commit = auto_commit if auto_commit is not None else self._config.auto_commit_beliefs
        if should_commit:
            for candidate in candidates:
                if not candidate.is_contested:
                    belief = await self._generator.commit_belief(candidate)
                    if belief:
                        committed_beliefs.append(belief)
                        committed_ids.append(belief.id)
                        result.updated_beliefs.append((belief.id, belief.confidence))

        # Step 7: Dream forward — transitive inference
        if committed_beliefs:
            try:
                inference_result = await self._inferencer.infer_forward(
                    committed_beliefs, max_hops=3, min_confidence=0.3
                )
                # Commit inferred beliefs
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

        # Step 8: Dream backward — update existing beliefs
        if committed_beliefs:
            try:
                backward_result = await self._inferencer.infer_backward(
                    committed_beliefs, min_confidence=0.3
                )
                logger.info(
                    "Dream backward: %d beliefs updated",
                    backward_result.backward_updates,
                )
            except Exception as e:
                logger.warning("Backward dreaming failed: %s", e)

        # Step 9: Monte Carlo consolidation
        if committed_ids:
            result.consolidation = await self._consolidate(
                committed_ids, candidates
            )

        # Step 10: Mark experiences as processed
        await self._processor.mark_processed([e.id for e in experiences])

        return result

    async def dream(self) -> dict[str, Any]:
        """Run deep dreaming: hypothesis generation + memory consolidation.

        This is a heavier operation than reflect() and should be run
        periodically (e.g., after processing many batches) rather than
        every cycle.

        Returns:
            Dict with hypothesis and consolidation statistics
        """
        stats: dict[str, Any] = {}

        # Phase 1: Generate hypotheses from graph structure
        try:
            hyp_result = await self._hypothesizer.generate(
                max_communities=10,
                min_community_size=3,
            )
            stats["hypotheses_generated"] = len(hyp_result.hypotheses)
            stats["communities_found"] = hyp_result.communities_found
            stats["pagerank_computed"] = hyp_result.pagerank_computed

            # Commit non-trivial hypotheses
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

        # Phase 2: Memory consolidation (generalization + decay + clustering)
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

        return stats

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
                    try:
                        await self._memory._backend.record_decision_outcome(
                            decision.id,
                            f"Auto-flagged: assumption '{assumption.description}' "
                            f"drifted {drift:.2f} from {assumption.confidence_at_decision:.2f} "
                            f"to {belief.confidence:.2f}",
                        )
                    except Exception:
                        pass
                    break

    async def _consolidate(
        self,
        committed_ids: list[UUID],
        candidates: list[BeliefCandidate],
    ) -> ConsolidationResult:
        """Run Monte Carlo consolidation on newly committed beliefs."""
        result = ConsolidationResult()
        backend = self._memory._backend

        # 1. Build co-occurrences
        try:
            exp_to_beliefs: dict[str, list[UUID]] = {}
            for candidate in candidates:
                if candidate.id not in committed_ids:
                    continue
                for exp_id in candidate.source_experiences:
                    key = str(exp_id)
                    if key not in exp_to_beliefs:
                        exp_to_beliefs[key] = []
                    exp_to_beliefs[key].append(candidate.id)

            for exp_key, belief_ids in exp_to_beliefs.items():
                if len(belief_ids) >= 2:
                    await backend.build_cooccurrences(
                        belief_ids, session_id=exp_key
                    )
                    result.cooccurrence_links += len(belief_ids) * (len(belief_ids) - 1) // 2
        except Exception as e:
            logger.warning("Co-occurrence building failed: %s", e)

        # 2. Monte Carlo probability update
        try:
            evidence = []
            for candidate in candidates:
                if candidate.id not in committed_ids:
                    continue
                external_id = backend._build_external_id("belief", candidate.id)
                evidence.append({
                    "external_id": external_id,
                    "confidence": candidate.confidence,
                })

            if evidence:
                mc_results = await backend.monte_carlo_update(
                    evidence, samples=10000, apply=True
                )
                result.mc_updates = len(mc_results)
        except Exception as e:
            logger.warning("Monte Carlo update failed: %s", e)

        # 3. Propagate high-confidence beliefs
        try:
            for candidate in candidates:
                if candidate.id not in committed_ids:
                    continue
                if candidate.confidence >= 0.8:
                    updates = await backend.propagate_belief(
                        candidate.id,
                        confidence=candidate.confidence,
                        decay=0.5,
                    )
                    result.propagations += len(updates)
        except Exception as e:
            logger.warning("Belief propagation failed: %s", e)

        # 4. Detect contradictions
        try:
            mc_contras = await backend.detect_mc_contradictions(
                samples=5000, min_conflict_score=0.6
            )
            result.mc_contradictions = len(mc_contras)
        except Exception as e:
            logger.warning("MC contradiction detection failed: %s", e)

        try:
            triple_contras = await backend.detect_triple_contradictions(
                min_probability=0.3
            )
            result.triple_contradictions = len(triple_contras)
        except Exception as e:
            logger.warning("Triple contradiction detection failed: %s", e)

        # 5. Count uncertain beliefs
        try:
            uncertain = await backend.get_uncertain_beliefs(min_entropy=0.7, k=100)
            result.uncertain_beliefs = len(uncertain)
        except Exception as e:
            logger.warning("Uncertainty query failed: %s", e)

        return result

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
        self._extractor._config = self._config
        self._generator._config = self._config
