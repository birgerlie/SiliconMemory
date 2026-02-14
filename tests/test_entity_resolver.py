"""TDD tests for the entity resolution system.

Tests are organized bottom-up:
  1. Types — data structures
  2. Cache — alias→canonical in-memory lookup
  3. RuleEngine — two-tier regex (detect + extract)
  4. Resolver — orchestration (detect → extract → disambiguate → cache)
  5. Learner — LLM-based bootstrap and incremental learning
"""

from __future__ import annotations

import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from silicon_memory.entities.types import (
    Candidate,
    DetectorRule,
    EntityReference,
    ExtractorRule,
    ResolveResult,
)


# =====================================================================
# 1. Types
# =====================================================================


class TestTypes:
    def test_candidate_fields(self):
        c = Candidate(text="§ 10-4", span=(5, 11), context_text="til § 10-4 første", detector_id="d1")
        assert c.text == "§ 10-4"
        assert c.span == (5, 11)
        assert c.context_text == "til § 10-4 første"

    def test_entity_reference_fields(self):
        ref = EntityReference(
            text="aml. § 10-4",
            canonical_id="arbeidsmiljøloven/§10-4",
            entity_type="law_ref",
            confidence=0.95,
            span=(0, 11),
            context_text="I henhold til aml. § 10-4 første ledd",
        )
        assert ref.canonical_id == "arbeidsmiljøloven/§10-4"
        assert ref.rule_id is None

    def test_resolve_result_defaults_empty(self):
        r = ResolveResult()
        assert r.resolved == []
        assert r.unresolved == []

    def test_resolve_result_supports_len_on_resolved(self):
        """Adapters call len(result.resolved)."""
        r = ResolveResult(resolved=[
            EntityReference("x", "y", "t", 0.9, (0, 1), "ctx"),
        ])
        assert len(r.resolved) == 1

    def test_detector_rule_has_default_timestamp(self):
        d = DetectorRule(id="d1", pattern=r"§\s*\d+", description="law sections")
        assert d.created_at is not None

    def test_extractor_rule_has_defaults(self):
        e = ExtractorRule(
            id="e1",
            entity_type="law_ref",
            detector_ids=["d1"],
            pattern=r"(?:aml|arbeidsmiljøloven)\.?\s*§\s*(\d+-\d+)",
            normalize_template="{entity_type}/{match}",
        )
        assert e.confidence == 1.0
        assert e.context_threshold == 0.6
        assert e.context_embedding is None


# =====================================================================
# 2. EntityCache
# =====================================================================


class TestEntityCache:
    @pytest.fixture
    def cache(self):
        from silicon_memory.entities.cache import EntityCache
        return EntityCache()

    def test_lookup_returns_none_for_unknown(self, cache):
        assert cache.lookup("nonexistent") is None

    def test_store_and_lookup(self, cache):
        cache.store("aml.", "arbeidsmiljøloven", "law")
        assert cache.lookup("aml.") == "arbeidsmiljøloven"

    def test_lookup_is_case_insensitive(self, cache):
        cache.store("AML.", "arbeidsmiljøloven", "law")
        assert cache.lookup("aml.") == "arbeidsmiljøloven"
        assert cache.lookup("AML.") == "arbeidsmiljøloven"

    def test_lookup_strips_whitespace(self, cache):
        cache.store("  aml.  ", "arbeidsmiljøloven", "law")
        assert cache.lookup("aml.") == "arbeidsmiljøloven"

    def test_store_multiple_aliases_to_same_canonical(self, cache):
        cache.store("aml.", "arbeidsmiljøloven", "law")
        cache.store("arbeidsmiljøloven", "arbeidsmiljøloven", "law")
        cache.store("arbmiljøl", "arbeidsmiljøloven", "law")
        assert cache.lookup("aml.") == "arbeidsmiljøloven"
        assert cache.lookup("arbeidsmiljøloven") == "arbeidsmiljøloven"
        assert cache.lookup("arbmiljøl") == "arbeidsmiljøloven"

    def test_get_entity_type(self, cache):
        cache.store("HR-2023-1234-A", "HR-2023-1234-A", "case_id")
        assert cache.get_type("HR-2023-1234-A") == "case_id"

    def test_get_type_returns_none_for_unknown(self, cache):
        assert cache.get_type("nonexistent") is None

    def test_all_aliases_for_canonical(self, cache):
        cache.store("aml.", "arbeidsmiljøloven", "law")
        cache.store("arbmiljøl", "arbeidsmiljøloven", "law")
        aliases = cache.aliases_for("arbeidsmiljøloven")
        assert "aml." in aliases
        assert "arbmiljøl" in aliases

    def test_aliases_for_unknown_returns_empty(self, cache):
        assert cache.aliases_for("nonexistent") == set()

    def test_size(self, cache):
        assert cache.size == 0
        cache.store("a", "A", "t")
        cache.store("b", "A", "t")
        assert cache.size == 2


# =====================================================================
# 3. RuleEngine
# =====================================================================


class TestRuleEngine:
    @pytest.fixture
    def engine(self):
        from silicon_memory.entities.rules import RuleEngine
        return RuleEngine()

    def test_detect_no_rules_returns_empty(self, engine):
        assert engine.detect("hello world") == []

    def test_add_detector_and_detect(self, engine):
        rule = DetectorRule(id="d1", pattern=r"§\s*\d+(?:-\d+)?", description="law sections")
        engine.add_detector(rule)
        candidates = engine.detect("I henhold til § 10-4 første ledd")
        assert len(candidates) == 1
        assert candidates[0].text == "§ 10-4"
        assert candidates[0].detector_id == "d1"

    def test_detect_captures_context_window(self, engine):
        rule = DetectorRule(id="d1", pattern=r"§\s*\d+(?:-\d+)?", description="law sections")
        engine.add_detector(rule)
        text = "I henhold til § 10-4 første ledd i arbeidsmiljøloven"
        candidates = engine.detect(text)
        ctx = candidates[0].context_text
        assert "henhold" in ctx
        assert "første" in ctx

    def test_detect_multiple_matches(self, engine):
        rule = DetectorRule(id="d1", pattern=r"§\s*\d+(?:-\d+)?", description="law sections")
        engine.add_detector(rule)
        text = "Se § 10-4 og § 14-9 for detaljer"
        candidates = engine.detect(text)
        assert len(candidates) == 2

    def test_detect_span_offsets_are_correct(self, engine):
        rule = DetectorRule(id="d1", pattern=r"§\s*\d+", description="law sections")
        engine.add_detector(rule)
        text = "abc § 5 xyz"
        candidates = engine.detect(text)
        start, end = candidates[0].span
        assert text[start:end] == "§ 5"

    def test_detect_deduplicates_overlapping_spans(self, engine):
        """Two detectors matching the same span should produce one candidate."""
        d1 = DetectorRule(id="d1", pattern=r"§\s*\d+(?:-\d+)?", description="narrow")
        d2 = DetectorRule(id="d2", pattern=r"§\s*\d+[-\d]*", description="broad")
        engine.add_detector(d1)
        engine.add_detector(d2)
        candidates = engine.detect("Se § 10-4 her")
        assert len(candidates) == 1

    def test_extract_no_extractors_returns_empty(self, engine):
        candidates = [Candidate("§ 10-4", (3, 9), "til § 10-4 første", "d1")]
        result = engine.extract(candidates)
        assert result == {}

    def test_add_extractor_and_extract(self, engine):
        ext = ExtractorRule(
            id="e1",
            entity_type="law_ref",
            detector_ids=["d1"],
            pattern=r"§\s*(\d+(?:-\d+)?)",
            normalize_template="law/{match}",
        )
        engine.add_extractor(ext)
        candidates = [Candidate("§ 10-4", (3, 9), "til § 10-4 første", "d1")]
        result = engine.extract(candidates)
        assert 0 in result
        refs = result[0]
        assert len(refs) == 1
        assert refs[0].entity_type == "law_ref"
        assert refs[0].text == "§ 10-4"

    def test_extract_applies_normalize_template(self, engine):
        ext = ExtractorRule(
            id="e1",
            entity_type="case_id",
            detector_ids=["d1"],
            pattern=r"(HR-\d{4}-\d+-[A-Z])",
            normalize_template="{match}",
        )
        engine.add_extractor(ext)
        candidates = [Candidate("HR-2023-1234-A", (0, 14), "dom i HR-2023-1234-A den", "d1")]
        result = engine.extract(candidates)
        ref = result[0][0]
        assert ref.canonical_id == "HR-2023-1234-A"

    def test_extract_multiple_extractors_same_candidate(self, engine):
        """Ambiguous match — two extractors match the same candidate."""
        e1 = ExtractorRule(
            id="e1", entity_type="law_ref", detector_ids=["d1"],
            pattern=r"§\s*(\d+)", normalize_template="law/{match}",
        )
        e2 = ExtractorRule(
            id="e2", entity_type="regulation_ref", detector_ids=["d1"],
            pattern=r"§\s*(\d+)", normalize_template="reg/{match}",
        )
        engine.add_extractor(e1)
        engine.add_extractor(e2)
        candidates = [Candidate("§ 5", (0, 3), "i forskrift § 5 annet ledd", "d1")]
        result = engine.extract(candidates)
        assert len(result[0]) == 2

    def test_extract_ignores_non_matching_extractor(self, engine):
        ext = ExtractorRule(
            id="e1", entity_type="case_id", detector_ids=["d1"],
            pattern=r"HR-\d{4}-\d+-[A-Z]", normalize_template="{match}",
        )
        engine.add_extractor(ext)
        candidates = [Candidate("§ 10-4", (0, 6), "context", "d1")]
        result = engine.extract(candidates)
        assert result == {}

    def test_normalize_template_lowercases(self, engine):
        ext = ExtractorRule(
            id="e1", entity_type="project", detector_ids=["d1"],
            pattern=r"(PROJ-\d+)", normalize_template="{match_lower}",
        )
        engine.add_extractor(ext)
        candidates = [Candidate("PROJ-123", (0, 8), "context", "d1")]
        result = engine.extract(candidates)
        assert result[0][0].canonical_id == "proj-123"

    def test_invalid_regex_raises_on_add(self, engine):
        bad = DetectorRule(id="d1", pattern=r"[invalid", description="broken")
        with pytest.raises(re.error):
            engine.add_detector(bad)

    def test_get_extractor_by_id(self, engine):
        ext = ExtractorRule(
            id="e1", entity_type="law_ref", detector_ids=["d1"],
            pattern=r"§\s*\d+", normalize_template="{match}",
        )
        engine.add_extractor(ext)
        assert engine.get_extractor("e1") is ext
        assert engine.get_extractor("nonexistent") is None


# =====================================================================
# 4. EntityResolver
# =====================================================================


class TestEntityResolver:
    @pytest.fixture
    def resolver(self):
        from silicon_memory.entities.cache import EntityCache
        from silicon_memory.entities.resolver import EntityResolver
        from silicon_memory.entities.rules import RuleEngine

        cache = EntityCache()
        rules = RuleEngine()
        return EntityResolver(cache=cache, rules=rules)

    @pytest.fixture
    def resolver_with_rules(self):
        """Resolver pre-loaded with law detection rules and cache entries."""
        from silicon_memory.entities.cache import EntityCache
        from silicon_memory.entities.resolver import EntityResolver
        from silicon_memory.entities.rules import RuleEngine

        cache = EntityCache()
        cache.store("aml.", "arbeidsmiljøloven", "law")
        cache.store("arbeidsmiljøloven", "arbeidsmiljøloven", "law")

        rules = RuleEngine()
        rules.add_detector(DetectorRule(
            id="d_law", pattern=r"(?:aml\.|arbeidsmiljøloven)\s*§\s*\d+(?:-\d+)?",
            description="law references",
        ))
        rules.add_extractor(ExtractorRule(
            id="e_law", entity_type="law_ref", detector_ids=["d_law"],
            pattern=r"((?:aml\.|arbeidsmiljøloven)\s*§\s*\d+(?:-\d+)?)",
            normalize_template="{match}",
        ))

        return EntityResolver(cache=cache, rules=rules)

    @pytest.mark.asyncio
    async def test_resolve_empty_text(self, resolver):
        result = await resolver.resolve("")
        assert result.resolved == []
        assert result.unresolved == []

    @pytest.mark.asyncio
    async def test_resolve_no_entities(self, resolver):
        result = await resolver.resolve("Just some plain text with no entity references.")
        assert result.resolved == []

    @pytest.mark.asyncio
    async def test_resolve_detects_known_entity(self, resolver_with_rules):
        text = "I henhold til aml. § 10-4 første ledd"
        result = await resolver_with_rules.resolve(text)
        assert len(result.resolved) == 1
        ref = result.resolved[0]
        assert ref.entity_type == "law_ref"
        assert "§ 10-4" in ref.text

    @pytest.mark.asyncio
    async def test_resolve_with_extractor_match_is_resolved(self):
        """Extractor match = resolved, even without cache entry."""
        from silicon_memory.entities.cache import EntityCache
        from silicon_memory.entities.resolver import EntityResolver
        from silicon_memory.entities.rules import RuleEngine

        cache = EntityCache()  # Empty cache — doesn't gate resolution
        rules = RuleEngine()
        rules.add_detector(DetectorRule(
            id="d1", pattern=r"HR-\d{4}-\d+-[A-Z]", description="case ids",
        ))
        rules.add_extractor(ExtractorRule(
            id="e1", entity_type="case_id", detector_ids=["d1"],
            pattern=r"(HR-\d{4}-\d+-[A-Z])", normalize_template="{match}",
        ))
        resolver = EntityResolver(cache=cache, rules=rules)
        result = await resolver.resolve("Dom i HR-2023-1234-A ble avsagt")
        assert len(result.resolved) == 1
        assert result.resolved[0].canonical_id == "HR-2023-1234-A"

    @pytest.mark.asyncio
    async def test_resolve_unresolved_when_no_extractor_matches(self):
        """Detected by detector but no extractor matches → unresolved."""
        from silicon_memory.entities.cache import EntityCache
        from silicon_memory.entities.resolver import EntityResolver
        from silicon_memory.entities.rules import RuleEngine

        cache = EntityCache()
        rules = RuleEngine()
        rules.add_detector(DetectorRule(
            id="d1", pattern=r"[A-Z]{3}-\d{3}", description="broad codes",
        ))
        # No extractors added — nothing can extract
        resolver = EntityResolver(cache=cache, rules=rules)
        result = await resolver.resolve("See ABC-123 for details")
        assert len(result.unresolved) == 1
        assert "ABC-123" in result.unresolved

    @pytest.mark.asyncio
    async def test_resolve_multiple_entities_in_text(self, resolver_with_rules):
        resolver_with_rules.cache.store("HR-2023-1234-A", "HR-2023-1234-A", "case_id")
        resolver_with_rules.rules.add_detector(DetectorRule(
            id="d_case", pattern=r"HR-\d{4}-\d+-[A-Z]", description="case ids",
        ))
        resolver_with_rules.rules.add_extractor(ExtractorRule(
            id="e_case", entity_type="case_id", detector_ids=["d_case"],
            pattern=r"(HR-\d{4}-\d+-[A-Z])", normalize_template="{match}",
        ))
        text = "I henhold til aml. § 10-4, se også HR-2023-1234-A"
        result = await resolver_with_rules.resolve(text)
        assert len(result.resolved) == 2
        types = {r.entity_type for r in result.resolved}
        assert "law_ref" in types
        assert "case_id" in types

    @pytest.mark.asyncio
    async def test_register_alias(self, resolver):
        await resolver.register_alias("strl.", "straffeloven", "law")
        assert resolver.cache.lookup("strl.") == "straffeloven"

    @pytest.mark.asyncio
    async def test_resolve_result_matches_adapter_interface(self, resolver):
        """Adapters check: if result.resolved: resolved_count += len(result.resolved)"""
        result = await resolver.resolve("no entities here")
        assert hasattr(result, "resolved")
        assert isinstance(result.resolved, list)
        assert len(result.resolved) == 0

    @pytest.mark.asyncio
    async def test_resolve_single_name(self, resolver):
        await resolver.register_alias("donald trump", "person/donald-trump", "person")
        ref = await resolver.resolve_single("Donald Trump")
        assert ref is not None
        assert ref.canonical_id == "person/donald-trump"

    @pytest.mark.asyncio
    async def test_resolve_single_unknown_returns_none(self, resolver):
        ref = await resolver.resolve_single("Unknown Person")
        assert ref is None


# =====================================================================
# 5. RuleLearner / Bootstrapper
# =====================================================================


class TestRuleLearner:
    @pytest.fixture
    def mock_llm(self):
        llm = AsyncMock()
        return llm

    @pytest.fixture
    def learner(self, mock_llm):
        from silicon_memory.entities.learner import RuleLearner
        return RuleLearner(llm=mock_llm)

    @pytest.mark.asyncio
    async def test_bootstrap_extracts_entities_from_text(self, learner, mock_llm):
        """LLM analyzes sample text and returns entity examples."""
        from silicon_memory.entities.learner import ExtractedEntity, ExtractResult

        mock_llm.generate_structured.return_value = ExtractResult(entities=[
            ExtractedEntity(text="aml. § 10-4", entity_type="law_ref",
                            canonical="arbeidsmiljøloven/§10-4",
                            context="I henhold til aml. § 10-4 første ledd"),
            ExtractedEntity(text="HR-2023-1234-A", entity_type="case_id",
                            canonical="HR-2023-1234-A",
                            context="Høyesterett avsa dom i HR-2023-1234-A"),
        ])
        examples = await learner.extract_entities(
            "I henhold til aml. § 10-4 første ledd. Se HR-2023-1234-A."
        )
        assert len(examples) == 2
        assert examples[0]["entity_type"] == "law_ref"
        assert examples[1]["entity_type"] == "case_id"

    @pytest.mark.asyncio
    async def test_generate_rules_from_examples(self, learner, mock_llm):
        """LLM generates detector + extractor rules from entity examples."""
        from silicon_memory.entities.learner import GeneratedRule, RulesResult

        examples = [
            {"text": "aml. § 10-4", "entity_type": "law_ref",
             "canonical": "arbeidsmiljøloven/§10-4",
             "context": "I henhold til aml. § 10-4 første ledd"},
            {"text": "aml. § 14-9", "entity_type": "law_ref",
             "canonical": "arbeidsmiljøloven/§14-9",
             "context": "Etter aml. § 14-9 annet ledd"},
        ]
        mock_llm.generate_structured.return_value = RulesResult(rules=[
            GeneratedRule(
                detector_pattern=r"(?:aml\.|arbeidsmiljøloven)\s*§\s*\d+(?:-\d+)?",
                detector_description="arbeidsmiljøloven references",
                extractor_pattern=r"((?:aml\.|arbeidsmiljøloven)\s*§\s*(\d+(?:-\d+)?))",
                entity_type="law_ref",
                normalize_template="arbeidsmiljøloven/§{group1}",
                context_examples=["I henhold til", "Etter", "jf."],
            ),
        ])
        detectors, extractors = await learner.generate_rules(examples)
        assert len(detectors) >= 1
        assert len(extractors) >= 1
        assert extractors[0].entity_type == "law_ref"

    @pytest.mark.asyncio
    async def test_generate_rules_validates_regex(self, learner, mock_llm):
        """Rules with invalid regex are filtered out."""
        from silicon_memory.entities.learner import GeneratedRule, RulesResult

        mock_llm.generate_structured.return_value = RulesResult(rules=[
            GeneratedRule(
                detector_pattern=r"[invalid",
                detector_description="broken",
                extractor_pattern=r"(good-\d+)",
                entity_type="test",
                normalize_template="{match}",
                context_examples=[],
            ),
            GeneratedRule(
                detector_pattern=r"good-\d+",
                detector_description="working",
                extractor_pattern=r"(good-\d+)",
                entity_type="test",
                normalize_template="{match}",
                context_examples=[],
            ),
        ])
        detectors, extractors = await learner.generate_rules([
            {"text": "good-123", "entity_type": "test", "canonical": "good-123", "context": "x"},
        ])
        assert len(detectors) == 1
        assert detectors[0].pattern == r"good-\d+"

    @pytest.mark.asyncio
    async def test_bootstrap_end_to_end(self, learner, mock_llm):
        """Full bootstrap: text → extract entities → generate rules."""
        from silicon_memory.entities.learner import (
            ExtractedEntity,
            ExtractResult,
            GeneratedRule,
            RulesResult,
        )

        mock_llm.generate_structured.side_effect = [
            ExtractResult(entities=[
                ExtractedEntity(text="§ 10-4", entity_type="law_ref",
                                canonical="§10-4", context="til § 10-4 første"),
            ]),
            RulesResult(rules=[
                GeneratedRule(
                    detector_pattern=r"§\s*\d+(?:-\d+)?",
                    detector_description="law section references",
                    extractor_pattern=r"(§\s*(\d+(?:-\d+)?))",
                    entity_type="law_ref",
                    normalize_template="§{group1}",
                    context_examples=["i henhold til", "etter", "jf."],
                ),
            ]),
        ]
        detectors, extractors, aliases = await learner.bootstrap(
            "I henhold til § 10-4 første ledd"
        )
        assert len(detectors) >= 1
        assert len(extractors) >= 1

    @pytest.mark.asyncio
    async def test_discover_parenthetical_aliases(self, learner):
        """Heuristic alias detection — no LLM needed."""
        text = "arbeidsmiljøloven (aml.) regulerer arbeidsforhold"
        aliases = learner.discover_aliases(text)
        assert ("aml.", "arbeidsmiljøloven") in aliases

    @pytest.mark.asyncio
    async def test_discover_aliases_multiple(self, learner):
        text = "straffeloven (strl.) og arbeidsmiljøloven (aml.) gjelder"
        aliases = learner.discover_aliases(text)
        assert len(aliases) == 2


# =====================================================================
# 6. Normalize template safety
# =====================================================================


class TestNormalizeTemplate:
    def test_basic_match_substitution(self):
        from silicon_memory.entities.rules import apply_template
        result = apply_template("{match}", match="§ 10-4", groups=[], entity_type="law")
        assert result == "§ 10-4"

    def test_group_substitution(self):
        from silicon_memory.entities.rules import apply_template
        result = apply_template("law/§{group1}", match="aml. § 10-4", groups=["10-4"], entity_type="law")
        assert result == "law/§10-4"

    def test_type_substitution(self):
        from silicon_memory.entities.rules import apply_template
        result = apply_template("{entity_type}/{match}", match="HR-2023-1", groups=[], entity_type="case_id")
        assert result == "case_id/HR-2023-1"

    def test_match_lower(self):
        from silicon_memory.entities.rules import apply_template
        result = apply_template("{match_lower}", match="PROJ-123", groups=[], entity_type="project")
        assert result == "proj-123"

    def test_match_upper(self):
        from silicon_memory.entities.rules import apply_template
        result = apply_template("{match_upper}", match="proj-123", groups=[], entity_type="project")
        assert result == "PROJ-123"

    def test_unknown_placeholder_left_as_is(self):
        from silicon_memory.entities.rules import apply_template
        result = apply_template("{unknown}", match="x", groups=[], entity_type="t")
        assert result == "{unknown}"
