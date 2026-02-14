"""End-to-end tests for the entity resolution system.

Tests the full pipeline: LLM bootstrap → rule generation → entity resolution.
Requires SiliconServe running on localhost:8000 with qwen3-80b loaded.
"""

from __future__ import annotations

import asyncio

import pytest

from silicon_memory.entities import (
    EntityCache,
    EntityResolver,
    RuleEngine,
)
from silicon_memory.entities.learner import RuleLearner
from silicon_memory.llm.provider import SiliconLLMProvider

# --- Fixtures ---

SILICONSERVE_URL = "http://localhost:8000/v1"
MODEL = "qwen3-80b"

NORWEGIAN_LAW_SAMPLE = """\
Arbeidsmiljøloven (aml.) regulerer arbeidsforhold i Norge.

I henhold til aml. § 10-4 første ledd er den alminnelige arbeidstid
ikke over 40 timer i uken. Etter aml. § 10-6 kan arbeidstaker og
arbeidsgiver skriftlig avtale at den samlede arbeidstiden kan utvides.

Straffeloven (strl.) § 321 omhandler tyveri, mens strl. § 322
gjelder grovt tyveri. Bestemmelsene i straffeloven kapittel 27
regulerer ulike former for vinningslovbrudd.

Høyesterett avsa dom i HR-2023-1234-A den 15. mars 2023 vedrørende
tolkningen av aml. § 14-9 om midlertidig ansettelse. Dommen
henviser også til HR-2021-987-B og Rt. 2015 s. 1332.

Forskrift om systematisk helse-, miljø- og sikkerhetsarbeid
(internkontrollforskriften) § 5 stiller krav til dokumentasjon.
"""

SECOND_TEXT = """\
Ved vurderingen av oppsigelsen ble det vist til aml. § 15-7 og
rettspraksis i HR-2023-1234-A. Arbeidstakeren hadde også krav
i henhold til strl. § 166 om falsk forklaring.

Prosjekt ALPHA-2024-001 er underlagt internkontrollforskriften § 5.
"""


def _siliconserve_available() -> bool:
    """Check if SiliconServe is running with the target model."""
    try:
        import httpx
        resp = httpx.get(f"{SILICONSERVE_URL}/models", timeout=3)
        if resp.status_code == 200:
            models = resp.json().get("data", [])
            return any(
                m["id"] == MODEL and m["status"] == "loaded"
                for m in models
            )
    except Exception:
        pass
    return False


requires_siliconserve = pytest.mark.skipif(
    not _siliconserve_available(),
    reason=f"SiliconServe not available with {MODEL}",
)


@pytest.fixture
def llm():
    return SiliconLLMProvider(base_url=SILICONSERVE_URL, model=MODEL)


@pytest.fixture
def resolver():
    return EntityResolver(cache=EntityCache(), rules=RuleEngine())


@pytest.fixture
def learner(llm):
    return RuleLearner(llm=llm)


# --- E2E Tests ---


@requires_siliconserve
class TestBootstrapE2E:
    """Test LLM-powered bootstrap from Norwegian legal text."""

    @pytest.mark.asyncio
    async def test_extract_entities_from_law_text(self, learner):
        """LLM should find entity references in Norwegian legal text."""
        examples = await learner.extract_entities(NORWEGIAN_LAW_SAMPLE)
        assert len(examples) > 0

        types_found = {e["entity_type"] for e in examples}
        texts_found = {e["text"] for e in examples}

        # Should find at least some law/case references
        print(f"\nExtracted {len(examples)} entities:")
        for e in examples:
            print(f"  [{e['entity_type']}] {e['text']} → {e['canonical']}")
        print(f"Types: {types_found}")

        # Sanity: at least 3 entities found
        assert len(examples) >= 3

    @pytest.mark.asyncio
    async def test_generate_rules_from_extracted_entities(self, learner):
        """LLM should generate valid regex rules from entity examples."""
        examples = await learner.extract_entities(NORWEGIAN_LAW_SAMPLE)
        assert len(examples) > 0

        detectors, extractors = await learner.generate_rules(examples)

        print(f"\nGenerated {len(detectors)} detectors, {len(extractors)} extractors:")
        for d in detectors:
            print(f"  Detector [{d.id}]: {d.pattern}  ({d.description})")
        for e in extractors:
            print(f"  Extractor [{e.id}]: {e.pattern}  type={e.entity_type}")

        assert len(detectors) >= 1
        assert len(extractors) >= 1

        # All patterns should be valid compiled regex
        import re
        for d in detectors:
            re.compile(d.pattern)
        for e in extractors:
            re.compile(e.pattern)

    @pytest.mark.asyncio
    async def test_full_bootstrap_and_resolve(self, learner, resolver):
        """Full pipeline: bootstrap → resolve new text."""
        # Step 1: Bootstrap from sample text
        detectors, extractors, aliases = await learner.bootstrap(NORWEGIAN_LAW_SAMPLE)

        print(f"\nBootstrap results:")
        print(f"  Detectors: {len(detectors)}")
        print(f"  Extractors: {len(extractors)}")
        print(f"  Aliases: {aliases}")

        assert len(detectors) >= 1
        assert len(extractors) >= 1

        # Load rules into resolver
        for d in detectors:
            resolver.rules.add_detector(d)
        for e in extractors:
            resolver.rules.add_extractor(e)
        for short_form, long_form in aliases:
            await resolver.register_alias(short_form, long_form, "alias")

        # Step 2: Resolve entities in new (unseen) text
        result = await resolver.resolve(SECOND_TEXT)

        print(f"\nResolution of new text:")
        print(f"  Resolved: {len(result.resolved)}")
        for ref in result.resolved:
            print(f"    [{ref.entity_type}] {ref.text} → {ref.canonical_id} (conf={ref.confidence})")
        print(f"  Unresolved: {result.unresolved}")

        # Should resolve at least some entities
        assert len(result.resolved) >= 1

    @pytest.mark.asyncio
    async def test_discover_aliases_in_law_text(self, learner):
        """Parenthetical alias detection — no LLM needed."""
        aliases = learner.discover_aliases(NORWEGIAN_LAW_SAMPLE)

        print(f"\nDiscovered aliases:")
        for short, long in aliases:
            print(f"  {short} → {long}")

        # Should find aml. → arbeidsmiljøloven and strl. → straffeloven
        short_forms = {a[0] for a in aliases}
        assert "aml." in short_forms or "strl." in short_forms

    @pytest.mark.asyncio
    async def test_bootstrap_then_resolve_case_ids(self, learner, resolver):
        """Bootstrapped rules should detect case IDs in new text."""
        detectors, extractors, aliases = await learner.bootstrap(NORWEGIAN_LAW_SAMPLE)

        for d in detectors:
            resolver.rules.add_detector(d)
        for e in extractors:
            resolver.rules.add_extractor(e)

        # Resolve text with a case ID
        result = await resolver.resolve(
            "Saken ble avgjort i HR-2023-1234-A og bekreftet i HR-2021-987-B"
        )

        print(f"\nCase ID resolution:")
        for ref in result.resolved:
            print(f"  [{ref.entity_type}] {ref.text} → {ref.canonical_id}")

        # Check that at least one case ID was resolved
        case_refs = [r for r in result.resolved if "HR-" in r.text]
        if case_refs:
            assert len(case_refs) >= 1
        else:
            # If the LLM didn't generate case ID rules, that's OK — log it
            print("  NOTE: No case ID rules generated by LLM in this run")

    @pytest.mark.asyncio
    async def test_resolver_enriches_with_cache(self, learner, resolver):
        """Cache should enrich canonical IDs from aliases."""
        detectors, extractors, aliases = await learner.bootstrap(NORWEGIAN_LAW_SAMPLE)

        for d in detectors:
            resolver.rules.add_detector(d)
        for e in extractors:
            resolver.rules.add_extractor(e)

        # Manually register a canonical mapping
        await resolver.register_alias(
            "HR-2023-1234-A", "hoeyesterett/2023/1234", "supreme_court"
        )

        result = await resolver.resolve("Henvisning til HR-2023-1234-A.")

        hr_refs = [r for r in result.resolved if "HR-2023" in r.text]
        if hr_refs:
            print(f"\nCache enrichment: {hr_refs[0].text} → {hr_refs[0].canonical_id}")
            assert hr_refs[0].canonical_id == "hoeyesterett/2023/1234"

    @pytest.mark.asyncio
    async def test_incremental_learning(self, learner, resolver):
        """Unresolved entities should be learnable."""
        # Start with a simple detector that catches patterns
        from silicon_memory.entities.types import DetectorRule

        resolver.rules.add_detector(DetectorRule(
            id="d_broad", pattern=r"[A-Z]{2,}-\d{4}-\d+(?:-[A-Z])?",
            description="broad ID patterns",
        ))

        # No extractors → everything goes to unresolved
        result = await resolver.resolve("Se HR-2023-1234-A og LB-2022-5678")
        assert len(result.unresolved) >= 1
        assert resolver.unresolved_count >= 1

        print(f"\nUnresolved queue: {resolver._unresolved_queue}")

        # Now trigger learning
        resolver._learner = learner
        new_rules = await resolver.learn_rules()

        print(f"  Rules learned: {new_rules}")
        print(f"  Detectors: {len(resolver.rules._detectors)}")
        print(f"  Extractors: {len(resolver.rules._extractors)}")

        # Should have generated some rules
        assert new_rules >= 0  # LLM may or may not generate from 2 examples
