"""Two-tier regex rule engine: detectors (broad) + extractors (precise)."""

from __future__ import annotations

import re
from typing import Any

from silicon_memory.entities.types import (
    Candidate,
    DetectorRule,
    EntityReference,
    ExtractorRule,
)

CONTEXT_WINDOW = 50  # chars before/after match for context


def apply_template(
    template: str,
    *,
    match: str,
    groups: list[str],
    entity_type: str,
) -> str:
    """Apply a safe normalization template.

    Supported placeholders:
      {match}        — full match text
      {match_lower}  — full match lowercased
      {match_upper}  — full match uppercased
      {group1}, {group2}, ... — capture groups
      {entity_type}  — the entity type string
    """
    replacements: dict[str, str] = {
        "match": match,
        "match_lower": match.lower(),
        "match_upper": match.upper(),
        "entity_type": entity_type,
    }
    for i, g in enumerate(groups, 1):
        replacements[f"group{i}"] = g

    def _replace(m: re.Match) -> str:
        key = m.group(1)
        return replacements.get(key, m.group(0))

    return re.sub(r"\{(\w+)\}", _replace, template)


class RuleEngine:
    """Two-tier regex engine: detectors (broad) + extractors (precise).

    Detectors run against full text to find candidates (pass 1).
    Extractors run per-candidate to determine entity type and normalize (pass 2).
    """

    def __init__(self) -> None:
        self._detectors: list[DetectorRule] = []
        self._extractors: list[ExtractorRule] = []
        self._compiled_detectors: list[tuple[DetectorRule, re.Pattern[str]]] = []
        self._compiled_extractors: list[tuple[ExtractorRule, re.Pattern[str]]] = []

    def add_detector(self, rule: DetectorRule) -> None:
        """Add a detector rule. Raises re.error if pattern is invalid."""
        compiled = re.compile(rule.pattern)
        self._detectors.append(rule)
        self._compiled_detectors.append((rule, compiled))

    def add_extractor(self, rule: ExtractorRule) -> None:
        """Add an extractor rule. Raises re.error if pattern is invalid."""
        compiled = re.compile(rule.pattern)
        self._extractors.append(rule)
        self._compiled_extractors.append((rule, compiled))

    def get_extractor(self, rule_id: str) -> ExtractorRule | None:
        """Get an extractor rule by ID."""
        for rule in self._extractors:
            if rule.id == rule_id:
                return rule
        return None

    def detect(self, text: str) -> list[Candidate]:
        """Pass 1: Run all detector patterns against full text.

        Returns deduplicated candidates with context windows.
        """
        if not text:
            return []

        seen_spans: dict[tuple[int, int], Candidate] = {}

        for rule, pattern in self._compiled_detectors:
            for m in pattern.finditer(text):
                span = (m.start(), m.end())
                if self._overlaps_any(span, seen_spans):
                    continue

                ctx_start = max(0, m.start() - CONTEXT_WINDOW)
                ctx_end = min(len(text), m.end() + CONTEXT_WINDOW)
                context = text[ctx_start:ctx_end]

                seen_spans[span] = Candidate(
                    text=m.group(0),
                    span=span,
                    context_text=context,
                    detector_id=rule.id,
                )

        return sorted(seen_spans.values(), key=lambda c: c.span[0])

    def extract(self, candidates: list[Candidate]) -> dict[int, list[EntityReference]]:
        """Pass 2: Run extractor patterns per candidate.

        Returns mapping of candidate index → list of matching EntityReferences.
        Multiple matches per candidate = ambiguity (resolved in pass 3).
        Empty dict entries are omitted.
        """
        results: dict[int, list[EntityReference]] = {}

        for idx, candidate in enumerate(candidates):
            matches: list[EntityReference] = []
            for rule, pattern in self._compiled_extractors:
                m = pattern.search(candidate.text)
                if m:
                    groups = list(m.groups())
                    canonical = apply_template(
                        rule.normalize_template,
                        match=m.group(0),
                        groups=groups[1:] if len(groups) > 1 else groups,
                        entity_type=rule.entity_type,
                    )
                    matches.append(EntityReference(
                        text=candidate.text,
                        canonical_id=canonical,
                        entity_type=rule.entity_type,
                        confidence=rule.confidence,
                        span=candidate.span,
                        context_text=candidate.context_text,
                        rule_id=rule.id,
                    ))
            if matches:
                results[idx] = matches

        return results

    @staticmethod
    def _overlaps_any(
        span: tuple[int, int], seen: dict[tuple[int, int], Any]
    ) -> bool:
        """Check if span overlaps with any already-seen span."""
        for existing in seen:
            if span[0] < existing[1] and span[1] > existing[0]:
                return True
        return False
