"""In-memory alias → canonical entity cache."""

from __future__ import annotations

import re


class EntityCache:
    """Fast alias → canonical ID mapping.

    All lookups are normalized (lowercased, stripped, collapsed whitespace).
    Backed by plain dicts — SiliconDB persistence is handled by the resolver.
    """

    def __init__(self) -> None:
        self._alias_to_canonical: dict[str, str] = {}
        self._canonical_to_type: dict[str, str] = {}
        self._canonical_to_aliases: dict[str, set[str]] = {}

    def store(self, alias: str, canonical_id: str, entity_type: str) -> None:
        """Register an alias → canonical mapping."""
        norm = self._normalize(alias)
        self._alias_to_canonical[norm] = canonical_id
        self._canonical_to_type[canonical_id] = entity_type
        if canonical_id not in self._canonical_to_aliases:
            self._canonical_to_aliases[canonical_id] = set()
        self._canonical_to_aliases[canonical_id].add(norm)

    def lookup(self, alias: str) -> str | None:
        """Fast in-memory lookup. Returns canonical_id or None."""
        return self._alias_to_canonical.get(self._normalize(alias))

    def lookup_with_context(self, alias: str, context_canonicals: set[str]) -> str | None:
        """Resolve alias with document-level canonical context.

        Strategy:
        1. Direct alias lookup first.
        2. If unresolved, try contextual last-token matching against canonicals
           already present in the same document context.
        """
        direct = self.lookup(alias)
        if direct:
            return direct
        if not context_canonicals:
            return None

        norm_alias = self._normalize(alias)
        if not norm_alias:
            return None

        # Exact normalized canonical hit inside context.
        for canonical in context_canonicals:
            if self._normalize(canonical) == norm_alias:
                return canonical

        # Last-token fallback (e.g. "Epstein" -> "Jeffrey Epstein").
        for canonical in context_canonicals:
            parts = self._normalize(canonical).split(" ")
            if parts and parts[-1] == norm_alias:
                return canonical

        return None

    def get_type(self, canonical_id: str) -> str | None:
        """Get entity type for a canonical ID."""
        return self._canonical_to_type.get(canonical_id)

    def aliases_for(self, canonical_id: str) -> set[str]:
        """Get all known aliases for a canonical ID."""
        return set(self._canonical_to_aliases.get(canonical_id, set()))

    @property
    def size(self) -> int:
        """Number of alias → canonical mappings."""
        return len(self._alias_to_canonical)

    @staticmethod
    def _normalize(text: str) -> str:
        """Lowercase, strip, collapse whitespace."""
        return re.sub(r"\s+", " ", text.strip().lower())
