"""File-based embedder cache for testing.

Caches embeddings to disk to avoid re-embedding the same text during tests.
This significantly speeds up test runs when using real embedders.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, List


# Default cache directory (in tests directory)
DEFAULT_CACHE_DIR = Path(__file__).parent / ".embedding_cache"


class FileCachedEmbedder:
    """Wrapper that caches embeddings to disk for faster test runs.

    Works with any embedding function that follows the signature:
        embed(texts: List[str], is_query: bool = False) -> List[List[float]]

    Example:
        >>> from silicondb.embedders import E5Embedder
        >>> embedder = E5Embedder("small")
        >>> cached = FileCachedEmbedder(
        ...     embed_fn=embedder.embed,
        ...     dimension=embedder.dimension,
        ...     model_name=embedder.model_name,
        ...     cache_name="e5_small",
        ... )
        >>> # First call computes and caches
        >>> emb1 = cached.embed(["hello world"])
        >>> # Second call loads from cache
        >>> emb2 = cached.embed(["hello world"])
    """

    def __init__(
        self,
        embed_fn: Callable[[List[str], bool], List[List[float]]],
        dimension: int,
        model_name: str,
        cache_name: str,
        cache_dir: Path | None = None,
        embed_numpy_fn: Callable | None = None,
    ) -> None:
        """Initialize the cached embedder.

        Args:
            embed_fn: The underlying embedding function.
            dimension: Embedding dimension.
            model_name: Name of the model (for identification).
            cache_name: Name for the cache file (e.g., "e5_small").
            cache_dir: Directory for cache files. Defaults to .embedding_cache.
            embed_numpy_fn: Optional numpy embedding function for SiliconDB.
        """
        self._embed_fn = embed_fn
        self._embed_numpy_fn = embed_numpy_fn
        self._dimension = dimension
        self._model_name = model_name

        self._cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self._cache_file = self._cache_dir / f"{cache_name}_embeddings.json"
        self._cache = self._load_cache()

        # Stats
        self._hits = 0
        self._misses = 0

    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        return self._dimension

    @property
    def model_name(self) -> str:
        """Model name."""
        return self._model_name

    def _load_cache(self) -> dict:
        """Load cache from disk if exists."""
        if self._cache_file.exists():
            try:
                with open(self._cache_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_cache(self) -> None:
        """Save cache to disk."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        with open(self._cache_file, "w") as f:
            json.dump(self._cache, f)

    def _hash_text(self, text: str, is_query: bool) -> str:
        """Create hash key for text."""
        key = f"{self._model_name}|{text}|query={is_query}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def embed(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """Embed texts with caching.

        Args:
            texts: List of texts to embed.
            is_query: Whether these are query texts (affects prefix).

        Returns:
            List of embedding vectors.
        """
        results: List[List[float] | None] = []
        uncached_texts: List[str] = []
        uncached_indices: List[int] = []

        # Check cache for each text
        for i, text in enumerate(texts):
            key = self._hash_text(text, is_query)
            if key in self._cache:
                self._hits += 1
                results.append(self._cache[key])
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)
                results.append(None)  # Placeholder

        # Embed uncached texts in batch
        if uncached_texts:
            self._misses += len(uncached_texts)
            embeddings = self._embed_fn(uncached_texts, is_query)
            for idx, text, emb in zip(uncached_indices, uncached_texts, embeddings):
                key = self._hash_text(text, is_query)
                self._cache[key] = emb
                results[idx] = emb

        return results  # type: ignore

    def embed_numpy(self, texts: List[str], is_query: bool = False):
        """Embed texts and return numpy array.

        Delegates to underlying embedder's embed_numpy if available.
        """
        if self._embed_numpy_fn:
            return self._embed_numpy_fn(texts, is_query)
        # Fallback: convert list to numpy
        import numpy as np
        return np.array(self.embed(texts, is_query), dtype=np.float32)

    def save(self) -> None:
        """Save cache to disk. Call at end of session."""
        if self._misses > 0:  # Only save if we computed new embeddings
            self._save_cache()

    def stats(self) -> str:
        """Return cache hit/miss stats."""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        return f"Cache: {self._hits}/{total} hits ({hit_rate:.0f}%)"


class MockEmbedder:
    """Deterministic mock embedder for testing without real models.

    Uses hashing to produce consistent embeddings for the same text,
    allowing tests to verify search works correctly without loading
    heavy ML models.

    Example:
        >>> embedder = MockEmbedder(dimension=384)
        >>> emb1 = embedder.embed(["hello"])
        >>> emb2 = embedder.embed(["hello"])
        >>> assert emb1 == emb2  # Same text = same embedding
    """

    def __init__(self, dimension: int = 384, model_name: str = "mock") -> None:
        """Initialize mock embedder.

        Args:
            dimension: Embedding dimension.
            model_name: Name for the mock model.
        """
        self._dimension = dimension
        self._model_name = model_name

    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        return self._dimension

    @property
    def model_name(self) -> str:
        """Model name."""
        return self._model_name

    def embed(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """Generate deterministic embeddings based on text hash.

        Args:
            texts: List of texts to embed.
            is_query: Whether these are query texts.

        Returns:
            List of normalized embedding vectors.
        """
        import random

        results = []
        for text in texts:
            # Include is_query in seed for different query vs passage embeddings
            prefix = "query:" if is_query else "passage:"
            seed = int(hashlib.sha256((prefix + text).encode()).hexdigest()[:8], 16)
            rng = random.Random(seed)

            # Generate normalized random embedding
            emb = [rng.gauss(0, 1) for _ in range(self._dimension)]
            norm = sum(x * x for x in emb) ** 0.5
            emb = [x / norm for x in emb]
            results.append(emb)

        return results

    def embed_numpy(self, texts: List[str], is_query: bool = False):
        """Embed texts and return numpy array."""
        import numpy as np
        return np.array(self.embed(texts, is_query), dtype=np.float32)
