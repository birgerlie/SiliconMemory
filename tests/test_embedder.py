"""Test that SiliconDB built-in embedders load and work with default config.

Requires:
  - SILICONDB_LIBRARY_PATH env var
  - sentence-transformers installed
"""

from __future__ import annotations

import os
import shutil
import tempfile

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("SILICONDB_LIBRARY_PATH"),
    reason="SILICONDB_LIBRARY_PATH not set",
)


@pytest.fixture
def temp_db_path():
    path = tempfile.mkdtemp(prefix="silicon-embedder-test-")
    yield path
    shutil.rmtree(path, ignore_errors=True)


class TestEmbedderDefaults:
    """Verify that SiliconDB initializes an embedder with default values."""

    def test_default_config_has_embedder(self, temp_db_path):
        """SiliconDB() with no embedder args should auto-load E5-base."""
        from silicondb import SiliconDB

        db = SiliconDB(temp_db_path)
        try:
            assert db.has_embedder is True
        finally:
            db.close()

    def test_default_embed_produces_vectors(self, temp_db_path):
        """Default embedder should produce 768-dim vectors (E5-base)."""
        from silicondb import SiliconDB

        db = SiliconDB(temp_db_path)
        try:
            vec = db.embed("hello world")
            assert isinstance(vec, list)
            assert len(vec) == 768
            assert all(isinstance(v, float) for v in vec)
        finally:
            db.close()

    def test_default_embed_batch(self, temp_db_path):
        """Batch embedding should work with defaults."""
        from silicondb import SiliconDB

        db = SiliconDB(temp_db_path)
        try:
            vecs = db.embed_batch(["alpha", "bravo", "charlie"])
            assert len(vecs) == 3
            assert all(len(v) == 768 for v in vecs)
        finally:
            db.close()

    def test_query_vs_passage_differ(self, temp_db_path):
        """E5 prefixes differ for query vs passage, so embeddings should differ."""
        from silicondb import SiliconDB

        db = SiliconDB(temp_db_path)
        try:
            q = db.embed("machine learning", is_query=True)
            p = db.embed("machine learning", is_query=False)
            assert q != p
        finally:
            db.close()


class TestEmbedderInsertAndSearch:
    """Verify insert_text + search_text round-trip with default embedder."""

    @pytest.fixture
    def db(self, temp_db_path):
        from silicondb import SiliconDB

        db = SiliconDB(temp_db_path)
        yield db
        db.close()

    def test_insert_and_retrieve(self, db):
        """Inserted text should be retrievable by ID."""
        db.insert_text("doc-1", "Transformers revolutionized NLP.")
        doc = db.get("doc-1")
        assert doc is not None
        assert doc["text"] == "Transformers revolutionized NLP."

    def test_semantic_search_ranks_relevant_higher(self, db):
        """Semantically related docs should rank above unrelated ones."""
        db.insert_text("ai-doc", "Neural networks learn representations from data.")
        db.insert_text("cook-doc", "Chop the onions and sauté in olive oil.")

        results = db.search_text("deep learning", k=2)
        assert len(results) >= 1
        assert results[0].external_id == "ai-doc"

    def test_search_multiple_relevant(self, db):
        """Multiple related documents should all appear in results."""
        db.insert_text("d1", "Python is a popular programming language.")
        db.insert_text("d2", "JavaScript runs in the browser.")
        db.insert_text("d3", "Rust provides memory safety without garbage collection.")
        db.insert_text("d4", "Banana bread recipe with walnuts.")

        results = db.search_text("software development", k=4)
        top_ids = {r.external_id for r in results[:3]}
        # Programming docs should dominate the top 3
        assert len(top_ids & {"d1", "d2", "d3"}) >= 2
        # Cooking doc should not be #1
        assert results[0].external_id != "d4"


class TestEmbedderPersistence:
    """Verify embeddings survive close/reopen."""

    def test_search_after_reopen(self, temp_db_path):
        from silicondb import SiliconDB

        db1 = SiliconDB(temp_db_path)
        db1.insert_text("doc-1", "Gradient descent optimizes neural networks.")
        db1.insert_text("doc-2", "Sourdough bread needs a long fermentation.")
        db1.close()

        db2 = SiliconDB(temp_db_path)
        try:
            assert db2.document_count == 2
            results = db2.search_text("optimization algorithms", k=2)
            assert len(results) == 2
            assert results[0].external_id == "doc-1"
        finally:
            db2.close()
