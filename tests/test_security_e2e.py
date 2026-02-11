"""End-to-end tests for multi-user security.

Tests verify:
- Tenant isolation
- Privacy level access control
- Forgetting operations
- Transparency (why do you know)
- Data export/import
"""

from __future__ import annotations

import os
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from silicon_memory.core.types import Belief, Experience, Procedure, Triplet
from silicon_memory.security.types import (
    DataClassification,
    PrivacyLevel,
    PrivacyMetadata,
    UserContext,
)

# Check if SiliconDB is available
SILICONDB_AVAILABLE = bool(os.environ.get("SILICONDB_LIBRARY_PATH"))


from tests.embedder_cache import MockEmbedder

_mock_embedder = MockEmbedder(dimension=768, model_name="mock-768")


def create_test_memory(db_path, user_context):
    """Create a SiliconMemory instance with mock embedder for tests."""
    from silicon_memory import SiliconMemory

    memory = SiliconMemory(db_path, user_context=user_context, auto_embedder=False)
    # Inject mock embedder into the underlying SiliconDB
    memory._backend._db._db.set_embedder(
        _mock_embedder.embed,
        dimension=_mock_embedder.dimension,
        model_name=_mock_embedder.model_name,
        warmup=True,
    )
    return memory


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_db_dir():
    """Create a temporary directory for test databases."""
    path = Path(tempfile.mkdtemp(prefix="security-e2e-"))
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def user_alice():
    """Create user context for Alice in Acme Corp."""
    return UserContext(
        user_id="alice",
        tenant_id="acme-corp",
        roles={"member"},
        default_privacy=PrivacyLevel.PRIVATE,
    )


@pytest.fixture
def user_bob():
    """Create user context for Bob in Acme Corp."""
    return UserContext(
        user_id="bob",
        tenant_id="acme-corp",
        roles={"member"},
        default_privacy=PrivacyLevel.PRIVATE,
    )


@pytest.fixture
def admin_acme():
    """Create admin context for Acme Corp."""
    return UserContext(
        user_id="admin",
        tenant_id="acme-corp",
        roles={"admin"},
        default_privacy=PrivacyLevel.PRIVATE,
    )


@pytest.fixture
def user_charlie():
    """Create user context for Charlie in Different Corp."""
    return UserContext(
        user_id="charlie",
        tenant_id="different-corp",
        roles={"member"},
        default_privacy=PrivacyLevel.PRIVATE,
    )


# ============================================================================
# Multi-User Isolation Tests
# ============================================================================


@pytest.mark.skipif(not SILICONDB_AVAILABLE, reason="SiliconDB not available")
class TestMultiUserIsolation:
    """Tests for multi-user/multi-tenant isolation."""

    @pytest.mark.asyncio
    async def test_users_cannot_see_each_others_private_data(
        self, temp_db_dir, user_alice, user_bob
    ):
        """Test that users cannot see each other's private memories."""
        from silicon_memory import SiliconMemory

        db_path = temp_db_dir / "isolation.db"

        # Alice stores a private belief
        async with create_test_memory(db_path, user_context=user_alice) as alice_memory:
            belief = Belief(
                id=uuid4(),
                content="Alice's secret information",
                confidence=0.9,
                user_id=user_alice.user_id,
                tenant_id=user_alice.tenant_id,
            )
            await alice_memory.commit_belief(belief)

        # Bob should not be able to see Alice's private belief
        async with create_test_memory(db_path, user_context=user_bob) as bob_memory:
            results = await bob_memory.query_beliefs("secret information")
            assert len(results) == 0, "Bob should not see Alice's private belief"

    @pytest.mark.asyncio
    async def test_workspace_data_visible_to_tenant_members(
        self, temp_db_dir, user_alice, user_bob
    ):
        """Test that workspace-level data is visible to tenant members."""
        from silicon_memory import SiliconMemory

        db_path = temp_db_dir / "workspace.db"

        # Alice stores a workspace-level belief
        async with create_test_memory(db_path, user_context=user_alice) as alice_memory:
            privacy = PrivacyMetadata.create_for_user(
                user_alice,
                privacy_level=PrivacyLevel.WORKSPACE,
            )
            belief = Belief(
                id=uuid4(),
                content="Shared team knowledge about Python",
                confidence=0.9,
                user_id=user_alice.user_id,
                tenant_id=user_alice.tenant_id,
                privacy=privacy,
            )
            await alice_memory.commit_belief(belief)

        # Bob should be able to see Alice's workspace belief
        async with create_test_memory(db_path, user_context=user_bob) as bob_memory:
            results = await bob_memory.query_beliefs("team knowledge Python")
            assert len(results) > 0, "Bob should see workspace belief"

    @pytest.mark.asyncio
    async def test_cross_tenant_isolation(
        self, temp_db_dir, user_alice, user_charlie
    ):
        """Test that users from different tenants cannot see each other's data."""
        from silicon_memory import SiliconMemory

        db_path = temp_db_dir / "cross_tenant.db"

        # Alice stores workspace data in Acme Corp
        async with create_test_memory(db_path, user_context=user_alice) as alice_memory:
            privacy = PrivacyMetadata.create_for_user(
                user_alice,
                privacy_level=PrivacyLevel.WORKSPACE,
            )
            belief = Belief(
                id=uuid4(),
                content="Acme Corp internal knowledge",
                confidence=0.9,
                user_id=user_alice.user_id,
                tenant_id=user_alice.tenant_id,
                privacy=privacy,
            )
            await alice_memory.commit_belief(belief)

        # Charlie from Different Corp should NOT see Acme's workspace data
        async with create_test_memory(db_path, user_context=user_charlie) as charlie_memory:
            results = await charlie_memory.query_beliefs("Acme internal")
            assert len(results) == 0, "Charlie should not see cross-tenant workspace data"

    @pytest.mark.asyncio
    async def test_public_data_visible_across_tenants(
        self, temp_db_dir, user_alice, user_charlie
    ):
        """Test that public data is visible across tenants."""
        from silicon_memory import SiliconMemory

        db_path = temp_db_dir / "public.db"

        # Alice stores public data
        async with create_test_memory(db_path, user_context=user_alice) as alice_memory:
            privacy = PrivacyMetadata.create_for_user(
                user_alice,
                privacy_level=PrivacyLevel.PUBLIC,
            )
            belief = Belief(
                id=uuid4(),
                content="Public knowledge about open source",
                confidence=0.9,
                user_id=user_alice.user_id,
                tenant_id=user_alice.tenant_id,
                privacy=privacy,
            )
            await alice_memory.commit_belief(belief)

        # Charlie from Different Corp CAN see public data
        async with create_test_memory(db_path, user_context=user_charlie) as charlie_memory:
            results = await charlie_memory.query_beliefs("open source")
            assert len(results) > 0, "Charlie should see public data"

    @pytest.mark.asyncio
    async def test_admin_can_access_all_tenant_data(
        self, temp_db_dir, user_alice, admin_acme
    ):
        """Test that admins can access all data within their tenant."""
        from silicon_memory import SiliconMemory

        db_path = temp_db_dir / "admin_access.db"

        # Alice stores private data
        async with create_test_memory(db_path, user_context=user_alice) as alice_memory:
            belief = Belief(
                id=uuid4(),
                content="Alice's very private secret",
                confidence=0.9,
                user_id=user_alice.user_id,
                tenant_id=user_alice.tenant_id,
            )
            await alice_memory.commit_belief(belief)

        # Admin CAN see Alice's private data
        async with create_test_memory(db_path, user_context=admin_acme) as admin_memory:
            results = await admin_memory.query_beliefs("very private secret")
            assert len(results) > 0, "Admin should see all tenant data"


# ============================================================================
# Explicit Sharing Tests
# ============================================================================


@pytest.mark.skipif(not SILICONDB_AVAILABLE, reason="SiliconDB not available")
class TestExplicitSharing:
    """Tests for explicit sharing functionality."""

    @pytest.mark.asyncio
    async def test_explicit_share_grants_access(
        self, temp_db_dir, user_alice, user_bob
    ):
        """Test that explicitly shared data is accessible."""
        from silicon_memory import SiliconMemory

        db_path = temp_db_dir / "sharing.db"

        # Alice stores private data and shares with Bob
        belief_id = None
        async with create_test_memory(db_path, user_context=user_alice) as alice_memory:
            belief = Belief(
                id=uuid4(),
                content="Secret shared with Bob only",
                confidence=0.9,
                user_id=user_alice.user_id,
                tenant_id=user_alice.tenant_id,
            )
            await alice_memory.commit_belief(belief)
            belief_id = str(belief.id)

            # Share with Bob
            await alice_memory.share_entity(f"belief-{belief_id}", "bob")

        # Bob should now see the shared belief
        async with create_test_memory(db_path, user_context=user_bob) as bob_memory:
            results = await bob_memory.query_beliefs("shared with Bob")
            assert len(results) > 0, "Bob should see explicitly shared belief"


# ============================================================================
# Forgetting Tests
# ============================================================================


@pytest.mark.skipif(not SILICONDB_AVAILABLE, reason="SiliconDB not available")
class TestForgetting:
    """Tests for forgetting (deletion) functionality."""

    @pytest.mark.asyncio
    async def test_forget_entity(self, temp_db_dir, user_alice):
        """Test forgetting a single entity."""
        from silicon_memory import SiliconMemory

        db_path = temp_db_dir / "forget_entity.db"

        async with create_test_memory(db_path, user_context=user_alice) as memory:
            # Store a belief
            belief = Belief(
                id=uuid4(),
                content="Information to be forgotten",
                confidence=0.9,
                user_id=user_alice.user_id,
                tenant_id=user_alice.tenant_id,
            )
            await memory.commit_belief(belief)

            # Verify it exists
            results = await memory.query_beliefs("to be forgotten")
            assert len(results) > 0

            # Forget it
            result = await memory.forget_entity(f"belief-{belief.id}")
            assert result.success
            assert result.deleted_count == 1

            # Verify it's gone
            results = await memory.query_beliefs("to be forgotten")
            assert len(results) == 0

    @pytest.mark.asyncio
    async def test_forget_session(self, temp_db_dir, user_alice):
        """Test forgetting all data from a session."""
        from silicon_memory import SiliconMemory

        db_path = temp_db_dir / "forget_session.db"

        async with create_test_memory(db_path, user_context=user_alice) as memory:
            session_id = "session-to-forget"

            # Store experiences in the session
            for i in range(3):
                exp = Experience(
                    id=uuid4(),
                    content=f"Experience {i} in session",
                    session_id=session_id,
                    user_id=user_alice.user_id,
                    tenant_id=user_alice.tenant_id,
                )
                await memory.record_experience(exp)

            # Forget the session
            result = await memory.forget_session(session_id)
            assert result.success
            assert result.deleted_count == 3


# ============================================================================
# Transparency Tests
# ============================================================================


@pytest.mark.skipif(not SILICONDB_AVAILABLE, reason="SiliconDB not available")
class TestTransparency:
    """Tests for transparency (why do you know) functionality."""

    @pytest.mark.asyncio
    async def test_why_do_you_know(self, temp_db_dir, user_alice):
        """Test getting provenance for a query."""
        from silicon_memory import SiliconMemory

        db_path = temp_db_dir / "transparency.db"

        async with create_test_memory(db_path, user_context=user_alice) as memory:
            # Store some beliefs about Python
            belief = Belief(
                id=uuid4(),
                triplet=Triplet("Python", "is", "programming language"),
                confidence=0.95,
                user_id=user_alice.user_id,
                tenant_id=user_alice.tenant_id,
            )
            await memory.commit_belief(belief)

            # Ask why we know about Python
            chains = await memory.why_do_you_know("Python")

            # Should return provenance chain
            assert len(chains) > 0
            assert chains[0].entity_type == "belief"


# ============================================================================
# Inspector Tests
# ============================================================================


@pytest.mark.skipif(not SILICONDB_AVAILABLE, reason="SiliconDB not available")
class TestInspector:
    """Tests for memory inspection functionality."""

    @pytest.mark.asyncio
    async def test_inspect_memories(self, temp_db_dir, user_alice):
        """Test inspecting memory contents."""
        from silicon_memory import SiliconMemory

        db_path = temp_db_dir / "inspect.db"

        async with create_test_memory(db_path, user_context=user_alice) as memory:
            # Store various items
            await memory.commit_belief(Belief(
                id=uuid4(),
                content="Belief 1",
                confidence=0.9,
                user_id=user_alice.user_id,
                tenant_id=user_alice.tenant_id,
            ))
            await memory.commit_belief(Belief(
                id=uuid4(),
                content="Belief 2",
                confidence=0.8,
                user_id=user_alice.user_id,
                tenant_id=user_alice.tenant_id,
            ))
            await memory.record_experience(Experience(
                id=uuid4(),
                content="Experience 1",
                user_id=user_alice.user_id,
                tenant_id=user_alice.tenant_id,
            ))

            # Inspect
            inspection = await memory.inspect_memories()

            assert inspection.user_id == "alice"
            assert inspection.tenant_id == "acme-corp"
            assert inspection.belief_count == 2
            assert inspection.experience_count == 1
            assert inspection.total_items == 3

    @pytest.mark.asyncio
    async def test_export_import_memories(self, temp_db_dir, user_alice):
        """Test exporting and importing memories."""
        from silicon_memory import SiliconMemory

        db_path1 = temp_db_dir / "export.db"
        db_path2 = temp_db_dir / "import.db"

        # Create and export memories
        records = []
        async with create_test_memory(db_path1, user_context=user_alice) as memory:
            # Store items
            await memory.commit_belief(Belief(
                id=uuid4(),
                content="Exportable belief",
                confidence=0.9,
                user_id=user_alice.user_id,
                tenant_id=user_alice.tenant_id,
            ))

            # Export
            async for record in memory.export_memories():
                records.append(record)

        assert len(records) > 0

        # Import to new database
        async with create_test_memory(db_path2, user_context=user_alice) as memory:
            result = await memory.import_memories(records)
            assert result["imported"] > 0

            # Verify imported
            results = await memory.query_beliefs("Exportable belief")
            assert len(results) > 0

    @pytest.mark.asyncio
    async def test_correct_memory(self, temp_db_dir, user_alice):
        """Test correcting a memory."""
        from silicon_memory import SiliconMemory

        db_path = temp_db_dir / "correct.db"

        async with create_test_memory(db_path, user_context=user_alice) as memory:
            # Store a belief with a typo
            belief = Belief(
                id=uuid4(),
                content="Pytohn is great",  # Typo
                confidence=0.9,
                user_id=user_alice.user_id,
                tenant_id=user_alice.tenant_id,
            )
            await memory.commit_belief(belief)

            # Correct it
            result = await memory.correct_memory(
                f"belief-{belief.id}",
                {"content": "Python is great"},  # Fixed
            )

            assert result.success
            assert "content" in result.corrections_applied


# ============================================================================
# Preferences Tests
# ============================================================================


@pytest.mark.skipif(not SILICONDB_AVAILABLE, reason="SiliconDB not available")
class TestPreferences:
    """Tests for memory preferences functionality."""

    @pytest.mark.asyncio
    async def test_blocked_topics(self, temp_db_dir, user_alice):
        """Test that blocked topics prevent storage."""
        from silicon_memory import SiliconMemory

        db_path = temp_db_dir / "preferences.db"

        async with create_test_memory(db_path, user_context=user_alice) as memory:
            # Add blocked topic
            await memory.add_blocked_topic("medical")

            # Get preferences
            prefs = await memory.get_preferences()
            assert "medical" in prefs.do_not_remember_topics

            # The application should check prefs.should_remember() before storing
            assert prefs.should_remember("My medical history") is False
            assert prefs.should_remember("My coding history") is True


# ============================================================================
# Consent Tests
# ============================================================================


@pytest.mark.skipif(not SILICONDB_AVAILABLE, reason="SiliconDB not available")
class TestConsent:
    """Tests for consent management."""

    @pytest.mark.asyncio
    async def test_consent_management(self, temp_db_dir, user_alice):
        """Test granting and revoking consent."""
        from silicon_memory import SiliconMemory

        db_path = temp_db_dir / "consent.db"

        async with create_test_memory(db_path, user_context=user_alice) as memory:
            # Store a belief
            belief = Belief(
                id=uuid4(),
                content="Test belief",
                confidence=0.9,
                user_id=user_alice.user_id,
                tenant_id=user_alice.tenant_id,
            )
            await memory.commit_belief(belief)
            entity_id = f"belief-{belief.id}"

            # Check consent (should be false initially)
            has_consent = await memory.check_consent(entity_id, "processing")
            assert has_consent is False

            # Grant consent
            await memory.grant_consent(entity_id, "processing")
            has_consent = await memory.check_consent(entity_id, "processing")
            assert has_consent is True

            # Revoke consent
            await memory.revoke_consent(entity_id, "processing")
            has_consent = await memory.check_consent(entity_id, "processing")
            assert has_consent is False
