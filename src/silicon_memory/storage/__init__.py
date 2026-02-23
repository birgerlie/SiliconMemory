"""Storage backends for silicon-memory."""

from silicon_memory.storage.beliefs import BeliefStore
from silicon_memory.storage.config import SiliconDBConfig
from silicon_memory.storage.decisions import DecisionStore
from silicon_memory.storage.engine import StorageLayer
from silicon_memory.storage.experiences import ExperienceStore
from silicon_memory.storage.knowledge import KnowledgeQuery
from silicon_memory.storage.observability import ObservabilityStore
from silicon_memory.storage.procedures import ProcedureStore
from silicon_memory.storage.raptor import RaptorStore
from silicon_memory.storage.reflection_tracking import ReflectionTracker

# Backwards-compat alias — callers that import SiliconDBBackend get a shim
# that creates all the stores and delegates to them.
from silicon_memory.storage.silicondb_backend import SiliconDBBackend
from silicon_memory.storage.snapshots import SnapshotStore
from silicon_memory.storage.working import WorkingStore

__all__ = [
    "SiliconDBConfig",
    "StorageLayer",
    "BeliefStore",
    "ExperienceStore",
    "ProcedureStore",
    "WorkingStore",
    "DecisionStore",
    "KnowledgeQuery",
    "ReflectionTracker",
    "RaptorStore",
    "ObservabilityStore",
    "SnapshotStore",
    "SiliconDBBackend",
]
