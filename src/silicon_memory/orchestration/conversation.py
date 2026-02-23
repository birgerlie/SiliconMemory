"""Event and signal driven conversation scheduler."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from silicon_memory.core.utils import utc_now
from silicon_memory.reflection.engine import ReflectionEngine
from silicon_memory.reflection.types import ReflectionConfig

logger = logging.getLogger(__name__)


class SchedulerEventType(str, Enum):
    """Events that can trigger orchestration decisions."""

    TURN_COMPLETED = "turn_completed"
    INGEST_COMPLETED = "ingest_completed"
    CONVERSATION_ENDED = "conversation_ended"
    BACKPRESSURE_HIGH = "backpressure_high"
    ERROR_SPIKE = "error_spike"
    TOKEN_BUDGET_HIGH = "token_budget_high"


class SchedulerAction(str, Enum):
    """Scheduler actions."""

    SNAPSHOT = "snapshot"
    REFLECT = "reflect"
    DREAM = "dream"


@dataclass
class SchedulerEvent:
    """Event payload passed into policy evaluation."""

    type: SchedulerEventType
    session_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp_iso: str = field(default_factory=lambda: utc_now().isoformat())


@dataclass
class SchedulerSignals:
    """Runtime signals used by policy decisions."""

    session_id: str
    turn_count: int = 0
    seconds_since_reflect: float = 1e9
    seconds_since_dream: float = 1e9
    llm_queue_depth: int = 0
    llm_error_rate: float = 0.0
    daily_token_spend_usd: float = 0.0
    backpressure: float = 0.0
    new_extracted_count: int = 0


@dataclass
class SchedulerDecision:
    """Decision output from policy evaluation."""

    actions: list[SchedulerAction] = field(default_factory=list)
    reason: str = ""
    blocked: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "actions": [a.value for a in self.actions],
            "reason": self.reason,
            "blocked": self.blocked,
        }


@dataclass
class ConversationSchedulerConfig:
    """Configuration for event/signal scheduling."""

    # Active reflection
    enable_active_reflection: bool = True
    active_reflect_every_turns: int = 4
    active_reflect_max_experiences: int = 24
    active_reflect_min_interval_seconds: int = 120

    # End-of-conversation
    run_end_snapshot: bool = True
    run_end_reflection: bool = True
    end_reflect_max_experiences: int = 200
    run_end_dream: bool = False
    end_dream_min_turns: int = 12
    # Fast/stable dream defaults for active systems.
    end_dream_enable_hypothesis_generation: bool = False
    end_dream_enable_hypothesis_validation: bool = False
    end_dream_max_hypothesis_validations: int = 50
    end_dream_enable_procedure_detection: bool = True
    end_dream_enable_question_generation: bool = True
    end_dream_enable_entity_consolidation: bool = False
    end_dream_enable_predicate_consolidation: bool = False

    # Ingest-triggered reflection
    ingest_reflect_threshold: int = 25

    # Guardrails from runtime signals
    max_llm_queue_depth: int = 50
    max_llm_error_rate: float = 0.25
    max_daily_token_spend_usd: float = 30.0
    max_backpressure: float = 0.85

    # Working memory persistence
    active_state_ttl_seconds: int = 1800
    context_prefix: str = "conversation"


@dataclass
class ConversationTickResult:
    """Outcome for active turn processing."""

    session_id: str
    turn_count: int
    reflected: bool = False
    reflection_experiences_processed: int = 0
    reflection_error: str | None = None
    decision: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConversationEndResult:
    """Outcome for conversation finalization."""

    session_id: str
    turn_count: int
    task_context: str
    snapshot_created: bool = False
    snapshot_id: str | None = None
    summary: str | None = None
    reflected: bool = False
    reflection_experiences_processed: int = 0
    dreamed: bool = False
    dream_stats: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    decision: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EventOutcome:
    """Generic result for emitted events."""

    event: SchedulerEvent
    signals: SchedulerSignals
    decision: SchedulerDecision
    result: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": {
                "type": self.event.type.value,
                "session_id": self.event.session_id,
                "payload": self.event.payload,
                "timestamp_iso": self.event.timestamp_iso,
            },
            "signals": asdict(self.signals),
            "decision": self.decision.to_dict(),
            "result": self.result,
            "errors": list(self.errors),
        }


class DefaultConversationPolicy:
    """Default policy mapping events + signals to scheduler actions."""

    def decide(
        self,
        event: SchedulerEvent,
        signals: SchedulerSignals,
        config: ConversationSchedulerConfig,
    ) -> SchedulerDecision:
        overloaded = (
            signals.backpressure >= config.max_backpressure
            or signals.llm_queue_depth >= config.max_llm_queue_depth
        )
        unstable = signals.llm_error_rate >= config.max_llm_error_rate
        expensive = signals.daily_token_spend_usd >= config.max_daily_token_spend_usd

        if event.type in (
            SchedulerEventType.BACKPRESSURE_HIGH,
            SchedulerEventType.ERROR_SPIKE,
            SchedulerEventType.TOKEN_BUDGET_HIGH,
        ):
            return SchedulerDecision(
                actions=[],
                reason=f"suppressed by event={event.type.value}",
                blocked=True,
            )

        if event.type == SchedulerEventType.TURN_COMPLETED:
            if not config.enable_active_reflection:
                return SchedulerDecision(reason="active reflection disabled")
            if overloaded or unstable or expensive:
                return SchedulerDecision(reason="guardrail blocked active reflection", blocked=True)
            cadence = max(1, config.active_reflect_every_turns)
            if signals.turn_count % cadence != 0:
                return SchedulerDecision(reason="turn cadence not reached")
            if signals.seconds_since_reflect < max(0, config.active_reflect_min_interval_seconds):
                return SchedulerDecision(reason="min reflect interval not reached")
            return SchedulerDecision(actions=[SchedulerAction.REFLECT], reason="cadence + signals")

        if event.type == SchedulerEventType.INGEST_COMPLETED:
            if overloaded or unstable:
                return SchedulerDecision(reason="guardrail blocked ingest-triggered reflection", blocked=True)
            if signals.new_extracted_count >= max(1, config.ingest_reflect_threshold):
                return SchedulerDecision(
                    actions=[SchedulerAction.REFLECT],
                    reason="ingest threshold reached",
                )
            return SchedulerDecision(reason="ingest threshold not reached")

        if event.type == SchedulerEventType.CONVERSATION_ENDED:
            actions: list[SchedulerAction] = []
            if config.run_end_snapshot:
                actions.append(SchedulerAction.SNAPSHOT)
            if config.run_end_reflection and not (overloaded or unstable):
                actions.append(SchedulerAction.REFLECT)
            if (
                config.run_end_dream
                and signals.turn_count >= config.end_dream_min_turns
                and not (overloaded or unstable or expensive)
            ):
                actions.append(SchedulerAction.DREAM)
            return SchedulerDecision(actions=actions, reason="conversation ended")

        return SchedulerDecision(reason=f"no policy for event={event.type.value}")


class ConversationScheduler:
    """Event and signal driven scheduler for chat memory operations."""

    def __init__(
        self,
        memory: Any,
        llm: Any | None = None,
        config: ConversationSchedulerConfig | None = None,
        policy: DefaultConversationPolicy | None = None,
    ) -> None:
        self._memory = memory
        self._llm = llm
        self._config = config or ConversationSchedulerConfig()
        self._policy = policy or DefaultConversationPolicy()
        self._turn_counts: dict[str, int] = {}
        self._last_reflect_at: dict[str, Any] = {}
        self._last_dream_at: dict[str, Any] = {}

    def _state_key(self, session_id: str) -> str:
        return f"{self._config.context_prefix}:{session_id}:state"

    def _summary_key(self, session_id: str) -> str:
        return f"{self._config.context_prefix}:{session_id}:summary"

    def _seconds_since(self, ts: Any) -> float:
        if ts is None:
            return 1e9
        try:
            return max(0.0, (utc_now() - ts).total_seconds())
        except Exception:
            return 1e9

    def _signals_for(self, session_id: str, payload: dict[str, Any]) -> SchedulerSignals:
        return SchedulerSignals(
            session_id=session_id,
            turn_count=self._turn_counts.get(session_id, 0),
            seconds_since_reflect=self._seconds_since(self._last_reflect_at.get(session_id)),
            seconds_since_dream=self._seconds_since(self._last_dream_at.get(session_id)),
            llm_queue_depth=int(payload.get("llm_queue_depth", 0) or 0),
            llm_error_rate=float(payload.get("llm_error_rate", 0.0) or 0.0),
            daily_token_spend_usd=float(payload.get("daily_token_spend_usd", 0.0) or 0.0),
            backpressure=float(payload.get("backpressure", 0.0) or 0.0),
            new_extracted_count=int(payload.get("new_extracted_count", 0) or 0),
        )

    async def _run_reflect(self, max_experiences: int) -> tuple[int, str | None]:
        if self._llm is None:
            return 0, "llm_missing"
        try:
            config = ReflectionConfig(
                max_experiences_per_batch=max_experiences,
                auto_commit_beliefs=True,
            )
            engine = ReflectionEngine(self._memory, self._llm, config=config)
            reflected = await engine.reflect(max_experiences=max_experiences, auto_commit=True)
            return reflected.experiences_processed, None
        except Exception as exc:  # pragma: no cover
            logger.warning("Scheduled reflection failed: %s", exc)
            return 0, str(exc)

    async def _run_dream(self) -> tuple[dict[str, Any], str | None]:
        if self._llm is None:
            return {}, "llm_missing"
        try:
            config = ReflectionConfig(
                auto_commit_beliefs=True,
                dream_enable_hypothesis_generation=self._config.end_dream_enable_hypothesis_generation,
                dream_enable_hypothesis_validation=self._config.end_dream_enable_hypothesis_validation,
                dream_max_hypothesis_validations=self._config.end_dream_max_hypothesis_validations,
                dream_enable_procedure_detection=self._config.end_dream_enable_procedure_detection,
                dream_enable_question_generation=self._config.end_dream_enable_question_generation,
                dream_enable_entity_consolidation=self._config.end_dream_enable_entity_consolidation,
                dream_enable_predicate_consolidation=self._config.end_dream_enable_predicate_consolidation,
            )
            engine = ReflectionEngine(self._memory, self._llm, config=config)
            return await engine.dream(), None
        except Exception as exc:  # pragma: no cover
            logger.warning("Scheduled dream failed: %s", exc)
            return {}, str(exc)

    async def _run_snapshot(
        self,
        session_id: str,
        task_context: str | None,
    ) -> tuple[dict[str, Any], str | None]:
        context = task_context or f"{self._config.context_prefix}:{session_id}"
        try:
            snapshot = await self._memory.create_snapshot(context, llm_provider=self._llm)
            summary_doc = {
                "snapshot_id": str(snapshot.id),
                "task_context": context,
                "summary": snapshot.summary,
                "next_steps": snapshot.next_steps,
                "open_questions": snapshot.open_questions,
            }
            await self._memory.set_context(
                self._summary_key(session_id),
                summary_doc,
                ttl_seconds=self._config.active_state_ttl_seconds,
            )
            return summary_doc, None
        except Exception as exc:  # pragma: no cover
            logger.warning("Scheduled snapshot failed: %s", exc)
            return {}, str(exc)

    async def emit_event(
        self,
        event_type: SchedulerEventType | str,
        session_id: str,
        payload: dict[str, Any] | None = None,
    ) -> EventOutcome:
        """Evaluate one event against current signals and execute actions."""
        payload = payload or {}
        if isinstance(event_type, str):
            event_type = SchedulerEventType(event_type)
        event = SchedulerEvent(type=event_type, session_id=session_id, payload=payload)
        signals = self._signals_for(session_id, payload)
        decision = self._policy.decide(event, signals, self._config)
        outcome = EventOutcome(event=event, signals=signals, decision=decision)

        if SchedulerAction.SNAPSHOT in decision.actions:
            snapshot_doc, err = await self._run_snapshot(
                session_id=session_id,
                task_context=payload.get("task_context"),
            )
            if err:
                outcome.errors.append(f"snapshot_failed:{err}")
            else:
                outcome.result["snapshot"] = snapshot_doc

        reflect_max = int(payload.get("reflect_max_experiences", 0) or 0)
        if reflect_max <= 0:
            if event.type == SchedulerEventType.TURN_COMPLETED:
                reflect_max = self._config.active_reflect_max_experiences
            else:
                reflect_max = self._config.end_reflect_max_experiences
        if SchedulerAction.REFLECT in decision.actions:
            processed, err = await self._run_reflect(reflect_max)
            if err:
                outcome.errors.append(f"reflect_failed:{err}")
            else:
                outcome.result["reflection_experiences_processed"] = processed
                self._last_reflect_at[session_id] = utc_now()

        if SchedulerAction.DREAM in decision.actions:
            stats, err = await self._run_dream()
            if err:
                outcome.errors.append(f"dream_failed:{err}")
            else:
                outcome.result["dream_stats"] = stats
                self._last_dream_at[session_id] = utc_now()

        return outcome

    async def on_interaction(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        signals: dict[str, Any] | None = None,
    ) -> ConversationTickResult:
        """Process a chat turn and execute policy-driven actions."""
        turn_count = self._turn_counts.get(session_id, 0) + 1
        self._turn_counts[session_id] = turn_count
        await self._memory.set_context(
            self._state_key(session_id),
            {
                "session_id": session_id,
                "turn_count": turn_count,
                "updated_at": utc_now().isoformat(),
                "last_user_message": user_message[:300],
                "last_assistant_message": assistant_message[:300],
            },
            ttl_seconds=self._config.active_state_ttl_seconds,
        )
        outcome = await self.emit_event(
            SchedulerEventType.TURN_COMPLETED,
            session_id=session_id,
            payload=signals or {},
        )
        tick = ConversationTickResult(
            session_id=session_id,
            turn_count=turn_count,
            decision=outcome.decision.to_dict(),
        )
        if "reflection_experiences_processed" in outcome.result:
            tick.reflected = True
            tick.reflection_experiences_processed = int(
                outcome.result["reflection_experiences_processed"]
            )
        if outcome.errors:
            tick.reflection_error = ";".join(outcome.errors)
        return tick

    async def on_conversation_end(
        self,
        session_id: str,
        task_context: str | None = None,
        run_dream: bool | None = None,
        signals: dict[str, Any] | None = None,
    ) -> ConversationEndResult:
        """Finalize conversation via event-driven policy execution."""
        turns = self._turn_counts.get(session_id, 0)
        effective_context = task_context or f"{self._config.context_prefix}:{session_id}"
        payload = dict(signals or {})
        payload["task_context"] = effective_context

        original_run_end_dream = self._config.run_end_dream
        if run_dream is not None:
            self._config.run_end_dream = run_dream
        try:
            outcome = await self.emit_event(
                SchedulerEventType.CONVERSATION_ENDED,
                session_id=session_id,
                payload=payload,
            )
        finally:
            self._config.run_end_dream = original_run_end_dream

        end = ConversationEndResult(
            session_id=session_id,
            turn_count=turns,
            task_context=effective_context,
            errors=list(outcome.errors),
            decision=outcome.decision.to_dict(),
        )
        snapshot_doc = outcome.result.get("snapshot")
        if snapshot_doc:
            end.snapshot_created = True
            end.snapshot_id = snapshot_doc.get("snapshot_id")
            end.summary = snapshot_doc.get("summary")
        if "reflection_experiences_processed" in outcome.result:
            end.reflected = True
            end.reflection_experiences_processed = int(
                outcome.result["reflection_experiences_processed"]
            )
        if "dream_stats" in outcome.result:
            end.dreamed = True
            end.dream_stats = outcome.result.get("dream_stats", {})

        # Reset per-session counters after closure.
        self._turn_counts.pop(session_id, None)
        self._last_reflect_at.pop(session_id, None)
        self._last_dream_at.pop(session_id, None)
        return end
