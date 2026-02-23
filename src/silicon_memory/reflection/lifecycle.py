"""LLM-based belief lifecycle manager.

Evaluates belief status transitions using LLM rather than hardcoded
thresholds. The LLM sees the full evidence picture and decides whether
a belief should transition between lifecycle states.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, TYPE_CHECKING
from uuid import UUID

from silicon_memory.core.types import Belief, BeliefStatus, TemporalContext
from silicon_memory.core.utils import utc_now

if TYPE_CHECKING:
    from silicon_memory.memory.silicondb_router import SiliconMemory

logger = logging.getLogger(__name__)

# System prompt for lifecycle evaluation
_LIFECYCLE_PROMPT = """\
You evaluate beliefs in a knowledge base and determine their lifecycle status.

Possible statuses:
- PROVISIONAL: Not yet confirmed by multiple sources
- VALIDATED: Confirmed by multiple independent sources, high confidence
- CONTESTED: Significant conflicting evidence exists
- REJECTED: Overwhelming counter-evidence, should not be trusted
- EXPIRED: Too old without reconfirmation, may be outdated

Respond with JSON only: {"status": "<STATUS>", "reason": "<brief reason>"}
"""

_BELIEF_TEMPLATE = """\
Belief: "{content}"
Current status: {status}
Confidence: {confidence:.2f}
Evidence for: {evidence_for} sources
Evidence against: {evidence_against} sources
Age: {age_days} days since observed
Last verified: {last_verified}

What status should this belief have?"""

_BATCH_TEMPLATE = """\
Evaluate each belief and determine its lifecycle status.

{beliefs_text}

Respond with JSON: {{"results": [{{"id": "<belief_id>", "status": "<STATUS>", "reason": "<reason>"}}]}}
"""

_EXPIRY_TEMPLATE = """\
This belief has exceeded its TTL (time-to-live) without reconfirmation:

Belief: "{content}"
TTL: {ttl_days} days
Last verified: {last_verified}
Age since verification: {age_days} days

Should this belief be marked EXPIRED? Consider whether the nature of this
knowledge makes it likely to still be valid despite the elapsed time.
Factual/scientific knowledge decays slowly; technical versions, prices,
or current events decay quickly.

Respond with JSON: {{"expire": true/false, "reason": "<reason>"}}
"""


class BeliefLifecycleManager:
    """LLM-based belief lifecycle evaluation.

    Uses LLM to evaluate whether beliefs should transition between
    lifecycle states based on evidence, age, and context.
    """

    def __init__(self, memory: "SiliconMemory", llm: Any) -> None:
        self._memory = memory
        self._llm = llm

    async def evaluate_belief(self, belief: Belief) -> BeliefStatus | None:
        """Use LLM to evaluate whether a belief should transition status.

        Returns the new status if a transition is warranted, or None
        if the current status is appropriate.
        """
        now = utc_now()
        age_days = 0
        last_verified = "never"

        if belief.temporal:
            age_days = (now - belief.temporal.observed_at).days
            if belief.temporal.last_verified:
                last_verified = f"{(now - belief.temporal.last_verified).days} days ago"

        prompt = _BELIEF_TEMPLATE.format(
            content=belief.content or (belief.triplet.as_text() if belief.triplet else ""),
            status=belief.status.value,
            confidence=belief.confidence,
            evidence_for=len(belief.evidence_for),
            evidence_against=len(belief.evidence_against),
            age_days=age_days,
            last_verified=last_verified,
        )

        try:
            response = await self._llm_call(_LIFECYCLE_PROMPT, prompt)
            parsed = self._parse_json(response)
            if not parsed or "status" not in parsed:
                return None

            new_status = BeliefStatus(parsed["status"].lower())
            if new_status != belief.status:
                return new_status
        except Exception as e:
            logger.debug("Lifecycle evaluation failed for %s: %s", belief.id, e)

        return None

    async def evaluate_batch(
        self, beliefs: list[Belief],
    ) -> dict[UUID, tuple[BeliefStatus, str]]:
        """Evaluate a batch of beliefs for lifecycle transitions.

        Returns dict mapping belief_id -> (new_status, reason) for
        beliefs that should transition.
        """
        if not beliefs:
            return {}

        now = utc_now()
        belief_texts = []
        for i, b in enumerate(beliefs[:20]):  # Cap batch size
            age_days = 0
            last_verified = "never"
            if b.temporal:
                age_days = (now - b.temporal.observed_at).days
                if b.temporal.last_verified:
                    last_verified = f"{(now - b.temporal.last_verified).days} days ago"

            content = b.content or (b.triplet.as_text() if b.triplet else "")
            belief_texts.append(
                f"[{i}] id={b.id}\n"
                f"  Content: {content}\n"
                f"  Status: {b.status.value}, Confidence: {b.confidence:.2f}\n"
                f"  Evidence for: {len(b.evidence_for)}, against: {len(b.evidence_against)}\n"
                f"  Age: {age_days} days, Last verified: {last_verified}"
            )

        prompt = _BATCH_TEMPLATE.format(beliefs_text="\n\n".join(belief_texts))

        transitions: dict[UUID, tuple[BeliefStatus, str]] = {}
        try:
            response = await self._llm_call(_LIFECYCLE_PROMPT, prompt)
            parsed = self._parse_json(response)
            if not parsed or "results" not in parsed:
                return transitions

            id_to_belief = {str(b.id): b for b in beliefs}
            for result in parsed["results"]:
                bid = result.get("id", "")
                if bid in id_to_belief:
                    belief = id_to_belief[bid]
                    try:
                        new_status = BeliefStatus(result["status"].lower())
                        if new_status != belief.status:
                            reason = result.get("reason", "")
                            transitions[belief.id] = (new_status, reason)
                    except (ValueError, KeyError):
                        continue
        except Exception as e:
            logger.warning("Batch lifecycle evaluation failed: %s", e)

        return transitions

    async def expire_stale_beliefs(
        self, beliefs: list[Belief], now: datetime | None = None,
    ) -> list[UUID]:
        """Check TTL expiry on beliefs. Returns IDs that expired.

        Uses TemporalContext.is_expired() for the TTL check, then
        LLM to confirm whether expiry is appropriate given the
        belief's nature.
        """
        if now is None:
            now = utc_now()

        expired_ids: list[UUID] = []
        for belief in beliefs:
            if not belief.temporal or not belief.temporal.is_expired(now):
                continue
            if belief.status in (BeliefStatus.EXPIRED, BeliefStatus.REJECTED):
                continue

            # LLM confirmation for expiry
            ttl_days = (belief.temporal.ttl_seconds or 0) / 86400
            last_verified = "never"
            age_days = 0
            if belief.temporal.last_verified:
                age_days = (now - belief.temporal.last_verified).days
                last_verified = f"{age_days} days ago"
            elif belief.temporal.observed_at:
                age_days = (now - belief.temporal.observed_at).days
                last_verified = f"{age_days} days ago (observed, never verified)"

            prompt = _EXPIRY_TEMPLATE.format(
                content=belief.content or (belief.triplet.as_text() if belief.triplet else ""),
                ttl_days=f"{ttl_days:.0f}",
                last_verified=last_verified,
                age_days=age_days,
            )

            try:
                response = await self._llm_call(_LIFECYCLE_PROMPT, prompt)
                parsed = self._parse_json(response)
                if parsed and parsed.get("expire", False):
                    expired_ids.append(belief.id)
            except Exception:
                # On LLM failure, respect the TTL check result
                expired_ids.append(belief.id)

        return expired_ids

    async def evaluate_rule_based(
        self, beliefs: list[Belief],
    ) -> dict[UUID, tuple[BeliefStatus, str]]:
        """Rule-based lifecycle transitions — fallback when LLM fails.

        Applies simple deterministic rules:
        - PROVISIONAL → VALIDATED: confidence > 0.7 AND evidence_count >= 3
        - PROVISIONAL → REJECTED: confidence < 0.2
        - Any → EXPIRED: temporal TTL exceeded
        """
        now = utc_now()
        transitions: dict[UUID, tuple[BeliefStatus, str]] = {}

        for belief in beliefs:
            if belief.status == BeliefStatus.PROVISIONAL:
                evidence_count = len(belief.evidence_for)
                if belief.confidence > 0.7 and evidence_count >= 3:
                    transitions[belief.id] = (
                        BeliefStatus.VALIDATED,
                        f"Rule: confidence {belief.confidence:.2f} > 0.7, "
                        f"{evidence_count} supporting sources",
                    )
                elif belief.confidence < 0.2:
                    transitions[belief.id] = (
                        BeliefStatus.REJECTED,
                        f"Rule: confidence {belief.confidence:.2f} < 0.2",
                    )

            # TTL expiry check for any status
            if belief.status not in (BeliefStatus.EXPIRED, BeliefStatus.REJECTED):
                if belief.temporal and belief.temporal.is_expired(now):
                    transitions[belief.id] = (
                        BeliefStatus.EXPIRED,
                        "Rule: TTL exceeded",
                    )

        return transitions

    async def _llm_call(self, system: str, user: str) -> str:
        """Make an LLM call, handling both scheduler and direct provider."""
        prompt = f"{system}\n\n{user}"
        if hasattr(self._llm, "complete"):
            return await self._llm.complete(
                prompt, system=system, temperature=0.3, max_tokens=512,
            )
        elif hasattr(self._llm, "generate"):
            return await self._llm.generate(
                prompt, max_tokens=512, temperature=0.3,
            )
        raise TypeError(f"Unknown LLM type: {type(self._llm)}")

    @staticmethod
    def _parse_json(text: str) -> dict | None:
        """Extract JSON from LLM response text."""
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Try extracting JSON from markdown code block
        for marker in ("```json", "```"):
            if marker in text:
                start = text.index(marker) + len(marker)
                end = text.index("```", start) if "```" in text[start:] else len(text)
                try:
                    return json.loads(text[start:end].strip())
                except json.JSONDecodeError:
                    pass
        # Try finding first { ... }
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start >= 0 and brace_end > brace_start:
            try:
                return json.loads(text[brace_start:brace_end + 1])
            except json.JSONDecodeError:
                pass
        return None
