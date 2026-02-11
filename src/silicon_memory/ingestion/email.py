"""Email ingestion adapter."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING
from uuid import uuid4

from silicon_memory.core.types import Experience
from silicon_memory.ingestion.types import IngestionConfig, IngestionResult
from silicon_memory.ingestion._helpers import (
    extract_action_items_from_text,
    parse_llm_json_array,
)

if TYPE_CHECKING:
    from silicon_memory.memory.silicondb_router import SiliconMemory
    from silicon_memory.reflection.llm import LLMProvider
    from silicon_memory.entities.resolver import EntityResolver


@dataclass
class EmailMessage:
    """A parsed email message."""

    from_addr: str = ""
    to_addrs: list[str] = field(default_factory=list)
    cc_addrs: list[str] = field(default_factory=list)
    subject: str = ""
    body: str = ""
    date: str | None = None
    message_id: str = ""
    in_reply_to: str | None = None
    thread_id: str | None = None
    attachments: list[str] = field(default_factory=list)


@dataclass
class EmailConfig(IngestionConfig):
    """Configuration for email ingestion."""

    extract_action_items: bool = True
    resolve_entities: bool = True
    parse_threads: bool = True
    max_thread_depth: int = 20


class EmailAdapter:
    """Ingests emails and email threads as experiences.

    Supports two input formats:
    - Raw RFC 2822 text (standard email format)
    - Pre-parsed dict with email fields

    Processing pipeline:
    1. Parse input (auto-detect raw vs dict)
    2. Split thread if it's a chain (replies/forwards)
    3. Create one Experience per email message
    4. Extract action items (LLM or heuristic)
    5. Optionally resolve entities (sender/recipient names)
    """

    def __init__(
        self,
        config: EmailConfig | None = None,
        entity_resolver: "EntityResolver | None" = None,
    ) -> None:
        self._config = config or EmailConfig()
        self._entity_resolver = entity_resolver

    @property
    def source_type(self) -> str:
        return "email"

    async def ingest(
        self,
        content: str | bytes,
        metadata: dict[str, Any],
        memory: "SiliconMemory",
        llm_provider: "LLMProvider | None" = None,
    ) -> IngestionResult:
        """Ingest an email or email thread into memory.

        Args:
            content: Raw RFC 2822 text, JSON string, or pre-parsed dict
            metadata: Should include email_id or thread context
            memory: SiliconMemory instance
            llm_provider: Optional LLM for enhanced extraction

        Returns:
            IngestionResult with statistics
        """
        result = IngestionResult(source_type=self.source_type)

        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")

        if isinstance(content, str):
            content = content.strip()

        if not content:
            result.errors.append("Empty email content")
            return result

        # Step 1+2: Detect format and parse into messages
        try:
            messages = self._detect_and_parse(content)
        except Exception as e:
            result.errors.append(f"Parse error: {e}")
            return result

        if not messages:
            result.errors.append("No parseable email messages found")
            return result

        # Limit thread depth
        messages = messages[: self._config.max_thread_depth]

        # Step 3: Create experiences for each message
        user_ctx = memory.user_context
        thread_id = metadata.get("thread_id") or messages[0].thread_id or str(uuid4())

        for i, msg in enumerate(messages):
            try:
                exp = Experience(
                    id=uuid4(),
                    content=msg.body,
                    context={
                        "message_id": msg.message_id or f"{thread_id}-{i}",
                        "thread_id": thread_id,
                        "from": msg.from_addr,
                        "to": msg.to_addrs,
                        "cc": msg.cc_addrs,
                        "subject": msg.subject,
                        "date": msg.date,
                        "source_type": "email",
                        **{k: v for k, v in metadata.items() if k not in ("thread_id",)},
                    },
                    session_id=thread_id,
                    user_id=user_ctx.user_id,
                    tenant_id=user_ctx.tenant_id,
                )
                await memory.record_experience(exp)
                result.experiences_created += 1
            except Exception as e:
                result.errors.append(f"Failed to store message {i}: {e}")

        # Step 4: Extract action items (optional)
        if self._config.extract_action_items:
            try:
                if llm_provider:
                    action_items = await self._extract_action_items_llm(
                        messages, llm_provider
                    )
                else:
                    action_items = self._extract_action_items_heuristic(messages)
                result.action_items_detected = len(action_items)
                result.details["action_items"] = action_items
            except Exception as e:
                result.errors.append(f"Action item extraction error: {e}")

        # Step 5: Resolve entities (optional)
        if self._config.resolve_entities and self._entity_resolver:
            try:
                resolved_count = await self._resolve_entities(messages, memory)
                result.entities_resolved = resolved_count
            except Exception as e:
                result.errors.append(f"Entity resolution error: {e}")

        return result

    def _parse_raw_email(self, content: str) -> EmailMessage:
        """Parse an RFC 2822 email from raw text.

        Extracts headers (From, To, CC, Subject, Date, Message-ID,
        In-Reply-To) and body. Strips HTML tags if body is HTML.
        """
        # Split headers from body at first blank line
        header_section, _, body = content.partition("\n\n")
        if not body:
            # Try \r\n\r\n
            header_section, _, body = content.partition("\r\n\r\n")

        headers: dict[str, str] = {}
        current_key = ""
        for line in header_section.split("\n"):
            line = line.rstrip("\r")
            if line.startswith((" ", "\t")) and current_key:
                # Continuation of previous header
                headers[current_key] += " " + line.strip()
            elif ":" in line:
                key, _, value = line.partition(":")
                current_key = key.strip()
                headers[current_key] = value.strip()

        # Parse address lists
        to_addrs = [
            a.strip()
            for a in headers.get("To", "").split(",")
            if a.strip()
        ]
        cc_addrs = [
            a.strip()
            for a in headers.get("Cc", headers.get("CC", "")).split(",")
            if a.strip()
        ]

        # Strip HTML tags from body if present
        body = body.strip()
        if "<html" in body.lower() or "<body" in body.lower() or "<div" in body.lower():
            body = re.sub(r"<[^>]+>", "", body)
            body = re.sub(r"\s+", " ", body).strip()

        # Parse date
        date_str = headers.get("Date", None)

        return EmailMessage(
            from_addr=headers.get("From", ""),
            to_addrs=to_addrs,
            cc_addrs=cc_addrs,
            subject=headers.get("Subject", ""),
            body=body,
            date=date_str,
            message_id=headers.get("Message-ID", headers.get("Message-Id", "")),
            in_reply_to=headers.get("In-Reply-To", headers.get("In-Reply-to", None)),
            thread_id=headers.get("Thread-ID", headers.get("Thread-Id", None)),
        )

    def _parse_dict_email(self, data: dict) -> EmailMessage:
        """Map a pre-parsed dict to EmailMessage."""
        to_addrs = data.get("to_addrs", data.get("to", []))
        if isinstance(to_addrs, str):
            to_addrs = [a.strip() for a in to_addrs.split(",") if a.strip()]

        cc_addrs = data.get("cc_addrs", data.get("cc", []))
        if isinstance(cc_addrs, str):
            cc_addrs = [a.strip() for a in cc_addrs.split(",") if a.strip()]

        attachments = data.get("attachments", [])
        if isinstance(attachments, str):
            attachments = [attachments]

        return EmailMessage(
            from_addr=data.get("from_addr", data.get("from", "")),
            to_addrs=to_addrs,
            cc_addrs=cc_addrs,
            subject=data.get("subject", ""),
            body=data.get("body", ""),
            date=data.get("date", None),
            message_id=data.get("message_id", ""),
            in_reply_to=data.get("in_reply_to", None),
            thread_id=data.get("thread_id", None),
            attachments=attachments,
        )

    def _detect_and_parse(self, content: str | dict) -> list[EmailMessage]:
        """Auto-detect format and parse into list of EmailMessages.

        Handles:
        - dict input -> parse as pre-parsed email
        - JSON string -> parse as dict
        - Raw RFC 2822 text -> parse headers + body, then split thread
        """
        if isinstance(content, dict):
            return [self._parse_dict_email(content)]

        content_str: str = content

        # Try JSON
        try:
            data = json.loads(content_str)
            if isinstance(data, dict):
                return [self._parse_dict_email(data)]
            if isinstance(data, list):
                return [self._parse_dict_email(d) for d in data if isinstance(d, dict)]
        except (json.JSONDecodeError, ValueError):
            pass

        # Check for RFC 2822 headers
        has_headers = bool(
            re.match(r"^(From|To|Subject|Date|Message-ID):", content_str, re.MULTILINE | re.IGNORECASE)
        )

        if has_headers:
            msg = self._parse_raw_email(content_str)
            # Try splitting thread from body
            if self._config.parse_threads:
                thread_parts = self._split_thread(msg.body)
                if len(thread_parts) > 1:
                    messages = []
                    # First part is the latest reply
                    messages.append(EmailMessage(
                        from_addr=msg.from_addr,
                        to_addrs=msg.to_addrs,
                        cc_addrs=msg.cc_addrs,
                        subject=msg.subject,
                        body=thread_parts[0]["body"],
                        date=msg.date,
                        message_id=msg.message_id,
                        in_reply_to=msg.in_reply_to,
                        thread_id=msg.thread_id,
                    ))
                    # Older messages from thread
                    for part in thread_parts[1:]:
                        messages.append(EmailMessage(
                            from_addr=part.get("author", ""),
                            subject=msg.subject,
                            body=part["body"],
                            date=part.get("date", None),
                            thread_id=msg.thread_id,
                        ))
                    return messages
            return [msg]

        # Fallback: treat as plain body text
        return [EmailMessage(body=content_str)]

    def _split_thread(self, body: str) -> list[dict[str, Any]]:
        """Detect reply chains in email body.

        Looks for patterns:
        - "On <date>, <name> wrote:" reply headers
        - ">" quoted lines
        - "---------- Forwarded message ----------" markers

        Returns list of {author, date, body} dicts, newest first.
        """
        # Pattern for "On ... wrote:" reply markers
        reply_pattern = re.compile(
            r"^On\s+(.+?),?\s+(.+?)\s+wrote:\s*$",
            re.MULTILINE,
        )

        # Pattern for forwarded message markers
        forward_pattern = re.compile(
            r"^-{5,}\s*Forwarded message\s*-{5,}\s*$",
            re.MULTILINE | re.IGNORECASE,
        )

        parts: list[dict[str, Any]] = []

        # Try splitting on "On ... wrote:" patterns
        splits = list(reply_pattern.finditer(body))
        if splits:
            # Text before first reply marker is the newest message
            first_body = body[: splits[0].start()].strip()
            if first_body:
                parts.append({"author": "", "date": None, "body": first_body})

            for i, match in enumerate(splits):
                start = match.end()
                end = splits[i + 1].start() if i + 1 < len(splits) else len(body)
                reply_body = body[start:end].strip()
                # Strip leading ">" quoting
                reply_body = self._strip_quoting(reply_body)
                if reply_body:
                    parts.append({
                        "author": match.group(2).strip(),
                        "date": match.group(1).strip(),
                        "body": reply_body,
                    })

            return parts if parts else [{"author": "", "date": None, "body": body}]

        # Try splitting on forwarded message markers
        fwd_splits = list(forward_pattern.finditer(body))
        if fwd_splits:
            first_body = body[: fwd_splits[0].start()].strip()
            if first_body:
                parts.append({"author": "", "date": None, "body": first_body})

            for i, match in enumerate(fwd_splits):
                start = match.end()
                end = fwd_splits[i + 1].start() if i + 1 < len(fwd_splits) else len(body)
                fwd_body = body[start:end].strip()
                if fwd_body:
                    parts.append({
                        "author": "",
                        "date": None,
                        "body": fwd_body,
                    })

            return parts if parts else [{"author": "", "date": None, "body": body}]

        # No thread detected - single message
        return [{"author": "", "date": None, "body": body}]

    def _strip_quoting(self, text: str) -> str:
        """Strip '>' quoting from reply text."""
        lines = []
        for line in text.split("\n"):
            stripped = line.lstrip()
            while stripped.startswith(">"):
                stripped = stripped[1:].lstrip()
            lines.append(stripped)
        return "\n".join(lines).strip()

    # Email-specific patterns prepended before the shared defaults.
    _EXTRA_PATTERNS: list[re.Pattern[str]] = [
        # "Please [verb]", "Can you [verb]"
        re.compile(r"(?:please|can you|could you)\s+(.+?)(?:\.|$)", re.IGNORECASE),
        # "By [date]" or "deadline" patterns
        re.compile(r"(?:by|before|deadline[:\s])\s*(.+?)(?:\.|$)", re.IGNORECASE),
    ]

    def _extract_action_items_heuristic(
        self,
        messages: list[EmailMessage],
    ) -> list[dict[str, Any]]:
        """Extract action items using keyword patterns."""
        action_items: list[dict[str, Any]] = []
        for i, msg in enumerate(messages):
            action_items.extend(
                extract_action_items_from_text(
                    msg.body, "message_index", i,
                    extra_patterns=self._EXTRA_PATTERNS,
                )
            )
        return action_items

    async def _extract_action_items_llm(
        self,
        messages: list[EmailMessage],
        provider: "LLMProvider",
    ) -> list[dict[str, Any]]:
        """Use LLM to extract action items from email messages."""
        text = "\n\n".join(
            f"[Message {i}] From: {m.from_addr}\nSubject: {m.subject}\n{m.body}"
            for i, m in enumerate(messages)
        )

        prompt = (
            "Extract action items from this email thread. "
            "Return a JSON array where each element has:\n"
            '- "action": what needs to be done\n'
            '- "owner": who is responsible (or null)\n'
            '- "message_index": which message it came from\n'
            "\nOnly return the JSON array, no other text.\n\n"
            f"{text}"
        )

        response = await provider.complete(
            prompt=prompt,
            system="You are an email action item extractor.",
            temperature=self._config.llm_temperature,
            max_tokens=1500,
        )

        try:
            return parse_llm_json_array(response)
        except (ValueError, json.JSONDecodeError):
            return []

    async def _resolve_entities(
        self,
        messages: list[EmailMessage],
        memory: "SiliconMemory",
    ) -> int:
        """Resolve sender/recipient names via EntityResolver."""
        if not self._entity_resolver:
            return 0

        resolved_count = 0
        all_names: set[str] = set()

        for msg in messages:
            if msg.from_addr:
                all_names.add(msg.from_addr)
            all_names.update(msg.to_addrs)
            all_names.update(msg.cc_addrs)

        for name in all_names:
            try:
                result = await self._entity_resolver.resolve(name)
                if result.resolved:
                    resolved_count += len(result.resolved)
            except Exception:
                pass

        return resolved_count
