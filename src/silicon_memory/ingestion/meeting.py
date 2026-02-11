"""Meeting transcript ingestion adapter."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING
from uuid import uuid4

from silicon_memory.core.types import Experience, Procedure
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
class TranscriptSegment:
    """A parsed segment of a meeting transcript."""

    content: str = ""
    speakers: list[str] = field(default_factory=list)
    start_time: str | None = None
    end_time: str | None = None
    topic: str | None = None
    segment_index: int = 0


@dataclass
class MeetingTranscriptConfig(IngestionConfig):
    """Configuration for meeting transcript ingestion."""

    segment_by_topic: bool = True
    time_block_minutes: int = 5
    extract_action_items: bool = True
    resolve_entities: bool = True
    auto_create_speakers: bool = True


class MeetingTranscriptAdapter:
    """Ingests meeting transcripts as experiences.

    Supports three transcript formats:
    - Timestamped: ``[HH:MM:SS] Speaker: text``
    - Speaker-labeled: ``Speaker: text``
    - Raw text (paragraphs)

    Processing pipeline:
    1. Parse transcript into lines
    2. Segment by topic (LLM) or heuristic (time blocks/speaker turns)
    3. Create experiences for each segment
    4. Optionally resolve entities (speaker names)
    5. Optionally extract action items
    """

    def __init__(
        self,
        config: MeetingTranscriptConfig | None = None,
        entity_resolver: "EntityResolver | None" = None,
    ) -> None:
        self._config = config or MeetingTranscriptConfig()
        self._entity_resolver = entity_resolver

    @property
    def source_type(self) -> str:
        return "meeting_transcript"

    async def ingest(
        self,
        content: str | bytes,
        metadata: dict[str, Any],
        memory: "SiliconMemory",
        llm_provider: "LLMProvider | None" = None,
    ) -> IngestionResult:
        """Ingest a meeting transcript into memory.

        Args:
            content: The transcript text
            metadata: Should include meeting_id, title, date, participants
            memory: SiliconMemory instance
            llm_provider: Optional LLM for enhanced segmentation

        Returns:
            IngestionResult with statistics
        """
        result = IngestionResult(source_type=self.source_type)

        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")

        content = content.strip()
        if not content:
            result.errors.append("Empty transcript content")
            return result

        meeting_id = metadata.get("meeting_id", str(uuid4()))

        # Step 1: Parse transcript into lines
        try:
            lines = self._parse_transcript_lines(content)
        except Exception as e:
            result.errors.append(f"Parse error: {e}")
            return result

        if not lines:
            result.errors.append("No parseable lines found in transcript")
            return result

        # Step 2: Segment
        if llm_provider and self._config.segment_by_topic:
            try:
                segments = await self._segment_with_llm(lines, llm_provider)
            except Exception as e:
                result.errors.append(f"LLM segmentation failed, falling back to heuristic: {e}")
                segments = self._segment_heuristic(lines)
        else:
            segments = self._segment_heuristic(lines)

        if not segments:
            result.errors.append("No segments produced from transcript")
            return result

        # Step 3: Create experiences for each segment
        user_ctx = memory.user_context

        for i, segment in enumerate(segments):
            segment.segment_index = i
            try:
                exp = Experience(
                    id=uuid4(),
                    content=segment.content,
                    context={
                        "meeting_id": meeting_id,
                        "speakers": segment.speakers,
                        "segment_index": segment.segment_index,
                        "topic": segment.topic,
                        "start_time": segment.start_time,
                        "end_time": segment.end_time,
                        "source_type": "meeting_transcript",
                        **{k: v for k, v in metadata.items() if k != "meeting_id"},
                    },
                    session_id=meeting_id,
                    user_id=user_ctx.user_id,
                    tenant_id=user_ctx.tenant_id,
                )
                await memory.record_experience(exp)
                result.experiences_created += 1
            except Exception as e:
                result.errors.append(f"Failed to store segment {i}: {e}")

        # Step 4: Resolve entities (optional)
        if self._config.resolve_entities and self._entity_resolver:
            try:
                resolved_count = await self._resolve_entities(
                    segments, memory, metadata
                )
                result.entities_resolved = resolved_count
            except Exception as e:
                result.errors.append(f"Entity resolution error: {e}")

        # Step 5: Extract action items and persist as Procedures
        if self._config.extract_action_items:
            try:
                if llm_provider:
                    action_items = await self._extract_action_items_llm(
                        segments, llm_provider
                    )
                else:
                    action_items = self._extract_action_items_heuristic(segments)
                result.action_items_detected = len(action_items)
                result.details["action_items"] = action_items

                # Persist action items as Procedure memories
                for item in action_items:
                    try:
                        procedure = Procedure(
                            id=uuid4(),
                            name=item.get("action", "")[:100],
                            description=item.get("action", ""),
                            trigger=f"From meeting {meeting_id}",
                            steps=[item.get("action", "")],
                            confidence=0.6,
                            tags={"action_item", "meeting"},
                            user_id=user_ctx.user_id,
                            tenant_id=user_ctx.tenant_id,
                        )
                        await memory.commit_procedure(procedure)
                    except Exception as e:
                        result.errors.append(f"Failed to store action item: {e}")
            except Exception as e:
                result.errors.append(f"Action item extraction error: {e}")

        # Step 6: Create graph edges (meeting → participants, meeting → actions)
        try:
            await self._create_graph_edges(memory, meeting_id, segments, result)
        except Exception as e:
            result.errors.append(f"Graph edge creation error: {e}")

        return result

    def _parse_transcript_lines(
        self,
        content: str,
    ) -> list[dict[str, Any]]:
        """Parse transcript into structured lines.

        Detects format:
        - ``[HH:MM:SS] Speaker: text``
        - ``Speaker: text``
        - Raw text (paragraphs)

        Returns:
            List of dicts with keys: content, speaker, timestamp
        """
        lines = []

        # Pattern 1: [HH:MM:SS] Speaker: text
        timestamped = re.compile(
            r'^\[?(\d{1,2}:\d{2}(?::\d{2})?)\]?\s+([^:]+):\s*(.+)$'
        )

        # Pattern 2: Speaker: text (speaker name is 1-3 words, no digits)
        speaker_only = re.compile(
            r'^([A-Z][a-zA-Z]*(?:\s[A-Z][a-zA-Z]*){0,2}):\s+(.+)$'
        )

        raw_lines = content.split("\n")
        detected_format = None

        # Detect format from first non-empty lines
        for raw_line in raw_lines[:20]:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            if timestamped.match(raw_line):
                detected_format = "timestamped"
                break
            if speaker_only.match(raw_line):
                detected_format = "speaker"
                break

        if detected_format is None:
            detected_format = "raw"

        for raw_line in raw_lines:
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            if detected_format == "timestamped":
                match = timestamped.match(raw_line)
                if match:
                    lines.append({
                        "timestamp": match.group(1),
                        "speaker": match.group(2).strip(),
                        "content": match.group(3).strip(),
                    })
                else:
                    # Continuation of previous line
                    if lines:
                        lines[-1]["content"] += " " + raw_line
                    else:
                        lines.append({
                            "timestamp": None,
                            "speaker": None,
                            "content": raw_line,
                        })
            elif detected_format == "speaker":
                match = speaker_only.match(raw_line)
                if match:
                    lines.append({
                        "timestamp": None,
                        "speaker": match.group(1).strip(),
                        "content": match.group(2).strip(),
                    })
                else:
                    # Continuation of previous line
                    if lines:
                        lines[-1]["content"] += " " + raw_line
                    else:
                        lines.append({
                            "timestamp": None,
                            "speaker": None,
                            "content": raw_line,
                        })
            else:
                # Raw text
                lines.append({
                    "timestamp": None,
                    "speaker": None,
                    "content": raw_line,
                })

        return lines

    async def _segment_with_llm(
        self,
        lines: list[dict[str, Any]],
        provider: "LLMProvider",
    ) -> list[TranscriptSegment]:
        """Use LLM to identify topic boundaries and segment transcript."""
        # Build transcript text for LLM
        transcript_text = "\n".join(
            f"{'[' + l['timestamp'] + '] ' if l.get('timestamp') else ''}"
            f"{l['speaker'] + ': ' if l.get('speaker') else ''}"
            f"{l['content']}"
            for l in lines
        )

        prompt = (
            "Segment the following meeting transcript by topic. "
            "Return a JSON array where each element has:\n"
            '- "topic": brief topic description\n'
            '- "start_line": 0-indexed start line\n'
            '- "end_line": 0-indexed end line (inclusive)\n'
            '- "speakers": list of speaker names in this segment\n'
            "\nOnly return the JSON array, no other text.\n\n"
            f"Transcript ({len(lines)} lines):\n{transcript_text}"
        )

        response = await provider.complete(
            prompt=prompt,
            system="You are a meeting transcript analyzer. Segment transcripts by topic.",
            temperature=self._config.llm_temperature,
            max_tokens=2000,
        )

        # Parse response
        segment_defs = parse_llm_json_array(response)
        segments: list[TranscriptSegment] = []

        for seg_def in segment_defs:
            start = seg_def.get("start_line", 0)
            end = seg_def.get("end_line", len(lines) - 1)
            seg_lines = lines[start:end + 1]

            content = "\n".join(
                f"{l['speaker'] + ': ' if l.get('speaker') else ''}{l['content']}"
                for l in seg_lines
            )
            speakers = seg_def.get("speakers", [])
            if not speakers:
                speakers = list({l["speaker"] for l in seg_lines if l.get("speaker")})

            start_time = seg_lines[0].get("timestamp") if seg_lines else None
            end_time = seg_lines[-1].get("timestamp") if seg_lines else None

            segments.append(TranscriptSegment(
                content=content,
                speakers=speakers,
                start_time=start_time,
                end_time=end_time,
                topic=seg_def.get("topic"),
            ))

        return segments

    def _segment_heuristic(
        self,
        lines: list[dict[str, Any]],
    ) -> list[TranscriptSegment]:
        """Segment transcript using heuristic rules.

        Strategy depends on format:
        - Timestamped: group by time blocks
        - Speaker-labeled: group by speaker turns
        - Raw: group by paragraphs (consecutive non-empty lines)
        """
        if not lines:
            return []

        has_timestamps = any(l.get("timestamp") for l in lines)
        has_speakers = any(l.get("speaker") for l in lines)

        if has_timestamps:
            return self._segment_by_time(lines)
        elif has_speakers:
            return self._segment_by_speaker(lines)
        else:
            return self._segment_by_paragraphs(lines)

    def _segment_by_time(
        self,
        lines: list[dict[str, Any]],
    ) -> list[TranscriptSegment]:
        """Group lines into time blocks."""
        block_minutes = self._config.time_block_minutes
        segments: list[TranscriptSegment] = []
        current_lines: list[dict[str, Any]] = []
        current_start: str | None = None
        block_start_minutes: float | None = None

        for line in lines:
            ts = line.get("timestamp")
            ts_minutes = self._parse_timestamp_minutes(ts) if ts else None

            if block_start_minutes is None:
                block_start_minutes = ts_minutes or 0.0
                current_start = ts

            # Check if we've passed the block boundary
            if ts_minutes is not None and (ts_minutes - block_start_minutes) >= block_minutes:
                if current_lines:
                    segments.append(self._lines_to_segment(current_lines, current_start))
                current_lines = [line]
                current_start = ts
                block_start_minutes = ts_minutes
            else:
                current_lines.append(line)

        if current_lines:
            segments.append(self._lines_to_segment(current_lines, current_start))

        return segments

    def _segment_by_speaker(
        self,
        lines: list[dict[str, Any]],
    ) -> list[TranscriptSegment]:
        """Group consecutive lines by the same speaker into segments."""
        segments: list[TranscriptSegment] = []
        current_lines: list[dict[str, Any]] = []
        current_speaker: str | None = None

        for line in lines:
            speaker = line.get("speaker")
            if speaker != current_speaker and current_lines:
                segments.append(self._lines_to_segment(current_lines))
                current_lines = []
            current_speaker = speaker
            current_lines.append(line)

        if current_lines:
            segments.append(self._lines_to_segment(current_lines))

        # Merge very short segments (< min_segment_length) with neighbors
        merged: list[TranscriptSegment] = []
        for seg in segments:
            if merged and len(seg.content) < self._config.min_segment_length:
                # Merge with previous
                merged[-1].content += "\n" + seg.content
                merged[-1].speakers = list(set(merged[-1].speakers + seg.speakers))
                merged[-1].end_time = seg.end_time
            else:
                merged.append(seg)

        return merged

    def _segment_by_paragraphs(
        self,
        lines: list[dict[str, Any]],
    ) -> list[TranscriptSegment]:
        """Group raw text lines into paragraph-based segments."""
        segments: list[TranscriptSegment] = []
        current_content: list[str] = []

        for line in lines:
            content = line["content"].strip()
            if not content:
                if current_content:
                    text = " ".join(current_content)
                    if len(text) >= self._config.min_segment_length:
                        segments.append(TranscriptSegment(content=text))
                    elif segments:
                        segments[-1].content += " " + text
                    else:
                        segments.append(TranscriptSegment(content=text))
                    current_content = []
            else:
                current_content.append(content)

        if current_content:
            text = " ".join(current_content)
            segments.append(TranscriptSegment(content=text))

        return segments

    def _lines_to_segment(
        self,
        lines: list[dict[str, Any]],
        start_time: str | None = None,
    ) -> TranscriptSegment:
        """Convert a group of parsed lines into a TranscriptSegment."""
        content = "\n".join(
            f"{l['speaker'] + ': ' if l.get('speaker') else ''}{l['content']}"
            for l in lines
        )
        speakers = list({l["speaker"] for l in lines if l.get("speaker")})
        end_time = None
        for l in reversed(lines):
            if l.get("timestamp"):
                end_time = l["timestamp"]
                break
        if not start_time:
            for l in lines:
                if l.get("timestamp"):
                    start_time = l["timestamp"]
                    break

        return TranscriptSegment(
            content=content,
            speakers=speakers,
            start_time=start_time,
            end_time=end_time,
        )

    def _parse_timestamp_minutes(self, ts: str) -> float:
        """Parse a timestamp string (HH:MM:SS or MM:SS) to minutes."""
        parts = ts.split(":")
        try:
            if len(parts) == 3:
                return int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 60.0
            elif len(parts) == 2:
                return int(parts[0]) + int(parts[1]) / 60.0
        except ValueError:
            pass
        return 0.0

    async def _extract_action_items_llm(
        self,
        segments: list[TranscriptSegment],
        provider: "LLMProvider",
    ) -> list[dict[str, Any]]:
        """Use LLM to extract action items from segments."""
        text = "\n\n".join(
            f"[Segment {s.segment_index}] {s.content}" for s in segments
        )

        prompt = (
            "Extract action items from this meeting transcript. "
            "Return a JSON array where each element has:\n"
            '- "action": what needs to be done\n'
            '- "owner": who is responsible (or null)\n'
            '- "segment_index": which segment it came from\n'
            "\nOnly return the JSON array, no other text.\n\n"
            f"{text}"
        )

        response = await provider.complete(
            prompt=prompt,
            system="You are a meeting action item extractor.",
            temperature=self._config.llm_temperature,
            max_tokens=1500,
        )

        try:
            return parse_llm_json_array(response)
        except (ValueError, json.JSONDecodeError):
            return []

    def _extract_action_items_heuristic(
        self,
        segments: list[TranscriptSegment],
    ) -> list[dict[str, Any]]:
        """Extract action items using keyword patterns."""
        action_items: list[dict[str, Any]] = []
        for segment in segments:
            action_items.extend(
                extract_action_items_from_text(
                    segment.content, "segment_index", segment.segment_index
                )
            )
        return action_items

    async def _create_graph_edges(
        self,
        memory: "SiliconMemory",
        meeting_id: str,
        segments: list[TranscriptSegment],
        result: IngestionResult,
    ) -> None:
        """Create graph edges linking meeting to participants and action items.

        Creates edges:
        - meeting → speaker (participated_in)
        - meeting → action_item (has_action)
        """
        backend = getattr(memory, "_backend", None)
        if backend is None:
            return

        db = getattr(backend, "_db", None)
        if db is None:
            return

        # Collect all speakers
        all_speakers: set[str] = set()
        for segment in segments:
            all_speakers.update(segment.speakers)

        # Create meeting → speaker edges
        for speaker in all_speakers:
            try:
                db.add_edge(
                    meeting_id,
                    speaker,
                    edge_type="participated_in",
                    metadata={"source": "meeting_transcript"},
                )
            except Exception:
                pass

        # Create meeting → action_item edges
        action_items = result.details.get("action_items", [])
        for item in action_items:
            action_text = item.get("action", "")
            if action_text:
                try:
                    db.add_edge(
                        meeting_id,
                        action_text[:80],
                        edge_type="has_action",
                        metadata={
                            "owner": item.get("owner"),
                            "source": "meeting_transcript",
                        },
                    )
                except Exception:
                    pass

    async def _resolve_entities(
        self,
        segments: list[TranscriptSegment],
        memory: "SiliconMemory",
        metadata: dict[str, Any],
    ) -> int:
        """Resolve speaker names and mentioned entities."""
        if not self._entity_resolver:
            return 0

        resolved_count = 0
        all_speakers: set[str] = set()

        for segment in segments:
            all_speakers.update(segment.speakers)

        # Resolve each speaker
        for speaker in all_speakers:
            try:
                result = await self._entity_resolver.resolve(speaker)
                if result.resolved:
                    resolved_count += len(result.resolved)
            except Exception:
                pass

        return resolved_count
