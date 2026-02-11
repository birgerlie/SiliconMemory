"""Passive ingestion adapters for Silicon Memory."""

from silicon_memory.ingestion.types import (
    IngestionAdapter,
    IngestionConfig,
    IngestionResult,
)
from silicon_memory.ingestion.meeting import (
    MeetingTranscriptAdapter,
    MeetingTranscriptConfig,
    TranscriptSegment,
)
from silicon_memory.ingestion.email import (
    EmailAdapter,
    EmailConfig,
    EmailMessage,
)
from silicon_memory.ingestion.document import (
    DocumentAdapter,
    DocumentConfig,
    DocumentSection,
)
from silicon_memory.ingestion.news import (
    NewsArticleAdapter,
    NewsArticle,
    NewsIngestionConfig,
)
from silicon_memory.ingestion.chat import (
    BaseChatAdapter,
    ChatConfig,
    ChatMessage,
    ChatThread,
)
from silicon_memory.ingestion.slack import (
    SlackAdapter,
    SlackConfig,
)
from silicon_memory.ingestion.teams import (
    TeamsAdapter,
    TeamsConfig,
)
from silicon_memory.ingestion.discord import (
    DiscordAdapter,
    DiscordConfig,
)

__all__ = [
    "IngestionAdapter",
    "IngestionConfig",
    "IngestionResult",
    "MeetingTranscriptAdapter",
    "MeetingTranscriptConfig",
    "TranscriptSegment",
    "EmailAdapter",
    "EmailConfig",
    "EmailMessage",
    "DocumentAdapter",
    "DocumentConfig",
    "DocumentSection",
    "NewsArticleAdapter",
    "NewsArticle",
    "NewsIngestionConfig",
    "BaseChatAdapter",
    "ChatConfig",
    "ChatMessage",
    "ChatThread",
    "SlackAdapter",
    "SlackConfig",
    "TeamsAdapter",
    "TeamsConfig",
    "DiscordAdapter",
    "DiscordConfig",
]
