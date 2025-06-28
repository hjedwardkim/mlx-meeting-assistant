"""Configuration for prompts and prompt behavior."""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class PromptStyle(Enum):
    """Available prompt styles for different use cases."""

    STRUCTURED = "structured"
    SIMPLE = "simple"
    DETAILED = "detailed"
    CONCISE = "concise"


class MeetingType(Enum):
    """Different types of meetings that may require specialized prompts."""

    GENERAL = "general"
    STANDUP = "standup"
    PLANNING = "planning"
    RETROSPECTIVE = "retrospective"
    CLIENT_CALL = "client_call"
    INTERVIEW = "interview"


@dataclass
class PromptConfig:
    """Configuration for prompt generation."""

    style: PromptStyle = PromptStyle.STRUCTURED
    meeting_type: MeetingType = MeetingType.GENERAL
    max_tokens: int = 10000
    include_participants: bool = True
    include_timestamps: bool = False
    focus_on_decisions: bool = True
    focus_on_action_items: bool = True
    custom_instructions: Optional[str] = None


# Default configurations for different scenarios
DEFAULT_CONFIGS = {
    "structured": PromptConfig(
        style=PromptStyle.STRUCTURED,
        meeting_type=MeetingType.GENERAL,
        max_tokens=10000,
    ),
    "simple": PromptConfig(
        style=PromptStyle.SIMPLE,
        meeting_type=MeetingType.GENERAL,
        max_tokens=5000,
        include_participants=False,
    ),
    "standup": PromptConfig(
        style=PromptStyle.STRUCTURED,
        meeting_type=MeetingType.STANDUP,
        max_tokens=3000,
        focus_on_decisions=False,
    ),
    "client_call": PromptConfig(
        style=PromptStyle.DETAILED,
        meeting_type=MeetingType.CLIENT_CALL,
        max_tokens=15000,
        include_timestamps=True,
    ),
}
