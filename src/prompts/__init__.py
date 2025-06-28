"""Prompts package for the transcription tool."""

from .base import PromptTemplate, StringPromptTemplate, validate_prompt_variables
from .config import DEFAULT_CONFIGS, MeetingType, PromptConfig, PromptStyle
from .meeting_prompts import (
    SIMPLE_MEETING_SUMMARY_PROMPT,
    SPEAKER_AWARE_MEETING_NOTES_PROMPT,
    STRUCTURED_MEETING_NOTES_PROMPT,
    create_simple_meeting_prompt,
    create_speaker_aware_meeting_prompt,
    create_structured_meeting_prompt,
)
from .specialized import (
    CLIENT_CALL_PROMPT,
    INTERVIEW_MEETING_PROMPT,
    PLANNING_MEETING_PROMPT,
    RETROSPECTIVE_MEETING_PROMPT,
    STANDUP_MEETING_PROMPT,
    get_specialized_prompt,
)

__all__ = [
    # Meeting prompts
    "STRUCTURED_MEETING_NOTES_PROMPT",
    "SIMPLE_MEETING_SUMMARY_PROMPT",
    "SPEAKER_AWARE_MEETING_NOTES_PROMPT",
    "create_structured_meeting_prompt",
    "create_simple_meeting_prompt",
    "create_speaker_aware_meeting_prompt",
    # Specialized prompts
    "STANDUP_MEETING_PROMPT",
    "CLIENT_CALL_PROMPT",
    "PLANNING_MEETING_PROMPT",
    "INTERVIEW_MEETING_PROMPT",
    "RETROSPECTIVE_MEETING_PROMPT",
    "get_specialized_prompt",
    # Base classes
    "PromptTemplate",
    "StringPromptTemplate",
    "validate_prompt_variables",
    # Configuration
    "PromptConfig",
    "PromptStyle",
    "MeetingType",
    "DEFAULT_CONFIGS",
]
