"""Meeting-specific prompts for transcription and summarization."""

# Template for structured meeting notes
STRUCTURED_MEETING_NOTES_PROMPT = """You are an expert meeting note-taker. Analyze the following meeting transcript and extract key information into a structured format. Be thorough but concise.

MEETING TRANSCRIPT:
{content}

INSTRUCTIONS:
Please organize the information from this meeting into the following structured format. Use markdown formatting and be specific and actionable where possible:

# Meeting Notes

## Summary
[Provide a high-level overview of the meeting in 2-3 sentences covering the main purpose and outcomes]

## Decisions Made
[List concrete decisions that were reached during the meeting. If no clear decisions were made, write "No explicit decisions were made during this meeting."]
- [Decision 1]
- [Decision 2]

## Action Items
[List specific tasks or actions that need to be completed. Include responsible parties when mentioned. If no action items, write "No specific action items were identified."]
- [Task description] - [Responsible party if mentioned, otherwise "TBD"]
- [Task description] - [Responsible party if mentioned, otherwise "TBD"]

## Open Questions
[List unresolved issues, questions that need answers, or topics requiring further discussion. If none, write "No open questions identified."]
- [Question 1]
- [Question 2]

## Follow-up Items
[List next steps, future meetings, or items that need scheduling. If none, write "No follow-up items specified."]
- [Follow-up item 1]
- [Follow-up item 2]

## Key Topics Discussed
[List the main themes, subjects, or areas of focus during the meeting]
- [Topic 1]: [Brief description]
- [Topic 2]: [Brief description]

## Participants
[List participants mentioned in the meeting. If not clearly identifiable, write "Participants not clearly identified in transcript."]
- [Participant 1]
- [Participant 2]

## Additional Notes
[Any other important information that doesn't fit in the above categories]

---

IMPORTANT: 
- Be factual and only include information explicitly discussed in the transcript
- Do not make assumptions about information not present
- Use bullet points for clarity
- If a section has no relevant information, explicitly state so rather than omitting the section
- Focus on actionable and concrete information

Please provide the structured meeting notes now:"""


# Template for simple meeting summary
SIMPLE_MEETING_SUMMARY_PROMPT = """Please provide a concise summary of the following text of a meeting:

{content}

Summary:"""


def create_structured_meeting_prompt(content: str) -> str:
    """
    Create a comprehensive prompt for structured meeting note extraction.

    Args:
        content: Meeting transcript content

    Returns:
        Formatted prompt for the LLM
    """
    return STRUCTURED_MEETING_NOTES_PROMPT.format(content=content)


def create_simple_meeting_prompt(content: str) -> str:
    """
    Create a simple prompt for basic meeting summarization.

    Args:
        content: Meeting transcript content

    Returns:
        Formatted prompt for the LLM
    """
    return SIMPLE_MEETING_SUMMARY_PROMPT.format(content=content)
