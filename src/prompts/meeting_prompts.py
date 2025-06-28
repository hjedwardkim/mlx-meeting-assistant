"""Meeting-specific prompts for transcription and summarization with speaker awareness."""

# Template for structured meeting notes with speaker attribution
SPEAKER_AWARE_MEETING_NOTES_PROMPT = """You are an expert meeting note-taker analyzing a speaker-attributed transcript. The transcript includes speaker labels (e.g., **Speaker_0**, **Speaker_1**) with timestamps.

MEETING TRANSCRIPT WITH SPEAKERS:
{content}

INSTRUCTIONS:
Please organize this speaker-attributed meeting transcript into structured notes. Pay attention to who said what and attribute decisions, action items, and key points to specific speakers when possible.

# Meeting Notes

## Summary
[Provide a high-level overview of the meeting covering main purpose, outcomes, and key participants]

## Participants
[List the speakers identified in the transcript. If speakers can be identified by name from context, include both speaker ID and name]
- Speaker_0: [Name if determinable from context, otherwise "Unidentified"]
- Speaker_1: [Name if determinable from context, otherwise "Unidentified"]

## Decisions Made
[List concrete decisions with attribution to speakers who made or supported them]
- [Decision] - Proposed/Decided by [Speaker_X]
- [Decision] - Agreed upon by [Speaker_Y and Speaker_Z]

## Action Items
[List specific tasks with speaker attribution for responsibility]
- [Task description] - Assigned to [Speaker_X] - [Due date if mentioned]
- [Task description] - Volunteered by [Speaker_Y] - [Due date if mentioned]

## Key Discussions by Topic
[Organize main topics with speaker contributions]

### [Topic 1]
- **Speaker_X**: [Main points or position]  
- **Speaker_Y**: [Response or different perspective]
- **Resolution**: [How the discussion concluded]

### [Topic 2]
- **Speaker_X**: [Main points or position]
- **Speaker_Y**: [Response or different perspective]  
- **Resolution**: [How the discussion concluded]

## Open Questions
[List unresolved issues with context of who raised them]
- [Question] - Raised by [Speaker_X]
- [Question] - Discussed by [Speaker_Y and Speaker_Z]

## Follow-up Items
[List next steps with speaker attribution]
- [Follow-up item] - [Speaker_X] to handle
- [Meeting/call] - [Speaker_Y] to schedule

## Speaking Time Analysis
[If notable patterns emerge about participation]
- Most active participant: [Speaker_X]
- Discussion leaders: [Speaker_Y, Speaker_Z]
- Areas needing more input: [Topics where certain speakers were quiet]

## Additional Notes
[Any other important observations about speaker dynamics, agreements, or concerns]

---

IMPORTANT:
- Maintain speaker attribution throughout the notes
- Look for patterns in who drives decisions vs. who provides input
- Note when speakers agree, disagree, or build on each other's points
- If speaker identity can be inferred from context (names mentioned), include this information
- Focus on actionable and concrete information with clear ownership
"""


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


def create_speaker_aware_meeting_prompt(content: str) -> str:
    """
    Create prompt for speaker-attributed meeting analysis.

    Args:
        content: Speaker-attributed meeting transcript content

    Returns:
        Formatted prompt for the LLM
    """
    return SPEAKER_AWARE_MEETING_NOTES_PROMPT.format(content=content)
