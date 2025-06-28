"""Specialized prompts for different types of meetings."""

from .config import MeetingType


# Standup meeting prompt
STANDUP_MEETING_PROMPT = """You are an expert at summarizing standup meetings. Analyze the following standup transcript and extract key information.

STANDUP TRANSCRIPT:
{content}

Please organize the information into this format:

# Daily Standup Notes

## Date & Participants
- Date: [If mentioned, otherwise "Not specified"]
- Participants: [List participants or "Not clearly identified"]

## What Was Done Yesterday
- [Person]: [What they accomplished]
- [Person]: [What they accomplished]

## What's Planned for Today  
- [Person]: [What they plan to do]
- [Person]: [What they plan to do]

## Blockers & Issues
- [Blocker 1]: [Description and who it affects]
- [Blocker 2]: [Description and who it affects]

## Team Updates
- [Any team-wide announcements or updates]

## Follow-up Actions
- [Any actions that came out of the standup]

Focus on individual contributions and blockers. Keep it concise and actionable."""


# Client call prompt
CLIENT_CALL_PROMPT = """You are an expert at summarizing client calls and business meetings. Analyze the following client call transcript.

CLIENT CALL TRANSCRIPT:
{content}

Please organize the information into this professional format:

# Client Call Summary

## Call Details
- Date: [If mentioned]
- Participants: [Client contacts and internal team members]
- Duration: [If mentioned]

## Executive Summary
[2-3 sentences covering the main purpose and key outcomes of the call]

## Client Requirements & Needs
- [Requirement 1]: [Description]
- [Requirement 2]: [Description]

## Decisions & Agreements
- [Decision 1]
- [Decision 2]

## Action Items
- [Action item] - [Owner] - [Due date if mentioned]
- [Action item] - [Owner] - [Due date if mentioned]

## Client Concerns & Questions
- [Concern 1]: [Description and resolution if provided]
- [Concern 2]: [Description and resolution if provided]

## Next Steps
- [Next step 1]
- [Next step 2]

## Commercial Discussion
- [Any pricing, contract, or commercial items discussed]

## Technical Discussion
- [Any technical requirements or solutions discussed]

## Follow-up Required
- [Items requiring follow-up with client]

Focus on client needs, business implications, and clear action items."""


# Planning meeting prompt
PLANNING_MEETING_PROMPT = """You are an expert at summarizing planning and strategy meetings. Analyze the following planning meeting transcript.

PLANNING MEETING TRANSCRIPT:
{content}

Please organize the information into this format:

# Planning Meeting Summary

## Meeting Overview
- Purpose: [Main objective of the planning session]
- Participants: [Team members involved]
- Time Frame: [Planning period discussed]

## Goals & Objectives
- [Goal 1]: [Description and success criteria]
- [Goal 2]: [Description and success criteria]

## Key Decisions Made
- [Decision 1]: [Impact and rationale]
- [Decision 2]: [Impact and rationale]

## Strategy & Approach
- [Strategic element 1]: [Description]
- [Strategic element 2]: [Description]

## Resource Requirements
- [Resource type]: [Quantity/details needed]
- [Resource type]: [Quantity/details needed]

## Timeline & Milestones
- [Milestone 1]: [Target date]
- [Milestone 2]: [Target date]

## Risk Assessment
- [Risk 1]: [Description and mitigation plan]
- [Risk 2]: [Description and mitigation plan]

## Action Items
- [Action] - [Owner] - [Due date]
- [Action] - [Owner] - [Due date]

## Success Metrics
- [Metric 1]: [How it will be measured]
- [Metric 2]: [How it will be measured]

## Next Review Points
- [When the plan will be reviewed next]

Focus on strategic decisions, resource allocation, and measurable outcomes."""


# Interview prompt
INTERVIEW_MEETING_PROMPT = """You are an expert at summarizing interview sessions. Analyze the following interview transcript.

INTERVIEW TRANSCRIPT:
{content}

Please organize the information into this format:

# Interview Summary

## Interview Details
- Date: [If mentioned]
- Position: [Role being interviewed for]
- Interviewer(s): [Names if mentioned]
- Candidate: [Name if mentioned]
- Duration: [If mentioned]

## Candidate Background
- [Key experience points discussed]
- [Educational background if mentioned]
- [Current role/company if mentioned]

## Technical Discussion
- [Technical skills assessed]
- [Technical questions asked and responses]
- [Technical challenges or problems solved]

## Behavioral & Cultural Fit
- [Behavioral questions and responses]
- [Cultural fit observations]
- [Team dynamics discussion]

## Candidate Questions
- [Questions the candidate asked about the role]
- [Questions about the company/team]

## Strengths Identified
- [Strength 1]: [Description]
- [Strength 2]: [Description]

## Areas of Concern
- [Concern 1]: [Description]
- [Concern 2]: [Description]

## Next Steps
- [Follow-up actions required]
- [Timeline for decision]
- [Additional interviews needed]

## Overall Assessment
[Brief summary of the interviewer's impression and recommendation]

Focus on candidate evaluation, skills assessment, and hiring decision factors."""


# Retrospective meeting prompt
RETROSPECTIVE_MEETING_PROMPT = """You are an expert at summarizing retrospective meetings. Analyze the following retrospective transcript.

RETROSPECTIVE TRANSCRIPT:
{content}

Please organize the information into this format:

# Retrospective Summary

## Retrospective Details
- Date: [If mentioned]
- Sprint/Period: [Time period being reviewed]
- Participants: [Team members involved]
- Facilitator: [If mentioned]

## What Went Well
- [Positive item 1]: [Description]
- [Positive item 2]: [Description]

## What Could Be Improved
- [Improvement area 1]: [Description]
- [Improvement area 2]: [Description]

## What Didn't Work
- [Issue 1]: [Description and impact]
- [Issue 2]: [Description and impact]

## Root Cause Analysis
- [Issue]: [Root cause identified]
- [Issue]: [Root cause identified]

## Action Items for Next Sprint
- [Action 1] - [Owner] - [Target completion]
- [Action 2] - [Owner] - [Target completion]

## Process Improvements
- [Process change 1]: [Description and expected benefit]
- [Process change 2]: [Description and expected benefit]

## Team Feedback
- [Feedback on team dynamics]
- [Feedback on collaboration]
- [Feedback on communication]

## Metrics & Outcomes
- [Metric 1]: [Value and trend]
- [Metric 2]: [Value and trend]

## Experiments to Try
- [Experiment 1]: [Description and success criteria]
- [Experiment 2]: [Description and success criteria]

Focus on continuous improvement, team dynamics, and actionable changes."""


def get_specialized_prompt(meeting_type: MeetingType, content: str) -> str:
    """
    Get a specialized prompt based on meeting type.

    Args:
        meeting_type: Type of meeting
        content: Meeting transcript content

    Returns:
        Formatted specialized prompt
    """
    prompt_map = {
        MeetingType.STANDUP: STANDUP_MEETING_PROMPT,
        MeetingType.CLIENT_CALL: CLIENT_CALL_PROMPT,
        MeetingType.PLANNING: PLANNING_MEETING_PROMPT,
        MeetingType.INTERVIEW: INTERVIEW_MEETING_PROMPT,
        MeetingType.RETROSPECTIVE: RETROSPECTIVE_MEETING_PROMPT,
    }

    if meeting_type in prompt_map:
        return prompt_map[meeting_type].format(content=content)
    else:
        # Fall back to general structured prompt
        from .meeting_prompts import create_structured_meeting_prompt

        return create_structured_meeting_prompt(content)
