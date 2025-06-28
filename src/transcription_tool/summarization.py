"""Summarization module using MLX LM with speaker awareness."""

import sys
import re
from pathlib import Path
from typing import Optional
from datetime import datetime

from mlx_lm.utils import load
from mlx_lm.generate import generate

from prompts import (
    create_structured_meeting_prompt,
    create_simple_meeting_prompt,
    create_speaker_aware_meeting_prompt,
    get_specialized_prompt,
    MeetingType,
)


def _remove_thinking_tokens(text: str) -> str:
    """
    Remove thinking tokens from model output.

    Args:
        text: Raw model output that may contain thinking tokens

    Returns:
        Cleaned text without thinking tokens
    """
    # Remove <think>...</think> blocks (including multiline)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # Remove other common thinking patterns
    text = re.sub(
        r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL | re.IGNORECASE
    )
    text = re.sub(
        r"\*thinking\*.*?\*/thinking\*", "", text, flags=re.DOTALL | re.IGNORECASE
    )

    # Clean up any extra whitespace left behind
    text = re.sub(
        r"\n\s*\n", "\n\n", text
    )  # Replace multiple newlines with double newlines
    text = text.strip()

    return text


def _validate_and_format_output(summary: str, meeting_type: str = "general") -> str:
    """
    Validate and ensure consistent formatting of the structured output.

    Args:
        summary: Raw structured summary from the model
        meeting_type: Type of meeting for appropriate header

    Returns:
        Validated and formatted summary
    """
    # Define expected headers based on meeting type
    expected_headers = {
        "standup": "# Daily Standup Notes",
        "client_call": "# Client Call Summary",
        "planning": "# Planning Meeting Summary",
        "interview": "# Interview Summary",
        "retrospective": "# Retrospective Summary",
        "general": "# Meeting Notes",
    }

    expected_header = expected_headers.get(meeting_type, "# Meeting Notes")

    # Check if the output starts with the expected header
    if not summary.strip().startswith(
        expected_header.split()[1]
    ):  # Check for the main title word
        # If the model didn't follow the format, wrap it
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        formatted_summary = f"""{expected_header}
        *Generated on {timestamp}*

        ## Summary
        {summary}

        ## Additional Notes
        *Note: The AI model did not follow the structured format template. The above content represents the raw summary output.*
        """
        return formatted_summary

    # Clean up any formatting inconsistencies
    summary = re.sub(r"\n{3,}", "\n\n", summary)  # Limit to max 2 consecutive newlines
    summary = summary.strip()

    return summary


def _validate_input_content(content: str) -> str:
    """
    Validate and preprocess input content.

    Args:
        content: Input content to validate

    Returns:
        Validated and cleaned content

    Raises:
        ValueError: If content is invalid
    """
    if not content or not content.strip():
        raise ValueError("No content to summarize")

    # Basic content validation
    content = content.strip()

    # Check if content is too short to be meaningful
    if len(content.split()) < 10:
        raise ValueError(
            "Content too short for meaningful summarization (minimum 10 words)"
        )

    # Check if content is reasonable length (not too long)
    if len(content) > 1000000:  # 1MB limit
        raise ValueError("Content too long for processing (maximum 1MB)")

    return content


def _detect_speaker_content(content: str) -> bool:
    """
    Detect if content contains speaker attribution markers.

    Args:
        content: Input content to analyze

    Returns:
        True if speaker markers are detected, False otherwise
    """
    # Look for speaker markers like **Speaker_0:** or **Speaker_1:**
    speaker_patterns = [
        r"\*\*Speaker_\d+:\*\*",  # **Speaker_0:**
        r"Speaker_\d+:",  # Speaker_0:
        r"\[Speaker_\d+\]",  # [Speaker_0]
        r"\*\*Speaker_SPEAKER_\d+:\*\*",  # **Speaker_SPEAKER_00:**
    ]

    for pattern in speaker_patterns:
        if re.search(pattern, content):
            return True

    return False


def _is_file_path(text_input: str) -> bool:
    """
    Determine if text_input is a file path or direct text content.

    Args:
        text_input: Input to check

    Returns:
        True if it's likely a file path, False if it's direct content
    """
    # If it's stdin indicator, return False
    if text_input == "-":
        return False

    # If it contains newlines or is very long, it's probably direct content
    if "\n" in text_input or len(text_input) > 500:
        return False

    # If it looks like a path and the file exists, it's a file path
    try:
        path = Path(text_input)
        return path.exists() and path.is_file()
    except (OSError, ValueError):
        # Invalid path characters or other path-related errors
        return False


def summarize_text(
    text_input: str,
    model: str = "mlx-community/Qwen3-30B-A3B-8bit",
    output_file: Optional[str] = None,
    max_tokens: int = 32000,
    structured: bool = True,
    meeting_type: str = "general",
) -> str:
    """
    Summarize text using MLX LM with speaker awareness.

    Args:
        text_input: Text to summarize (file path, stdin if '-', or direct text content)
        model: MLX LM model to use
        output_file: Optional output file path for summary
        max_tokens: Maximum tokens for summary
        structured: Whether to use structured meeting notes format
        meeting_type: Type of meeting for specialized formatting

    Returns:
        Summary text

    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If input is invalid
        RuntimeError: If summarization fails
    """
    # Handle input text
    try:
        if text_input == "-":
            # Read from stdin
            content = sys.stdin.read()
        elif _is_file_path(text_input):
            # Read from file
            with open(text_input, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            # Treat as direct text content
            content = text_input

    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {text_input}")
    except Exception as e:
        raise RuntimeError(f"Failed to read input: {str(e)}")

    # Validate content
    content = _validate_input_content(content)

    try:
        # Load model with error handling
        try:
            model_obj, tokenizer = load(model)
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{model}': {str(e)}")

        # Create appropriate prompt based on structured flag and speaker content
        if structured:
            # Check if content has speaker attribution
            has_speakers = _detect_speaker_content(content)

            if has_speakers:
                # Use speaker-aware prompt for content with speaker attribution
                prompt = create_speaker_aware_meeting_prompt(content)
            else:
                # Use regular structured prompt or specialized prompt
                try:
                    meeting_type_enum = MeetingType(meeting_type.lower())
                    if meeting_type_enum in [
                        MeetingType.STANDUP,
                        MeetingType.CLIENT_CALL,
                        MeetingType.PLANNING,
                        MeetingType.INTERVIEW,
                        MeetingType.RETROSPECTIVE,
                    ]:
                        prompt = get_specialized_prompt(meeting_type_enum, content)
                    else:
                        prompt = create_structured_meeting_prompt(content)
                except ValueError:
                    # Fall back to general structured prompt if meeting_type is invalid
                    prompt = create_structured_meeting_prompt(content)
        else:
            # Use simple prompt for backward compatibility
            prompt = create_simple_meeting_prompt(content)

        # Apply chat template if available
        if tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
            )

        # Generate summary with error handling
        try:
            response = generate(
                model_obj,
                tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to generate summary: {str(e)}")

        # Extract and clean summary from response
        summary = response.strip()
        summary = _remove_thinking_tokens(summary)

        if structured:
            summary = _validate_and_format_output(summary, meeting_type)

        # Save to file if specified
        if output_file:
            try:
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(summary)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to save summary to '{output_file}': {str(e)}"
                )

        return summary

    except Exception as e:
        if isinstance(e, (ValueError, FileNotFoundError, RuntimeError)):
            raise
        else:
            raise RuntimeError(f"Summarization failed: {str(e)}")
