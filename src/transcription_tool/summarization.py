"""Summarization module using MLX LM."""

import sys
from pathlib import Path
from typing import Optional

from mlx_lm.utils import load
from mlx_lm.generate import generate


def summarize_text(
    text_input: str,
    model: str = "mlx-community/Qwen3-30B-A3B-8bit",
    output_file: Optional[str] = None,
    max_tokens: int = 32000,
) -> str:
    """
    Summarize text using MLX LM.

    Args:
        text_input: Text to summarize (file path or stdin if '-')
        model: MLX LM model to use
        output_file: Optional output file path for summary
        max_tokens: Maximum tokens for summary

    Returns:
        Summary text

    Raises:
        FileNotFoundError: If input file doesn't exist
        RuntimeError: If summarization fails
    """
    # Handle input text
    if text_input == "-":
        content = sys.stdin.read()
    elif Path(text_input).exists():
        with open(text_input, "r", encoding="utf-8") as f:
            content = f.read()
    else:
        # Treat as direct text input
        content = text_input

    if not content.strip():
        raise ValueError("No content to summarize")

    try:
        # Load model
        model_obj, tokenizer = load(model)

        # Create prompt for summarization
        prompt = f"""
        Please provide a concise summary of the following text of a meeting:

        {content}

        Summary:"""

        if tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        # Generate summary
        response = generate(
            model_obj,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
        )

        # Extract summary from response
        summary = response.strip()

        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(summary)

        return summary

    except Exception as e:
        raise RuntimeError(f"Summarization failed: {str(e)}")

