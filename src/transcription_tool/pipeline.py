"""Pipeline module for orchestrating transcription and summarization."""

from pathlib import Path
from typing import Optional, Tuple

from .transcription import transcribe_audio
from .summarization import summarize_text


def run_pipeline(
    file_path: str,
    transcription_model: str = "mlx-community/parakeet-tdt-0.6b-v2",
    summarization_model: str = "mlx-community/Qwen3-30B-A3B-8bit",
    output_file: Optional[str] = None,
    save_transcription: Optional[str] = None,
    max_tokens: int = 500,
) -> Tuple[str, str]:
    """
    Run complete pipeline: transcription followed by summarization.

    Args:
        file_path: Path to the audio/video file
        transcription_model: MLX Whisper model to use
        summarization_model: MLX LM model to use
        output_file: Optional output file path for final summary
        save_transcription: Optional file path to save intermediate transcription
        max_tokens: Maximum tokens for summary

    Returns:
        Tuple of (transcription, summary)

    Raises:
        FileNotFoundError: If input file doesn't exist
        RuntimeError: If any step fails
    """
    try:
        # Step 1: Transcribe audio/video
        transcription = transcribe_audio(
            file_path=file_path,
            model=transcription_model,
            output_file=save_transcription,
        )

        # Step 2: Summarize transcription
        summary = summarize_text(
            text_input=transcription,
            model=summarization_model,
            output_file=output_file,
            max_tokens=max_tokens,
        )

        return transcription, summary

    except Exception as e:
        raise RuntimeError(f"Pipeline failed: {str(e)}")

