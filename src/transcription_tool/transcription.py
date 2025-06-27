"""Transcription module using MLX Whisper."""

import os
from pathlib import Path
from typing import Optional

import mlx_whisper


def transcribe_audio(
    file_path: str,
    model: str = "mlx-community/parakeet-tdt-0.6b-v2",
    # model: str = "mlx-community/whisper-large-v3-mlx",
    output_file: Optional[str] = None,
) -> str:
    """
    Transcribe audio/video file to text using MLX Whisper.

    Args:
        file_path: Path to the audio/video file
        model: MLX Whisper model to use
        output_file: Optional output file path for transcription

    Returns:
        Transcribed text

    Raises:
        FileNotFoundError: If input file doesn't exist
        RuntimeError: If transcription fails
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    try:
        result = mlx_whisper.transcribe(file_path, path_or_hf_repo=model)
        transcription = result["text"]

        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(transcription)

        return transcription

    except Exception as e:
        raise RuntimeError(f"Transcription failed: {str(e)}")
