"""Transcription module using MLX Whisper with FFmpeg preprocessing for diarization."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlx_whisper

from .config import config


def transcribe_audio(
    file_path: str,
    model: str = None,
    output_file: Optional[str] = None,
) -> str:
    """
    Transcribe audio/video file to text using MLX Whisper.

    Args:
        file_path: Path to the audio/video file
        model: MLX Whisper model to use (defaults to config value)
        output_file: Optional output file path for transcription

    Returns:
        Transcribed text

    Raises:
        FileNotFoundError: If input file doesn't exist
        RuntimeError: If transcription fails
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    # Use default model from config if not specified
    if model is None:
        model = config.models.transcription_model

    try:
        # Get transcription options from environment configuration
        transcribe_options = config.transcription.to_dict()

        result = mlx_whisper.transcribe(
            file_path,
            path_or_hf_repo=model,
            **transcribe_options,
        )
        transcription = result["text"]

        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(transcription)

        return transcription

    except Exception as e:
        raise RuntimeError(f"Transcription failed: {str(e)}")


def transcribe_audio_detailed(
    file_path: str,
    model: str = None,
) -> Dict:
    """
    Transcribe audio/video file with detailed segment information.

    Args:
        file_path: Path to the audio/video file
        model: MLX Whisper model to use (defaults to config value)

    Returns:
        Dictionary with 'text' and 'segments' keys

    Raises:
        FileNotFoundError: If input file doesn't exist
        RuntimeError: If transcription fails
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    # Use default model from config if not specified
    if model is None:
        model = config.models.transcription_model

    try:
        # Get transcription options from environment configuration
        transcribe_options = config.transcription.to_dict()

        result = mlx_whisper.transcribe(
            file_path,
            path_or_hf_repo=model,
            **transcribe_options,
        )

        return result

    except Exception as e:
        raise RuntimeError(f"Transcription failed: {str(e)}")


def transcribe_with_diarization(
    file_path: str,
    transcription_model: str = None,
    diarization_model: str = None,
    use_auth_token: Optional[str] = None,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    output_file: Optional[str] = None,
    save_rttm: Optional[str] = None,
) -> Tuple[str, List[Dict], List[Dict]]:
    """
    Transcribe audio with speaker diarization and automatic FFmpeg preprocessing.

    Args:
        file_path: Path to the audio/video file
        transcription_model: MLX Whisper model to use (defaults to config value)
        diarization_model: Pyannote diarization model to use (defaults to config value)
        use_auth_token: HuggingFace access token (defaults to config value)
        num_speakers: Known number of speakers (optional)
        min_speakers: Minimum number of speakers (defaults to config value)
        max_speakers: Maximum number of speakers (defaults to config value)
        output_file: Optional output file path for transcription
        save_rttm: Optional file path to save diarization RTTM

    Returns:
        Tuple of (raw_transcription, aligned_segments, diarization_segments)

    Raises:
        FileNotFoundError: If input file doesn't exist
        RuntimeError: If transcription or diarization fails
    """
    from .alignment import align_transcription_with_diarization
    from .diarization import (
        format_diarization_segments,
        load_diarization_pipeline,
        perform_diarization,
        save_diarization_rttm,
    )

    # Use defaults from config if not specified
    if transcription_model is None:
        transcription_model = config.models.transcription_model
    if diarization_model is None:
        diarization_model = config.models.diarization_model
    if use_auth_token is None:
        use_auth_token = config.huggingface_token
    if min_speakers is None:
        min_speakers = config.diarization.min_speakers
    if max_speakers is None:
        max_speakers = config.diarization.max_speakers

    try:
        # Step 1: Perform detailed transcription using MLX Whisper
        # MLX Whisper handles format conversion automatically
        transcription_result = transcribe_audio_detailed(file_path, transcription_model)

        # Step 2: Load and run diarization with automatic FFmpeg preprocessing
        diar_pipeline = load_diarization_pipeline(diarization_model, use_auth_token)

        # Perform diarization with automatic preprocessing
        diarization = perform_diarization(
            file_path=file_path,
            pipeline=diar_pipeline,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            show_progress=True,
        )

        # Step 3: Format diarization segments
        diarization_segments = format_diarization_segments(diarization)

        # Step 4: Align transcription with diarization
        aligned_segments_objects = align_transcription_with_diarization(
            transcription_result["segments"], diarization_segments
        )

        # Convert aligned segments to dictionaries for JSON serialization
        aligned_segments = []
        for seg in aligned_segments_objects:
            aligned_segments.append(
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                    "speaker": seg.speaker,
                    "speaker_id": seg.speaker_id,
                    "confidence": seg.confidence,
                }
            )

        # Step 5: Save RTTM if requested
        if save_rttm:
            save_diarization_rttm(diarization, save_rttm)

        return transcription_result["text"], aligned_segments, diarization_segments

    except Exception as e:
        raise RuntimeError(f"Transcription with diarization failed: {str(e)}")


def estimate_processing_time(file_path: str) -> dict:
    """
    Estimate processing time for transcription and diarization.

    Args:
        file_path: Path to audio file

    Returns:
        Dictionary with time estimates
    """
    from .audio_preprocessing import get_audio_info

    info = get_audio_info(file_path)

    estimates = {
        "file_duration_minutes": 0,
        "transcription_time_minutes": 0,
        "diarization_time_minutes": 0,
        "total_time_minutes": 0,
    }

    if info["duration_seconds"]:
        duration_minutes = info["duration_seconds"] / 60
        estimates["file_duration_minutes"] = duration_minutes

        # Rough estimates based on typical performance
        # Transcription: ~0.1x real time on Apple Silicon
        # Diarization: ~0.5x real time
        estimates["transcription_time_minutes"] = duration_minutes * 0.1
        estimates["diarization_time_minutes"] = duration_minutes * 0.5
        estimates["total_time_minutes"] = (
            estimates["transcription_time_minutes"]
            + estimates["diarization_time_minutes"]
        )

    return estimates


def get_current_transcription_config() -> Dict:
    """
    Get the current transcription configuration for debugging/logging.

    Returns:
        Dictionary with current configuration values
    """
    return {
        "transcription_model": config.models.transcription_model,
        "transcription_options": config.transcription.to_dict(),
        "diarization_model": config.models.diarization_model,
        "min_speakers": config.diarization.min_speakers,
        "max_speakers": config.diarization.max_speakers,
        "huggingface_token_set": bool(config.huggingface_token),
    }
