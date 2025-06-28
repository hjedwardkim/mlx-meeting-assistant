"""Transcription module using MLX Whisper with diarization support."""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import mlx_whisper


def transcribe_audio(
    file_path: str,
    model: str = "mlx-community/whisper-large-v3-mlx",
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
        transcribe_options = {
            "temperature": 0.0,
            "no_speech_threshold": 0.6,
            "logprob_threshold": -1.0,
            "compression_ratio_threshold": 2.4,  # lower if repetition happens
            "condition_on_previous_text": False,  # true can cause repetition
        }

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
    model: str = "mlx-community/whisper-large-v3-mlx",
) -> Dict:
    """
    Transcribe audio/video file with detailed segment information.

    Args:
        file_path: Path to the audio/video file
        model: MLX Whisper model to use

    Returns:
        Dictionary with 'text' and 'segments' keys

    Raises:
        FileNotFoundError: If input file doesn't exist
        RuntimeError: If transcription fails
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    try:
        transcribe_options = {
            "temperature": 0.0,
            "no_speech_threshold": 0.6,
            "logprob_threshold": -1.0,
            "compression_ratio_threshold": 2.4,
            "condition_on_previous_text": False,
        }

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
    transcription_model: str = "mlx-community/whisper-large-v3-mlx",
    diarization_model: str = "pyannote/speaker-diarization-3.1",
    use_auth_token: Optional[str] = None,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    output_file: Optional[str] = None,
    save_rttm: Optional[str] = None,
) -> Tuple[str, List[Dict], List[Dict]]:
    """
    Transcribe audio with speaker diarization.

    Args:
        file_path: Path to the audio/video file
        transcription_model: MLX Whisper model to use
        diarization_model: Pyannote diarization model to use
        use_auth_token: HuggingFace access token
        num_speakers: Known number of speakers (optional)
        min_speakers: Minimum number of speakers (optional)
        max_speakers: Maximum number of speakers (optional)
        output_file: Optional output file path for transcription
        save_rttm: Optional file path to save diarization RTTM

    Returns:
        Tuple of (raw_transcription, aligned_segments, diarization_segments)

    Raises:
        FileNotFoundError: If input file doesn't exist
        RuntimeError: If transcription or diarization fails
    """
    from .diarization import (
        load_diarization_pipeline,
        perform_diarization,
        format_diarization_segments,
        save_diarization_rttm,
    )
    from .alignment import align_transcription_with_diarization, AlignedSegment

    try:
        # Perform detailed transcription
        transcription_result = transcribe_audio_detailed(file_path, transcription_model)

        # Load and run diarization
        diar_pipeline = load_diarization_pipeline(diarization_model, use_auth_token)
        diarization = perform_diarization(
            file_path, diar_pipeline, num_speakers, min_speakers, max_speakers
        )

        # Format segments
        diarization_segments = format_diarization_segments(diarization)

        # Align transcription with diarization
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

        # Save RTTM if requested
        if save_rttm:
            save_diarization_rttm(diarization, save_rttm)

        return transcription_result["text"], aligned_segments, diarization_segments

    except Exception as e:
        raise RuntimeError(f"Transcription with diarization failed: {str(e)}")
