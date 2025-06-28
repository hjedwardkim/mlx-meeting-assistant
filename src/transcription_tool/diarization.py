"""Speaker diarization module using pyannote.audio with FFmpeg preprocessing support."""

import os
from pathlib import Path
from typing import Optional, Dict, List
import warnings

import torch
from pyannote.audio import Pipeline
from pyannote.core import Annotation
from pyannote.audio.pipelines.utils.hook import ProgressHook

from .audio_preprocessing import (
    preprocess_audio_for_diarization,
    cleanup_temporary_file,
    validate_audio_for_diarization,
    probe_ffmpeg_availability,
    AudioPreprocessingError,
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


class DiarizationError(Exception):
    """Custom exception for diarization-related errors."""

    pass


def load_diarization_pipeline(
    model: str = "pyannote/speaker-diarization-3.1",
    use_auth_token: Optional[str] = None,
    device: Optional[str] = None,
) -> Pipeline:
    """
    Load pyannote.audio diarization pipeline.

    Args:
        model: HuggingFace model identifier
        use_auth_token: HuggingFace access token
        device: Device to run on ('cuda', 'cpu', or None for auto)

    Returns:
        Loaded diarization pipeline

    Raises:
        DiarizationError: If pipeline loading fails
    """
    try:
        # Get token from environment if not provided
        if use_auth_token is None:
            use_auth_token = os.getenv("HUGGINGFACE_TOKEN")

        if use_auth_token is None:
            raise DiarizationError(
                "HuggingFace token required. Set HUGGINGFACE_TOKEN environment variable "
                "or pass use_auth_token parameter. Get token at https://hf.co/settings/tokens"
            )

        pipeline = Pipeline.from_pretrained(model, use_auth_token=use_auth_token)

        # Move to specified device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cuda" and torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))

        return pipeline

    except Exception as e:
        raise DiarizationError(f"Failed to load diarization pipeline: {str(e)}")


def perform_diarization(
    file_path: str,
    pipeline: Pipeline,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    show_progress: bool = True,
) -> Annotation:
    """
    Perform speaker diarization on audio file with automatic preprocessing.

    Args:
        file_path: Path to audio file
        pipeline: Loaded diarization pipeline
        num_speakers: Known number of speakers (optional)
        min_speakers: Minimum number of speakers (optional)
        max_speakers: Maximum number of speakers (optional)
        show_progress: Whether to show progress bar

    Returns:
        Diarization annotation with speaker segments

    Raises:
        DiarizationError: If diarization fails
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    processed_file = None
    is_temporary = False

    try:
        # Validate and preprocess audio file
        validation = validate_audio_for_diarization(file_path)

        if not validation["is_valid"] and not validation["needs_conversion"]:
            issues = "; ".join(validation["issues"])
            recommendations = "; ".join(validation["recommendations"])
            raise DiarizationError(
                f"Audio file validation failed: {issues}. "
                f"Recommendations: {recommendations}"
            )

        # Check FFmpeg availability
        ffmpeg_status = probe_ffmpeg_availability()

        if validation["needs_conversion"]:
            if not ffmpeg_status["available"]:
                raise DiarizationError(
                    "Audio format conversion required but FFmpeg is not available.\n"
                    f"FFmpeg error: {ffmpeg_status.get('error', 'Unknown error')}\n"
                    "Please install FFmpeg: https://ffmpeg.org/download.html"
                )

            if show_progress:
                print(
                    f"Converting {Path(file_path).suffix} to WAV format for diarization..."
                )

            try:
                processed_file, is_temporary = preprocess_audio_for_diarization(
                    file_path
                )
            except AudioPreprocessingError as e:
                raise DiarizationError(f"Audio preprocessing failed: {str(e)}")
        else:
            # Use original file if it's already compatible
            processed_file = file_path
            is_temporary = False

        # Set up diarization parameters
        kwargs = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

        # Perform diarization
        if show_progress:
            print(f"Performing speaker diarization on: {processed_file}")
            with ProgressHook() as hook:
                diarization = pipeline(processed_file, hook=hook, **kwargs)
        else:
            diarization = pipeline(processed_file, **kwargs)

        return diarization

    except Exception as e:
        if isinstance(e, (DiarizationError, FileNotFoundError)):
            raise
        else:
            raise DiarizationError(f"Diarization failed: {str(e)}")

    finally:
        # Clean up temporary file if created
        if is_temporary and processed_file:
            cleanup_temporary_file(processed_file)


def format_diarization_segments(diarization: Annotation) -> List[Dict]:
    """
    Convert diarization annotation to structured format.

    Args:
        diarization: Pyannote diarization annotation

    Returns:
        List of speaker segments with timestamps
    """
    segments = []

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(
            {
                "start": turn.start,
                "end": turn.end,
                "duration": turn.end - turn.start,
                "speaker": f"Speaker_{speaker}",
                "speaker_id": speaker,
            }
        )

    return segments


def save_diarization_rttm(
    diarization: Annotation,
    output_file: str,
) -> None:
    """
    Save diarization results in RTTM format.

    Args:
        diarization: Pyannote diarization annotation
        output_file: Output file path
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as rttm:
        diarization.write_rttm(rttm)


def diagnose_audio_compatibility(file_path: str) -> dict:
    """
    Diagnose audio file compatibility with diarization pipeline.

    Args:
        file_path: Path to audio file

    Returns:
        Dictionary with diagnostic information
    """
    diagnosis = {
        "file_path": file_path,
        "validation": validate_audio_for_diarization(file_path),
        "ffmpeg_available": probe_ffmpeg_availability(),
        "recommendations": [],
    }

    validation = diagnosis["validation"]
    ffmpeg_status = diagnosis["ffmpeg_available"]

    # Generate recommendations
    if not validation["is_valid"]:
        if validation["needs_conversion"]:
            if ffmpeg_status["available"]:
                diagnosis["recommendations"].append(
                    "✓ Audio conversion available - file will be automatically converted to WAV"
                )
            else:
                diagnosis["recommendations"].append(
                    "✗ Install FFmpeg for audio conversion: https://ffmpeg.org/download.html"
                )

        for issue in validation["issues"]:
            diagnosis["recommendations"].append(f"⚠ {issue}")
    else:
        diagnosis["recommendations"].append(
            "✓ Audio file is compatible with diarization"
        )

    return diagnosis
