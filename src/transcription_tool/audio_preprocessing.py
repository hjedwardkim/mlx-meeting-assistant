"""Audio preprocessing module for format conversion using FFmpeg."""

import os
from pathlib import Path
import tempfile
from typing import Optional, Tuple
import warnings

import ffmpeg

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


class AudioPreprocessingError(Exception):
    """Custom exception for audio preprocessing errors."""

    pass


def convert_to_wav_ffmpeg(
    input_path: str, output_path: str, sample_rate: int = 16000, channels: int = 1
) -> None:
    """
    Convert audio file to WAV format using ffmpeg.

    Args:
        input_path: Path to input audio file
        output_path: Path for output WAV file
        sample_rate: Target sample rate (default: 16000 Hz)
        channels: Number of channels (default: 1 for mono)

    Raises:
        AudioPreprocessingError: If conversion fails
    """
    try:
        # Create output directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Use ffmpeg to convert to WAV with specific parameters
        stream = ffmpeg.input(input_path)
        stream = ffmpeg.output(
            stream,
            output_path,
            acodec="pcm_s16le",  # 16-bit PCM
            ac=channels,  # Number of channels
            ar=sample_rate,  # Sample rate
            y=None,  # Overwrite output file if exists
        )

        # Run the conversion
        ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)

    except ffmpeg.Error as e:
        stderr_output = e.stderr.decode("utf-8") if e.stderr else "No error details"
        raise AudioPreprocessingError(f"FFmpeg conversion failed: {stderr_output}")
    except Exception as e:
        raise AudioPreprocessingError(f"Audio conversion failed: {str(e)}")


def preprocess_audio_for_diarization(
    input_path: str,
    temp_dir: Optional[str] = None,
    sample_rate: int = 16000,
    cleanup: bool = True,
) -> Tuple[str, bool]:
    """
    Preprocess audio file for diarization compatibility.

    Args:
        input_path: Path to input audio file
        temp_dir: Temporary directory for converted files (optional)
        sample_rate: Target sample rate for diarization
        cleanup: Whether to clean up temporary files automatically

    Returns:
        Tuple of (processed_file_path, is_temporary_file)

    Raises:
        AudioPreprocessingError: If preprocessing fails
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input audio file not found: {input_path}")

    input_path = Path(input_path)

    # Check if file is already in WAV format
    if input_path.suffix.lower() == ".wav":
        # Verify it's compatible (we'll assume it is for now)
        return str(input_path), False

    # Create temporary file for conversion
    if temp_dir is None:
        temp_dir = tempfile.gettempdir()

    temp_file = Path(temp_dir) / f"{input_path.stem}_diarization.wav"

    try:
        # Convert to WAV using FFmpeg
        convert_to_wav_ffmpeg(str(input_path), str(temp_file), sample_rate)

        return str(temp_file), True

    except Exception as e:
        # Clean up on failure
        if temp_file.exists():
            temp_file.unlink()
        raise AudioPreprocessingError(f"Audio preprocessing failed: {str(e)}")


def cleanup_temporary_file(file_path: str) -> None:
    """
    Clean up temporary audio file.

    Args:
        file_path: Path to temporary file to remove
    """
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
    except Exception:
        # Ignore cleanup errors
        pass


def get_audio_info(file_path: str) -> dict:
    """
    Get basic information about an audio file using FFmpeg.

    Args:
        file_path: Path to audio file

    Returns:
        Dictionary with audio file information
    """
    info = {
        "exists": os.path.exists(file_path),
        "format": Path(file_path).suffix.lower(),
        "size_mb": 0,
        "duration_seconds": None,
        "sample_rate": None,
        "channels": None,
    }

    if not info["exists"]:
        return info

    try:
        # Get file size
        info["size_mb"] = os.path.getsize(file_path) / (1024 * 1024)

        # Use ffmpeg.probe to get audio metadata
        probe = ffmpeg.probe(file_path)

        # Extract audio stream info
        audio_streams = [
            stream for stream in probe["streams"] if stream["codec_type"] == "audio"
        ]
        if audio_streams:
            audio_stream = audio_streams[0]  # Use first audio stream
            info["sample_rate"] = int(audio_stream.get("sample_rate", 0))
            info["channels"] = int(audio_stream.get("channels", 0))

        # Get duration from format info
        if "format" in probe and "duration" in probe["format"]:
            info["duration_seconds"] = float(probe["format"]["duration"])

    except Exception:
        pass  # Metadata extraction failed, but file exists

    return info


def validate_audio_for_diarization(file_path: str) -> dict:
    """
    Validate audio file for diarization compatibility.

    Args:
        file_path: Path to audio file

    Returns:
        Dictionary with validation results
    """
    validation = {
        "is_valid": False,
        "needs_conversion": False,
        "issues": [],
        "recommendations": [],
    }

    info = get_audio_info(file_path)

    if not info["exists"]:
        validation["issues"].append("File does not exist")
        return validation

    # Check format
    supported_formats = [
        ".wav",
        ".mp3",
        ".m4a",
        ".mp4",
        ".flac",
        ".ogg",
        ".avi",
        ".mov",
        ".wmv",
    ]
    if info["format"] not in supported_formats:
        validation["issues"].append(f"Unsupported format: {info['format']}")
        validation["recommendations"].append(
            "Convert to a supported format (WAV, MP3, MP4, M4A)"
        )

    # Check if conversion is needed
    if info["format"] != ".wav":
        validation["needs_conversion"] = True
        validation["recommendations"].append(
            "Will be automatically converted to WAV format for diarization"
        )

    # Check file size
    if info["size_mb"] > 500:  # 500MB limit
        validation["issues"].append(f"File very large ({info['size_mb']:.1f}MB)")
        validation["recommendations"].append("Consider splitting large files")

    # Check duration
    if info["duration_seconds"] and info["duration_seconds"] > 7200:  # 2 hours
        validation["issues"].append(
            f"File very long ({info['duration_seconds'] / 60:.1f} minutes)"
        )
        validation["recommendations"].append("Consider splitting long recordings")

    # Overall validation
    validation["is_valid"] = (
        len(validation["issues"]) == 0 or validation["needs_conversion"]
    )

    return validation


def probe_ffmpeg_availability() -> dict:
    """
    Check if FFmpeg is available and working.

    Returns:
        Dictionary with FFmpeg availability information
    """
    result = {"available": False, "version": None, "error": None}

    try:
        # Try to get FFmpeg version
        probe = ffmpeg.probe("/dev/null", v="quiet")
        result["available"] = True
    except ffmpeg.Error:
        # FFmpeg is available but probe failed (expected for /dev/null)
        result["available"] = True
    except Exception as e:
        result["available"] = False
        result["error"] = str(e)

    # Try to get version info
    if result["available"]:
        try:
            version_info = ffmpeg.run(
                ffmpeg.input("pipe:", f="lavfi", t=0.1, i="testsrc2=size=1x1:rate=1"),
                ffmpeg.output("pipe:", f="null"),
                capture_stdout=True,
                capture_stderr=True,
            )
            result["version"] = "Available"
        except Exception:
            pass

    return result
