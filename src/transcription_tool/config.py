"""Configuration module for managing environment variables and settings."""

import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


def get_bool_env(key: str, default: bool = False) -> bool:
    """Get a boolean environment variable."""
    value = os.getenv(key, str(default)).lower()
    return value in ("true", "1", "yes", "on")


def get_float_env(key: str, default: float) -> float:
    """Get a float environment variable."""
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


def get_int_env(key: str, default: int) -> int:
    """Get an integer environment variable."""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


class TranscriptionConfig:
    """Configuration for MLX Whisper transcription."""

    def __init__(self):
        # Whisper transcription options from environment
        self.temperature = get_float_env("WHISPER_TEMPERATURE", 0.0)
        self.no_speech_threshold = get_float_env("WHISPER_NO_SPEECH_THRESHOLD", 0.6)
        self.logprob_threshold = get_float_env("WHISPER_LOGPROB_THRESHOLD", -1.0)
        self.compression_ratio_threshold = get_float_env(
            "WHISPER_COMPRESSION_RATIO_THRESHOLD", 2.4
        )
        self.condition_on_previous_text = get_bool_env(
            "WHISPER_CONDITION_ON_PREVIOUS_TEXT", False
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for MLX Whisper."""
        return {
            "temperature": self.temperature,
            "no_speech_threshold": self.no_speech_threshold,
            "logprob_threshold": self.logprob_threshold,
            "compression_ratio_threshold": self.compression_ratio_threshold,
            "condition_on_previous_text": self.condition_on_previous_text,
        }


class DiarizationConfig:
    """Configuration for speaker diarization."""

    def __init__(self):
        self.model = os.getenv(
            "DEFAULT_DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1"
        )
        self.min_speakers = get_int_env("DEFAULT_MIN_SPEAKERS", 2)
        self.max_speakers = get_int_env("DEFAULT_MAX_SPEAKERS", 10)


class ModelConfig:
    """Configuration for default models."""

    def __init__(self):
        self.transcription_model = os.getenv(
            "DEFAULT_TRANSCRIPTION_MODEL", "mlx-community/whisper-large-v3-mlx"
        )
        self.summarization_model = os.getenv(
            "DEFAULT_SUMMARIZATION_MODEL", "mlx-community/Qwen3-30B-A3B-8bit"
        )
        self.diarization_model = os.getenv(
            "DEFAULT_DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1"
        )


class DirectoryConfig:
    """Configuration for default output directories."""

    def __init__(self):
        self.transcriptions_dir = Path(
            os.getenv("DEFAULT_TRANSCRIPTIONS_DIR", "./transcriptions")
        )
        self.summaries_dir = Path(os.getenv("DEFAULT_SUMMARIES_DIR", "./summaries"))

    def ensure_directories(self) -> None:
        """Create directories if they don't exist."""
        self.transcriptions_dir.mkdir(exist_ok=True)
        self.summaries_dir.mkdir(exist_ok=True)


class AppConfig:
    """Main application configuration."""

    def __init__(self):
        self.transcription = TranscriptionConfig()
        self.diarization = DiarizationConfig()
        self.models = ModelConfig()
        self.directories = DirectoryConfig()
        self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

    def validate(self) -> None:
        """Validate configuration and raise errors for critical missing values."""
        if not self.huggingface_token:
            print(
                "Warning: HUGGINGFACE_TOKEN not set. Diarization features will not work."
            )

    def __repr__(self) -> str:
        """String representation of configuration (excluding sensitive data)."""
        return (
            f"AppConfig(\n"
            f"  transcription_model='{self.models.transcription_model}',\n"
            f"  summarization_model='{self.models.summarization_model}',\n"
            f"  diarization_model='{self.models.diarization_model}',\n"
            f"  whisper_temperature={self.transcription.temperature},\n"
            f"  whisper_no_speech_threshold={self.transcription.no_speech_threshold},\n"
            f"  min_speakers={self.diarization.min_speakers},\n"
            f"  max_speakers={self.diarization.max_speakers},\n"
            f"  transcriptions_dir='{self.directories.transcriptions_dir}',\n"
            f"  summaries_dir='{self.directories.summaries_dir}',\n"
            f"  huggingface_token_set={'***' if self.huggingface_token else 'False'}\n"
            f")"
        )


# Global configuration instance
config = AppConfig()
