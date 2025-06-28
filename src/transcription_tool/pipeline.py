"""Pipeline module for orchestrating transcription and summarization with diarization support."""

from typing import Optional, Tuple, List, Dict
from pathlib import Path

from .transcription import transcribe_audio
from .summarization import summarize_text


def run_pipeline(
    file_path: str,
    transcription_model: str = "mlx-community/whisper-large-v3-mlx",
    summarization_model: str = "mlx-community/Qwen3-30B-A3B-8bit",
    output_file: Optional[str] = None,
    save_transcription: Optional[str] = None,
    max_tokens: int = 32000,
    structured: bool = True,
    meeting_type: str = "general",
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
        structured: Whether to use structured meeting notes format
        meeting_type: Type of meeting for specialized formatting

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

        # Step 2: Summarize transcription with structured output
        summary = summarize_text(
            text_input=transcription,
            model=summarization_model,
            output_file=output_file,
            max_tokens=max_tokens,
            structured=structured,
            meeting_type=meeting_type,
        )

        return transcription, summary

    except Exception as e:
        raise RuntimeError(f"Pipeline failed: {str(e)}")


def run_diarization_pipeline(
    file_path: str,
    transcription_model: str = "mlx-community/whisper-large-v3-mlx",
    diarization_model: str = "pyannote/speaker-diarization-3.1",
    summarization_model: str = "mlx-community/Qwen3-30B-A3B-8bit",
    use_auth_token: Optional[str] = None,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    output_file: Optional[str] = None,
    save_transcription: Optional[str] = None,
    save_rttm: Optional[str] = None,
    max_tokens: int = 32000,
    structured: bool = True,
    meeting_type: str = "general",
) -> Tuple[str, str, List[Dict]]:
    """
    Run complete diarization pipeline: transcription + diarization + summarization.

    Args:
        file_path: Path to the audio/video file
        transcription_model: MLX Whisper model to use
        diarization_model: Pyannote diarization model to use
        summarization_model: MLX LM model to use
        use_auth_token: HuggingFace access token
        num_speakers: Known number of speakers (optional)
        min_speakers: Minimum number of speakers (optional)
        max_speakers: Maximum number of speakers (optional)
        output_file: Optional output file path for final summary
        save_transcription: Optional file path to save intermediate transcription
        save_rttm: Optional file path to save diarization RTTM
        max_tokens: Maximum tokens for summary
        structured: Whether to use structured meeting notes format
        meeting_type: Type of meeting for specialized formatting

    Returns:
        Tuple of (transcription, summary, aligned_segments)

    Raises:
        FileNotFoundError: If input file doesn't exist
        RuntimeError: If any step fails
    """
    try:
        # Step 1: Transcribe with diarization
        from .transcription import transcribe_with_diarization
        from .alignment import format_aligned_transcript, AlignedSegment

        raw_transcription, aligned_segments_dict, diarization_segments = (
            transcribe_with_diarization(
                file_path=file_path,
                transcription_model=transcription_model,
                diarization_model=diarization_model,
                use_auth_token=use_auth_token,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                save_rttm=save_rttm,
            )
        )

        # Step 2: Create speaker-attributed transcript
        # Convert dict segments back to AlignedSegment objects for formatting
        aligned_segments_objects = []
        for seg_dict in aligned_segments_dict:
            # Create a minimal AlignedSegment for formatting
            aligned_seg = type(
                "AlignedSegment",
                (),
                {
                    "start": seg_dict["start"],
                    "end": seg_dict["end"],
                    "text": seg_dict["text"],
                    "speaker": seg_dict["speaker"],
                    "speaker_id": seg_dict["speaker_id"],
                    "confidence": seg_dict["confidence"],
                },
            )()
            aligned_segments_objects.append(aligned_seg)

        speaker_transcript = format_aligned_transcript(aligned_segments_objects)

        # Save transcription if requested
        if save_transcription:
            output_path = Path(save_transcription)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(speaker_transcript)

        # Step 3: Summarize with speaker context
        summary = summarize_text(
            text_input=speaker_transcript,
            model=summarization_model,
            output_file=output_file,
            max_tokens=max_tokens,
            structured=structured,
            meeting_type=meeting_type,
        )

        return speaker_transcript, summary, aligned_segments_dict

    except Exception as e:
        raise RuntimeError(f"Diarization pipeline failed: {str(e)}")
