"""Alignment module for matching transcription segments with diarization."""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class TranscriptionSegment:
    """Represents a transcription segment from Whisper."""

    id: int
    start: float
    end: float
    text: str
    tokens: List[int]
    avg_logprob: float


@dataclass
class DiarizationSegment:
    """Represents a speaker diarization segment."""

    start: float
    end: float
    speaker: str
    speaker_id: str


@dataclass
class AlignedSegment:
    """Represents an aligned transcription + diarization segment."""

    start: float
    end: float
    text: str
    speaker: str
    speaker_id: str
    confidence: float
    transcription_segment: TranscriptionSegment
    diarization_segments: List[DiarizationSegment]


def calculate_overlap(
    seg1_start: float, seg1_end: float, seg2_start: float, seg2_end: float
) -> float:
    """
    Calculate temporal overlap between two segments.

    Args:
        seg1_start, seg1_end: First segment boundaries
        seg2_start, seg2_end: Second segment boundaries

    Returns:
        Overlap duration in seconds
    """
    overlap_start = max(seg1_start, seg2_start)
    overlap_end = min(seg1_end, seg2_end)
    return max(0, overlap_end - overlap_start)


def calculate_overlap_ratio(
    seg1_start: float, seg1_end: float, seg2_start: float, seg2_end: float
) -> float:
    """
    Calculate overlap ratio (overlap / shorter segment duration).

    Returns:
        Overlap ratio between 0 and 1
    """
    overlap = calculate_overlap(seg1_start, seg1_end, seg2_start, seg2_end)
    seg1_duration = seg1_end - seg1_start
    seg2_duration = seg2_end - seg2_start
    shorter_duration = min(seg1_duration, seg2_duration)

    if shorter_duration == 0:
        return 0.0

    return overlap / shorter_duration


def align_transcription_with_diarization(
    transcription_segments: List[Dict],
    diarization_segments: List[Dict],
    overlap_threshold: float = 0.3,
) -> List[AlignedSegment]:
    """
    Align transcription segments with diarization segments.

    Args:
        transcription_segments: Whisper transcription segments
        diarization_segments: Pyannote diarization segments
        overlap_threshold: Minimum overlap ratio for matching

    Returns:
        List of aligned segments with speaker attribution
    """
    aligned_segments = []

    # Convert to structured objects
    trans_segs = [
        TranscriptionSegment(
            id=seg.get("id", i),
            start=seg["start"],
            end=seg["end"],
            text=seg["text"].strip(),
            tokens=seg.get("tokens", []),
            avg_logprob=seg.get("avg_logprob", 0.0),
        )
        for i, seg in enumerate(transcription_segments)
    ]

    diar_segs = [
        DiarizationSegment(
            start=seg["start"],
            end=seg["end"],
            speaker=seg["speaker"],
            speaker_id=seg["speaker_id"],
        )
        for seg in diarization_segments
    ]

    # Align each transcription segment
    for trans_seg in trans_segs:
        best_match = None
        best_overlap_ratio = 0.0
        matching_diar_segments = []

        # Find all overlapping diarization segments
        for diar_seg in diar_segs:
            overlap_ratio = calculate_overlap_ratio(
                trans_seg.start, trans_seg.end, diar_seg.start, diar_seg.end
            )

            if overlap_ratio >= overlap_threshold:
                matching_diar_segments.append((diar_seg, overlap_ratio))

                if overlap_ratio > best_overlap_ratio:
                    best_overlap_ratio = overlap_ratio
                    best_match = diar_seg

        # Create aligned segment
        if best_match:
            # Use the best matching speaker
            speaker = best_match.speaker
            speaker_id = best_match.speaker_id
            confidence = best_overlap_ratio

            # Collect all matching diarization segments for reference
            matching_segments = [seg for seg, _ in matching_diar_segments]

        else:
            # No clear speaker match - assign to "Unknown"
            speaker = "Unknown_Speaker"
            speaker_id = "UNKNOWN"
            confidence = 0.0
            matching_segments = []

        aligned_segment = AlignedSegment(
            start=trans_seg.start,
            end=trans_seg.end,
            text=trans_seg.text,
            speaker=speaker,
            speaker_id=speaker_id,
            confidence=confidence,
            transcription_segment=trans_seg,
            diarization_segments=matching_segments,
        )

        aligned_segments.append(aligned_segment)

    return aligned_segments


def format_aligned_transcript(aligned_segments: List[AlignedSegment]) -> str:
    """
    Format aligned segments into readable transcript.

    Args:
        aligned_segments: List of aligned segments

    Returns:
        Formatted transcript string
    """
    transcript_lines = []
    current_speaker = None

    for segment in aligned_segments:
        # Add speaker header when speaker changes
        if segment.speaker != current_speaker:
            if transcript_lines:  # Add spacing between speakers
                transcript_lines.append("")

            transcript_lines.append(f"**{segment.speaker}:**")
            current_speaker = segment.speaker

        # Format timestamp and text
        start_time = f"{int(segment.start // 60):02d}:{int(segment.start % 60):02d}"
        transcript_lines.append(f"[{start_time}] {segment.text}")

    return "\n".join(transcript_lines)
