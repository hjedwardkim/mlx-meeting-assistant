# Transcription Tool

A CLI tool for audio/video transcription and summarization using MLX backend with speaker diarization support.

## Installation

Install the tool using uv:

```bash
uv sync
```

The CLI entry point `transcription-tool` or `meeting-assistant` will be available after installation.

## Commands

The `mlx-meeting-assistant` tool has 4 commands:

- `transcribe`: Transcribe audio file to text.
- `summarize`: Summarize text transcription to summaries.
- `pipeline`: Transcribe and summarize without diarization (distinguishing speakers).
- `diarize`: Transcribe, diarize, and summarize with speaker distinctions.

### Transcribe

Transcribe audio/video files to text:

```bash
# Basic transcription
transcription-tool transcribe video.mp4

# Specify custom model
transcription-tool transcribe video.mp4 --model mlx-community/whisper-large-v3-turbo

# Save to file
transcription-tool transcribe video.mp4 --output transcript.txt
```

### Summarize

Summarize text from files or stdin:

```bash
# Summarize from file
transcription-tool summarize transcript.txt

# Summarize from stdin
cat transcript.txt | transcription-tool summarize -

# Custom model and output
transcription-tool summarize transcript.txt --model mlx-community/Llama-3.2-3B-Instruct-4bit --output summary.txt

# Control summary length
transcription-tool summarize transcript.txt --max-tokens 300
```

### Pipeline

Run complete transcription + summarization pipeline:

```bash
# End-to-end processing
transcription-tool pipeline video.mp4

# Custom models
transcription-tool pipeline video.mp4 \
  --transcription-model mlx-community/whisper-large-v3-turbo \
  --summarization-model mlx-community/Llama-3.2-3B-Instruct-4bit

# Save outputs
transcription-tool pipeline video.mp4 \
  --save-transcription transcript.txt \
  --output summary.txt
```

## Speaker Diarization

The tool supports speaker diarization (identifying "who spoke when") using pyannote.audio:

### Setup for Diarization

1. **Get HuggingFace Token**:

   - Visit <https://hf.co/settings/tokens>
   - Create an access token

2. **Accept Model Conditions**:

   - Visit <https://hf.co/pyannote/speaker-diarization-3.1> and accept conditions
   - Visit <https://hf.co/pyannote/segmentation-3.0> and accept conditions

3. **Set Environment Variable**:

   ```bash
   export HUGGINGFACE_TOKEN=your_token_here
   ```

### Diarization Commands

```bash
# Complete diarization pipeline
meeting-assistant diarize meeting.mp4

# Specify number of speakers if known
meeting-assistant diarize meeting.mp4 --num-speakers 3

# Set speaker bounds for better accuracy
meeting-assistant diarize meeting.mp4 --min-speakers 2 --max-speakers 5

# Custom models and output paths
meeting-assistant diarize meeting.mp4 \
  --diarization-model pyannote/speaker-diarization-3.1 \
  --save-transcription transcript_with_speakers.txt \
  --save-rttm diarization.rttm \
  --output summary_with_speakers.md
```

### Output Formats

- **Speaker-attributed transcript**: Text with speaker labels and timestamps
- **RTTM file**: Standard diarization format for further processing
- **Structured meeting notes**: Enhanced summaries with speaker attribution

## Supported Formats

- **Input**: MPEG-4 and other common audio/video formats supported by MLX Whisper
- **Output**: Plain text transcriptions and summaries

## Models

### Default Models

- **Transcription**: `mlx-community/whisper-large-v3-mlx`
- **Diarization**: `pyannote/speaker-diarization-3.1`
- **Summarization**: `lmstudio-community/Qwen3-30B-A3B-MLX-8bit`

### Custom Models

All commands support custom model selection via `--model`, `--transcription-model`, `--diarization-model`, and `--summarization-model` flags. Models must be compatible with MLX Whisper, pyannote.audio, and MLX LM respectively.

## Error Handling

The tool provides clear error messages for:

- Invalid file paths or formats
- Model loading failures
- MLX backend issues
- File I/O operations
- HuggingFace authentication issues

## Development

Run the tool in development mode:

```bash
uv run python -m transcription_tool.cli --help
```

## Requirements

- Python 3.12+
- MLX-compatible hardware (Apple Silicon recommended)
- HuggingFace token for diarization features
- Required dependencies are automatically installed via uv

## Example Usage

### Basic Meeting Processing

```bash
# Simple transcription and summary
meeting-assistant pipeline meeting.mp4

# With speaker identification
meeting-assistant diarize meeting.mp4 --min-speakers 2 --max-speakers 4
```

### Advanced Workflows

```bash
# High-quality transcription with custom models
meeting-assistant diarize interview.m4a \
  --transcription-model mlx-community/whisper-large-v3-turbo \
  --diarization-model pyannote/speaker-diarization-3.1 \
  --summarization-model mlx-community/Qwen3-30B-A3B-MLX-8bit \
  --meeting-type interview \
  --structured

# Batch processing multiple files
for file in meetings/*.mp4; do
  meeting-assistant diarize "$file" --min-speakers 2 --max-speakers 6
done
```

### Output Examples

#### Speaker-Attributed Transcript

```
**Speaker_0:**
[00:15] Welcome everyone to today's project review meeting.
[00:22] Let's start by going through our progress from last week.

**Speaker_1:**
[00:35] Thanks for organizing this. I have updates on the backend implementation.
[00:42] We've completed the API endpoints for user authentication.
```

#### Structured Meeting Notes with Speaker Attribution

```markdown
# Meeting Notes

## Summary

Project review meeting covering backend progress, frontend updates, and deployment planning with clear action items assigned.

## Participants

- Speaker_0: Project Manager (leading discussion)
- Speaker_1: Backend Developer
- Speaker_2: Frontend Developer

## Decisions Made

- Deploy to staging environment next week - Agreed by Speaker_0 and Speaker_1
- Use React hooks for state management - Proposed by Speaker_2

## Action Items

- Complete API documentation - Assigned to Speaker_1 - Due Friday
- Review UI mockups - Assigned to Speaker_2 - Due Wednesday
```
