# MLX Meeting Assistant

An AI-powered CLI tool to transcribe, diarize, and summarize audio/video files using MLX and Pyannote.

**Generate structured meeting notes, action items, and summaries tailored to specific meeting types.**

> Keep your confidential meetings secure with on-device processing using Apple's MLX framework.

⭐ **Please star this repo if you find it useful!** ⭐

[![GitHub stars](https://img.shields.io/github/stars/hjedwardkim/mlx-meeting-assistant?style=social)](https://github.com/hjedwardkim/mlx-meeting-assistant/stargazers)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Features

- **High-Quality Transcription**: Powered by MLX-ported Whisper models for fast, local transcription on Apple Silicon.
- **Speaker Diarization**: Identifies who spoke when using `pyannote.audio`, attributing transcript lines to different speakers.
- **Intelligent Summarization**: Uses MLX-powered LLMs to generate summaries.
- **Structured Notes**: Creates detailed, structured meeting notes in Markdown format, including summaries, decisions, and action items.
- **Specialized Prompts**: Generates notes tailored to the context of different meeting types (`standup`, `interview`, `client_call`, `planning`, `retrospective`).
- **Automatic Format Conversion**: Uses FFmpeg to automatically convert common audio/video formats to the required WAV format for processing.
- **Dependency Checking**: Includes built-in commands to check for necessary system dependencies like FFmpeg.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-repo/mlx-meeting-assistant.git
   cd mlx-meeting-assistant
   ```

2. **Install dependencies using `uv`:**

   ```bash
   uv sync
   ```

This will install all required Python packages and make the `meeting-assistant` command available in your shell.

## System Dependencies

This tool relies on **FFmpeg** for audio preprocessing. You can check if it's installed correctly using the built-in command:

```bash
meeting-assistant check-deps
```

If FFmpeg is not available, please install it:

- **macOS (Homebrew)**: `brew install ffmpeg`

## Commands

The primary entry point for the tool is `meeting-assistant`.

### `diarize`

The main command to run the full pipeline: transcription, speaker diarization, and summarization.

```bash
# Basic usage: Transcribe, identify speakers, and create structured notes
meeting-assistant diarize meeting.mp4

# Specify the number of speakers if known for better accuracy
meeting-assistant diarize meeting.mp4 --num-speakers 3

# Use a specialized prompt for a specific meeting type (e.g., a standup)
meeting-assistant diarize standup.m4a --meeting-type standup

# Save all outputs to custom paths
meeting-assistant diarize call.mp4 \
  --output notes.md \
  --save-transcription transcript_with_speakers.txt \
  --save-rttm diarization.rttm
```

**Diarization Setup:**
The `diarize` command requires a Hugging Face token to download the speaker identification models.

1. **Get Hugging Face Token**:
   - Create an access token at [hf.co/settings/tokens](https://hf.co/settings/tokens).
2. **Accept Model User Agreements**:
   - Accept the terms for [pyannote/speaker-diarization-3.1](https://hf.co/pyannote/speaker-diarization-3.1).
   - Accept the terms for [pyannote/segmentation-3.0](https://hf.co/pyannote/segmentation-3.0).
3. **Set Environment Variable**:

   - Create a `.env` file in the project root and add your token:

     ```
     HUGGINGFACE_TOKEN=your_token_here
     ```

   - Alternatively, you can pass the token directly with the `--hf-token` flag.

### `summarize`

Summarize a text file. This is useful for summarizing an existing transcript. The command automatically detects if the transcript has speaker labels and uses a more detailed prompt accordingly.

```bash
# Summarize a plain transcript into structured notes
meeting-assistant summarize transcript.txt

# Summarize a speaker-attributed transcript with specialized notes for an interview
meeting-assistant summarize speaker_transcript.txt --meeting-type interview

# Summarize from stdin
cat transcript.txt | meeting-assistant summarize -
```

### `pipeline`

Runs transcription and summarization without speaker diarization.

```bash
# Basic usage
meeting-assistant pipeline meeting.mp4

# Use a custom summarization model and save the transcript
meeting-assistant pipeline meeting.mp4 \
  --summarization-model mlx-community/Llama-3.1-8B-Instruct-4bit \
  --save-transcription transcript.txt
```

### `transcribe`

Performs transcription only.

```bash
# Basic transcription
meeting-assistant transcribe video.mp4 --output transcript.txt
```

### `diagnose`

Checks if an audio/video file is compatible with the diarization pipeline and provides recommendations.

```bash
meeting-assistant diagnose "long meeting.mp4"
```

### `check-deps`

Verifies that all necessary system dependencies (like FFmpeg) and Python packages are installed correctly.

```bash
meeting-assistant check-deps
```

## Advanced Summarization

You can control the output of the summarization by using the following flags with the `diarize`, `pipeline`, and `summarize` commands.

- `--structured` / `--simple`: Choose between a detailed, structured Markdown output (default) or a simple paragraph summary.
- `--meeting-type`: Tailor the summary to a specific context. This changes the structure and focus of the generated notes.
  - `general` (default)
  - `standup`
  - `planning`
  - `client_call`
  - `interview`
  - `retrospective`

**Example**:
To get notes specifically for a client call, which focuses on requirements, action items, and client concerns:

```bash
meeting-assistant diarize client_call.mp4 --meeting-type client_call
```

## Advanced Configuration

You can configure default models and other parameters by creating a `.env` file in the project's root directory.

**Example `.env` file:**

```
# Set default models
DEFAULT_TRANSCRIPTION_MODEL=mlx-community/whisper-large-v3-mlx
DEFAULT_SUMMARIZATION_MODEL=mlx-community/Llama-3.1-8B-Instruct-4bit
DEFAULT_DIARIZATION_MODEL=pyannote/speaker-diarization-3.1

# Set Hugging Face Token for diarization
HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx

# Configure Whisper transcription parameters
WHISPER_TEMPERATURE=0.0
WHISPER_NO_SPEECH_THRESHOLD=0.6
```

## Development

To run the tool directly from the source code for development, use `uv`:

```bash
# Run with --help to see all commands and options
uv run python -m src.transcription_tool.cli --help

# Example: Run the diarize command in development mode
uv run python -m src.transcription_tool.cli diarize "path/to/audio.mp4"
```

