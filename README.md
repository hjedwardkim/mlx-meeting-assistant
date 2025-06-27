# Transcription Tool

A CLI tool for audio/video transcription and summarization using MLX backend.

## Installation

Install the tool using uv:

```bash
uv sync
```

The CLI entry point `transcription-tool` will be available after installation.

## Commands

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

## Supported Formats

- **Input**: MPEG-4 and other common audio/video formats supported by MLX Whisper
- **Output**: Plain text transcriptions and summaries

## Models

### Default Models

- **Transcription**: `mlx-community/parakeet-tdt-0.6b-v2`
- **Summarization**: `lmstudio-community/Qwen3-30B-A3B-MLX-8bit`

### Custom Models

All commands support custom model selection via `--model`, `--transcription-model`, and `--summarization-model` flags. Models must be compatible with MLX Whisper and MLX LM respectively.

## Error Handling

The tool provides clear error messages for:

- Invalid file paths or formats
- Model loading failures
- MLX backend issues
- File I/O operations

## Development

Run the tool in development mode:

```bash
uv run python -m transcription_tool.cli --help
```

## Requirements

- Python 3.12+
- MLX-compatible hardware (Apple Silicon recommended)
- Required dependencies are automatically installed via uv

