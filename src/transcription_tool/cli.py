"""CLI interface for the transcription tool."""

import sys
from pathlib import Path

import click

from .transcription import transcribe_audio
from .summarization import summarize_text
from .pipeline import run_pipeline


@click.group()
def main():
    """Transcription tool with MLX backend for audio/video processing."""
    pass


@main.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option(
    "--model",
    "-m",
    default="mlx-community/whisper-large-v3-mlx",
    help="MLX Whisper model to use",
)
@click.option(
    "--output", "-o", type=click.Path(), help="Output file path for transcription"
)
def transcribe(file_path: str, model: str, output: str):
    """Transcribe audio/video file to text."""
    try:
        # Generate automatic output path if not specified
        if not output:
            input_path = Path(file_path)
            transcriptions_dir = Path("./transcriptions")
            transcriptions_dir.mkdir(exist_ok=True)
            output = transcriptions_dir / f"{input_path.stem}.txt"

        transcription = transcribe_audio(
            file_path=file_path,
            model=model,
            output_file=output,
        )

        # Always output to stdout and save to file
        click.echo(transcription)
        click.echo(f"Transcription saved to: {output}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument("text_input")
@click.option(
    "--model",
    "-m",
    default="mlx-community/Qwen3-30B-A3B-MLX-8bit",
    help="MLX LM model to use",
)
@click.option("--output", "-o", type=click.Path(), help="Output file path for summary")
@click.option(
    "--max-tokens", default=32000, type=int, help="Maximum tokens for summary"
)
@click.option(
    "--structured/--simple",
    default=True,
    help="Use structured meeting notes format (default) or simple summary",
)
@click.option(
    "--meeting-type",
    type=click.Choice(
        ["general", "standup", "planning", "client_call", "interview", "retrospective"]
    ),
    default="general",
    help="Type of meeting for specialized formatting",
)
def summarize(
    text_input: str,
    model: str,
    output: str,
    max_tokens: int,
    structured: bool,
    meeting_type: str,
):
    """Summarize text from file or stdin with structured meeting notes."""
    try:
        # Generate automatic output path if not specified
        if not output:
            if text_input != "-" and Path(text_input).exists():
                # Use input file name for output
                input_path = Path(text_input)
                summaries_dir = Path("./summaries")
                summaries_dir.mkdir(exist_ok=True)
                # Use .md extension for structured output
                extension = ".md" if structured else ".txt"
                output = summaries_dir / f"{input_path.stem}{extension}"
            else:
                # For stdin or direct text, use timestamp
                from datetime import datetime

                summaries_dir = Path("./summaries")
                summaries_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                extension = ".md" if structured else ".txt"
                output = summaries_dir / f"summary_{timestamp}{extension}"

        summary = summarize_text(
            text_input=text_input,
            model=model,
            output_file=output,
            max_tokens=max_tokens,
            structured=structured,
            meeting_type=meeting_type,
        )

        # Always output to stdout and save to file
        click.echo(summary)
        click.echo(f"Summary saved to: {output}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option(
    "--transcription-model",
    default="mlx-community/whisper-large-v3-mlx",
    help="MLX Whisper model to use",
)
@click.option(
    "--summarization-model",
    default="mlx-community/Qwen3-30B-A3B-MLX-8bit",
    help="MLX LM model to use",
)
@click.option(
    "--output", "-o", type=click.Path(), help="Output file path for final summary"
)
@click.option(
    "--save-transcription",
    type=click.Path(),
    help="Save intermediate transcription to file",
)
@click.option(
    "--max-tokens", default=32000, type=int, help="Maximum tokens for summary"
)
@click.option(
    "--structured/--simple",
    default=True,
    help="Use structured meeting notes format (default) or simple summary",
)
@click.option(
    "--meeting-type",
    type=click.Choice(
        ["general", "standup", "planning", "client_call", "interview", "retrospective"]
    ),
    default="general",
    help="Type of meeting for specialized formatting",
)
def pipeline(
    file_path: str,
    transcription_model: str,
    summarization_model: str,
    output: str,
    save_transcription: str,
    max_tokens: int,
    structured: bool,
    meeting_type: str,
):
    """Run complete pipeline: transcription + summarization."""
    try:
        # Generate automatic output paths if not specified
        if not output:
            input_path = Path(file_path)
            summaries_dir = Path("./summaries")
            summaries_dir.mkdir(exist_ok=True)
            # Use .md extension for structured output
            extension = ".md" if structured else ".txt"
            output = summaries_dir / f"{input_path.stem}_summary{extension}"

        if not save_transcription:
            input_path = Path(file_path)
            transcriptions_dir = Path("./transcriptions")
            transcriptions_dir.mkdir(exist_ok=True)
            save_transcription = transcriptions_dir / f"{input_path.stem}.txt"

        transcription, summary = run_pipeline(
            file_path=file_path,
            transcription_model=transcription_model,
            summarization_model=summarization_model,
            output_file=output,
            save_transcription=save_transcription,
            max_tokens=max_tokens,
            structured=structured,
            meeting_type=meeting_type,
        )

        click.echo(f"Transcription saved to: {save_transcription}")
        click.echo(f"Summary saved to: {output}")

        # Also display the summary
        click.echo("\n" + "=" * 50)
        click.echo("MEETING NOTES" if structured else "SUMMARY")
        click.echo("=" * 50)
        click.echo(summary)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
