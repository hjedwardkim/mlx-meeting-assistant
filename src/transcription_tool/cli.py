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
    # default="mlx-community/parakeet-tdt-0.6b-v2",
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
            file_path=file_path, model=model, output_file=output
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
    default="mlx-community/Qwen3-30B-A3B-8bit",
    help="MLX LM model to use",
)
@click.option("--output", "-o", type=click.Path(), help="Output file path for summary")
@click.option(
    "--max-tokens", default=10000, type=int, help="Maximum tokens for summary"
)
def summarize(text_input: str, model: str, output: str, max_tokens: int):
    """Summarize text from file or stdin."""
    try:
        # Generate automatic output path if not specified
        if not output:
            if text_input != "-" and Path(text_input).exists():
                # Use input file name for output
                input_path = Path(text_input)
                summaries_dir = Path("./summaries")
                summaries_dir.mkdir(exist_ok=True)
                output = summaries_dir / f"{input_path.stem}.txt"
            else:
                # For stdin or direct text, use timestamp
                from datetime import datetime

                summaries_dir = Path("./summaries")
                summaries_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output = summaries_dir / f"summary_{timestamp}.txt"

        summary = summarize_text(
            text_input=text_input,
            model=model,
            output_file=output,
            max_tokens=max_tokens,
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
@click.option("--max-tokens", default=500, type=int, help="Maximum tokens for summary")
def pipeline(
    file_path: str,
    transcription_model: str,
    summarization_model: str,
    output: str,
    save_transcription: str,
    max_tokens: int,
):
    """Run complete pipeline: transcription + summarization."""
    try:
        transcription, summary = run_pipeline(
            file_path=file_path,
            transcription_model=transcription_model,
            summarization_model=summarization_model,
            output_file=output,
            save_transcription=save_transcription,
            max_tokens=max_tokens,
        )

        if save_transcription:
            click.echo(f"Transcription saved to: {save_transcription}")

        if not output:
            click.echo("--- SUMMARY ---")
            click.echo(summary)
        else:
            click.echo(f"Summary saved to: {output}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
