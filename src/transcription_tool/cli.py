"""CLI interface for the transcription tool with FFmpeg audio preprocessing support."""

import sys
from pathlib import Path

import click

from dotenv import load_dotenv

from .transcription import transcribe_audio
from .summarization import summarize_text
from .pipeline import run_pipeline, run_diarization_pipeline
from .diarization import diagnose_audio_compatibility
from .audio_preprocessing import probe_ffmpeg_availability

load_dotenv()


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
    default="mlx-community/Qwen3-30B-A3B-8bit",
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
    default="mlx-community/Qwen3-30B-A3B-8bit",
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


@main.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option(
    "--transcription-model",
    default="mlx-community/whisper-large-v3-mlx",
    help="MLX Whisper model to use",
)
@click.option(
    "--diarization-model",
    default="pyannote/speaker-diarization-3.1",
    help="Pyannote diarization model to use",
)
@click.option(
    "--summarization-model",
    default="mlx-community/Qwen3-30B-A3B-8bit",
    help="MLX LM model to use",
)
@click.option(
    "--hf-token",
    envvar="HUGGINGFACE_TOKEN",
    help="HuggingFace access token (or set HUGGINGFACE_TOKEN env var)",
)
@click.option("--num-speakers", type=int, help="Known number of speakers")
@click.option("--min-speakers", type=int, help="Minimum number of speakers")
@click.option("--max-speakers", type=int, help="Maximum number of speakers")
@click.option("--output", "-o", type=click.Path(), help="Output file path for summary")
@click.option(
    "--save-transcription", type=click.Path(), help="Save speaker-attributed transcript"
)
@click.option("--save-rttm", type=click.Path(), help="Save diarization in RTTM format")
@click.option(
    "--max-tokens", default=32000, type=int, help="Maximum tokens for summary"
)
@click.option(
    "--structured/--simple",
    default=True,
    help="Use structured meeting notes format",
)
@click.option(
    "--meeting-type",
    type=click.Choice(
        ["general", "standup", "planning", "client_call", "interview", "retrospective"]
    ),
    default="general",
    help="Type of meeting for specialized formatting",
)
@click.option(
    "--check-compatibility",
    is_flag=True,
    help="Check audio file compatibility before processing",
)
def diarize(
    file_path: str,
    transcription_model: str,
    diarization_model: str,
    summarization_model: str,
    hf_token: str,
    num_speakers: int,
    min_speakers: int,
    max_speakers: int,
    output: str,
    save_transcription: str,
    save_rttm: str,
    max_tokens: int,
    structured: bool,
    meeting_type: str,
    check_compatibility: bool,
):
    """Complete diarization pipeline: transcription + speaker identification + summarization."""
    try:
        # Check audio compatibility if requested
        if check_compatibility:
            click.echo("Checking audio file compatibility...")
            diagnosis = diagnose_audio_compatibility(file_path)

            click.echo(f"File: {diagnosis['file_path']}")
            click.echo(f"Format: {Path(file_path).suffix}")

            for rec in diagnosis["recommendations"]:
                click.echo(rec)

            if (
                not diagnosis["validation"]["is_valid"]
                and not diagnosis["validation"]["needs_conversion"]
            ):
                click.echo("\nAudio file is not compatible with diarization.", err=True)
                sys.exit(1)

            click.echo("")  # Add spacing

        # Check FFmpeg availability
        ffmpeg_status = probe_ffmpeg_availability()
        if not ffmpeg_status["available"]:
            click.echo(
                "Warning: FFmpeg not available for audio preprocessing.\n"
                f"Error: {ffmpeg_status.get('error', 'Unknown error')}\n"
                "Install FFmpeg: https://ffmpeg.org/download.html\n"
                "Note: Audio files must be in WAV format without FFmpeg.\n",
                err=True,
            )

        if not hf_token:
            click.echo(
                "Error: HuggingFace token required for diarization.\n"
                "Get token at https://hf.co/settings/tokens\n"
                "Add HUGGINGFACE_TOKEN=your_token to .env file or use --hf-token option.",
                err=True,
            )
            sys.exit(1)

        # Generate automatic output paths if not specified
        if not output:
            input_path = Path(file_path)
            summaries_dir = Path("./summaries")
            summaries_dir.mkdir(exist_ok=True)
            extension = ".md" if structured else ".txt"
            output = summaries_dir / f"{input_path.stem}_diarized_summary{extension}"

        if not save_transcription:
            input_path = Path(file_path)
            transcriptions_dir = Path("./transcriptions")
            transcriptions_dir.mkdir(exist_ok=True)
            save_transcription = transcriptions_dir / f"{input_path.stem}_speakers.txt"

        if not save_rttm:
            input_path = Path(file_path)
            transcriptions_dir = Path("./transcriptions")
            transcriptions_dir.mkdir(exist_ok=True)
            save_rttm = transcriptions_dir / f"{input_path.stem}.rttm"

        click.echo("Running diarization pipeline...")
        file_format = Path(file_path).suffix.lower()
        if file_format in [".mp3", ".m4a", ".mp4"]:
            click.echo(
                f"Note: {file_format} format detected. Will convert to WAV for diarization."
            )
        click.echo("This may take several minutes for longer audio files.")

        transcript, summary, aligned_segments = run_diarization_pipeline(
            file_path=file_path,
            transcription_model=transcription_model,
            diarization_model=diarization_model,
            summarization_model=summarization_model,
            use_auth_token=hf_token,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            output_file=output,
            save_transcription=save_transcription,
            save_rttm=save_rttm,
            max_tokens=max_tokens,
            structured=structured,
            meeting_type=meeting_type,
        )

        # Display results
        click.echo(f"\nSpeaker-attributed transcript saved to: {save_transcription}")
        click.echo(f"Diarization (RTTM) saved to: {save_rttm}")
        click.echo(f"Summary saved to: {output}")

        # Show speaker statistics
        speakers = set(seg["speaker"] for seg in aligned_segments)
        click.echo(
            f"\nIdentified {len(speakers)} speakers: {', '.join(sorted(speakers))}"
        )

        # Display the summary
        click.echo("\n" + "=" * 60)
        click.echo("SPEAKER-ATTRIBUTED MEETING NOTES" if structured else "SUMMARY")
        click.echo("=" * 60)
        click.echo(summary)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument("file_path", type=click.Path(exists=True))
def diagnose(file_path: str):
    """Diagnose audio file compatibility with diarization pipeline."""
    try:
        click.echo("Diagnosing audio file compatibility...\n")

        diagnosis = diagnose_audio_compatibility(file_path)

        click.echo(f"📁 File: {diagnosis['file_path']}")
        click.echo(f"📋 Format: {Path(file_path).suffix}")

        # Show validation results
        validation = diagnosis["validation"]
        click.echo(f"✓ Valid: {'Yes' if validation['is_valid'] else 'No'}")
        click.echo(
            f"🔄 Needs conversion: {'Yes' if validation['needs_conversion'] else 'No'}"
        )

        # Show issues if any
        if validation["issues"]:
            click.echo("\n⚠️  Issues found:")
            for issue in validation["issues"]:
                click.echo(f"   • {issue}")

        # Show FFmpeg availability
        ffmpeg_status = diagnosis["ffmpeg_available"]
        click.echo(f"\n🔧 FFmpeg status:")
        click.echo(f"   • Available: {'✓' if ffmpeg_status['available'] else '✗'}")
        if not ffmpeg_status["available"] and ffmpeg_status.get("error"):
            click.echo(f"   • Error: {ffmpeg_status['error']}")

        # Show recommendations
        click.echo("\n💡 Recommendations:")
        for rec in diagnosis["recommendations"]:
            click.echo(f"   {rec}")

        # Final assessment
        can_process = validation["is_valid"] or (
            validation["needs_conversion"] and ffmpeg_status["available"]
        )

        if can_process:
            click.echo(f"\n✅ File can be processed with diarization pipeline")
        else:
            click.echo(f"\n❌ File cannot be processed - see recommendations above")
            sys.exit(1)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
def check_deps():
    """Check system dependencies for audio processing."""
    try:
        click.echo("Checking system dependencies...\n")

        # Check FFmpeg
        ffmpeg_status = probe_ffmpeg_availability()
        click.echo("🔧 Audio Preprocessing:")
        if ffmpeg_status["available"]:
            click.echo("   • FFmpeg: ✓ Available")
        else:
            click.echo("   • FFmpeg: ✗ Not available")
            if ffmpeg_status.get("error"):
                click.echo(f"     Error: {ffmpeg_status['error']}")

        # Check MLX dependencies (basic check)
        try:
            import mlx_whisper

            click.echo("🎤 MLX Whisper: ✓ Available")
        except ImportError:
            click.echo("🎤 MLX Whisper: ✗ Not installed")

        try:
            import mlx_lm

            click.echo("🧠 MLX LM: ✓ Available")
        except ImportError:
            click.echo("🧠 MLX LM: ✗ Not installed")

        try:
            import pyannote.audio

            click.echo("👥 Pyannote Audio: ✓ Available")
        except ImportError:
            click.echo("👥 Pyannote Audio: ✗ Not installed")

        # Recommendations
        click.echo("\n💡 Installation recommendations:")
        if not ffmpeg_status["available"]:
            click.echo("   • Install FFmpeg: https://ffmpeg.org/download.html")
            click.echo("   • macOS: brew install ffmpeg")
            click.echo("   • Ubuntu: sudo apt install ffmpeg")

        click.echo("\n✅ Dependency check complete")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
