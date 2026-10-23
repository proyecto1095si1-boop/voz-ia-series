"""
VoxSync CLI — Interfaz de línea de comandos profesional.
"""

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from voxsync import __version__, __product_name__
from voxsync.pipeline import run_pipeline
from voxsync.batch import process_batch

app = typer.Typer(
    name="voxsync",
    help="VoxSync — Transcripción + diarización + traducción profesional de diálogos (100% local)",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


def _print_banner() -> None:
    console.print(
        Panel.fit(
            f"[bold cyan]{__product_name__}[/bold cyan] v{__version__}\n"
            "Transcripción · Diarización · Traducción · 100% local",
            border_style="cyan",
        )
    )


@app.command("process")
def process(
    input_path: str = typer.Argument(..., help="Video/audio local o URL de YouTube"),
    output_dir: Path = typer.Option(Path("output"), "--output", "-o"),
    source_lang: str = typer.Option("en", "--source-lang", "-s"),
    target_lang: str = typer.Option("es", "--target-lang", "-t"),
    model_size: str = typer.Option("large-v3", "--model"),
    device: str = typer.Option("auto", "--device"),
    keep_audio: bool = typer.Option(False, "--keep-audio"),
    bilingual: bool = typer.Option(
        False,
        "--bilingual",
        help="Subtítulos con original + traducción en cada línea",
    ),
    no_speaker: bool = typer.Option(
        False,
        "--no-speaker",
        help="No mostrar etiqueta de hablante en los subtítulos",
    ),
):
    """Procesa un video o audio: transcribe, diariza, detecta atributos y traduce."""
    _print_banner()

    try:
        result = run_pipeline(
            input_path=input_path,
            output_dir=output_dir,
            source_lang=source_lang,
            target_lang=target_lang,
            model_size=model_size,
            device=device,
            keep_audio=keep_audio,
            bilingual=bilingual,
            show_speaker=not no_speaker,
        )

        console.print("\n[bold green]✓ Procesamiento completado[/bold green]")
        console.print(f"Resultados en: [cyan]{result['output_dir']}[/cyan]")
        for key, label in [
            ("srt_path", "SRT traducido"),
            ("srt_original_path", "SRT original"),
            ("vtt_path", "VTT"),
            ("json_path", "JSON"),
            ("txt_path", "Guion TXT"),
        ]:
            if result.get(key):
                console.print(f"  • {label}: {result[key]}")

    except Exception as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command("batch")
def batch(
    input_folder: Path = typer.Argument(..., help="Carpeta con videos/audios"),
    output_folder: Path = typer.Option(Path("output/batch"), "--output", "-o"),
    source_lang: str = typer.Option("en", "--source-lang", "-s"),
    target_lang: str = typer.Option("es", "--target-lang", "-t"),
    model_size: str = typer.Option("large-v3", "--model"),
    device: str = typer.Option("auto", "--device"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive"),
    keep_audio: bool = typer.Option(False, "--keep-audio"),
):
    """Procesa todos los videos/audios de una carpeta (series completas)."""
    _print_banner()
    console.print(f"[bold]Modo lote[/bold] — [cyan]{input_folder}[/cyan]\n")

    try:
        summary = process_batch(
            input_folder=input_folder,
            output_folder=output_folder,
            source_lang=source_lang,
            target_lang=target_lang,
            model_size=model_size,
            device=device,
            recursive=recursive,
            keep_audio=keep_audio,
        )

        if summary["failed"] == 0 and summary["success"] > 0:
            console.print("\n[bold green]✓ Lote completado sin errores[/bold green]")
        elif summary["success"] > 0:
            console.print(f"\n[bold yellow]⚠ Terminado con {summary['failed']} error(es)[/bold yellow]")
        else:
            console.print("\n[bold red]✗ No se procesó ningún archivo[/bold red]")
            raise typer.Exit(code=1)

    except Exception as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command("gui")
def gui(
    port: int = typer.Option(7860, "--port", "-p"),
    share: bool = typer.Option(False, "--share", help="Enlace público temporal"),
):
    """Abre la interfaz gráfica web."""
    _print_banner()
    console.print(f"Iniciando GUI en puerto [cyan]{port}[/cyan]...\n")

    try:
        from voxsync.gui import launch

        launch(share=share, server_port=port)
    except ImportError:
        console.print("[bold red]Gradio no instalado.[/bold red] pip install gradio")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]✗ Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command("doctor")
def doctor():
    """Diagnostica el sistema (FFmpeg, GPU, modelos, token HF, etc.)."""
    _print_banner()
    from voxsync.doctor import run_doctor

    ok = run_doctor()
    if not ok:
        raise typer.Exit(code=1)


@app.command("version")
def version():
    """Muestra la versión de VoxSync."""
    console.print(f"{__product_name__} v{__version__}")


if __name__ == "__main__":
    app()
