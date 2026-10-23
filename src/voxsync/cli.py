"""
VoxSync CLI — Interfaz de línea de comandos profesional.
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from voxsync import __version__, __product_name__
from voxsync.pipeline import run_pipeline

app = typer.Typer(
    name="voxsync",
    help="VoxSync — Transcripción + diarización + traducción profesional de diálogos (100% local)",
    add_completion=False,
)
console = Console()


@app.command()
def process(
    input_path: str = typer.Argument(
        ...,
        help="Ruta a un video/audio local o URL de YouTube",
    ),
    output_dir: Path = typer.Option(
        Path("output"),
        "--output", "-o",
        help="Carpeta donde se guardarán los resultados",
    ),
    source_lang: str = typer.Option(
        "en",
        "--source-lang", "-s",
        help="Idioma de origen (código ISO, ej: en, es, fr)",
    ),
    target_lang: str = typer.Option(
        "es",
        "--target-lang", "-t",
        help="Idioma de destino (código ISO)",
    ),
    model_size: str = typer.Option(
        "large-v3",
        "--model",
        help="Tamaño del modelo Whisper (tiny, base, small, medium, large-v2, large-v3)",
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Dispositivo: auto, cpu, cuda",
    ),
    keep_audio: bool = typer.Option(
        False,
        "--keep-audio",
        help="Conservar el archivo de audio extraído",
    ),
):
    """
    Procesa un video o audio: transcribe, diariza, detecta atributos de hablantes y traduce.
    """
    console.print(
        Panel.fit(
            f"[bold cyan]{__product_name__}[/bold cyan] v{__version__}\n"
            "Transcripción · Diarización · Traducción · 100% local",
            border_style="cyan",
        )
    )

    try:
        result = run_pipeline(
            input_path=input_path,
            output_dir=output_dir,
            source_lang=source_lang,
            target_lang=target_lang,
            model_size=model_size,
            device=device,
            keep_audio=keep_audio,
        )

        console.print("\n[bold green]✓ Procesamiento completado[/bold green]")
        console.print(f"Resultados guardados en: [cyan]{result['output_dir']}[/cyan]")

        if result.get("srt_path"):
            console.print(f"  • Subtítulos SRT : {result['srt_path']}")
        if result.get("json_path"):
            console.print(f"  • Datos JSON    : {result['json_path']}")
        if result.get("txt_path"):
            console.print(f"  • Texto legible : {result['txt_path']}")

    except Exception as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command()
def version():
    """Muestra la versión de VoxSync."""
    console.print(f"{__product_name__} v{__version__}")


if __name__ == "__main__":
    app()
