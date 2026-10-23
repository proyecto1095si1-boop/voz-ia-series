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
        )

        console.print("\n[bold green]✓ Procesamiento completado[/bold green]")
        console.print(f"Resultados guardados en: [cyan]{result['output_dir']}[/cyan]")

        if result.get("srt_path"):
            console.print(f"  • Subtítulos SRT : {result['srt_path']}")
        if result.get("vtt_path"):
            console.print(f"  • Subtítulos VTT : {result['vtt_path']}")
        if result.get("json_path"):
            console.print(f"  • Datos JSON    : {result['json_path']}")
        if result.get("txt_path"):
            console.print(f"  • Texto legible : {result['txt_path']}")

    except Exception as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command("batch")
def batch(
    input_folder: Path = typer.Argument(
        ...,
        help="Carpeta que contiene los videos/audios a procesar",
    ),
    output_folder: Path = typer.Option(
        Path("output/batch"),
        "--output", "-o",
        help="Carpeta raíz donde se guardarán los resultados",
    ),
    source_lang: str = typer.Option("en", "--source-lang", "-s"),
    target_lang: str = typer.Option("es", "--target-lang", "-t"),
    model_size: str = typer.Option("large-v3", "--model"),
    device: str = typer.Option("auto", "--device"),
    recursive: bool = typer.Option(
        True,
        "--recursive/--no-recursive",
        help="Buscar archivos en subcarpetas",
    ),
    keep_audio: bool = typer.Option(False, "--keep-audio"),
):
    """
    Procesa todos los videos/audios de una carpeta (ideal para series completas).
    """
    _print_banner()
    console.print(f"[bold]Modo lote[/bold] — Carpeta: [cyan]{input_folder}[/cyan]\n")

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
            console.print(
                f"\n[bold yellow]⚠ Lote terminado con {summary['failed']} error(es)[/bold yellow]"
            )
        else:
            console.print("\n[bold red]✗ No se pudo procesar ningún archivo[/bold red]")
            raise typer.Exit(code=1)

    except Exception as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command("gui")
def gui(
    port: int = typer.Option(7860, "--port", "-p", help="Puerto del servidor web"),
    share: bool = typer.Option(
        False,
        "--share",
        help="Crear un enlace público temporal (Gradio share)",
    ),
):
    """
    Abre la interfaz gráfica web (ideal para usuarios no técnicos).
    """
    _print_banner()
    console.print(f"Iniciando interfaz gráfica en el puerto [cyan]{port}[/cyan]...\n")

    try:
        from voxsync.gui import launch

        launch(share=share, server_port=port)
    except ImportError:
        console.print(
            "[bold red]Gradio no está instalado.[/bold red]\n"
            "Instalalo con: [cyan]pip install gradio[/cyan]"
        )
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]✗ Error al iniciar la GUI:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command("version")
def version():
    """Muestra la versión de VoxSync."""
    console.print(f"{__product_name__} v{__version__}")


if __name__ == "__main__":
    app()
