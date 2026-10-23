"""
Procesamiento por lotes (batch) de múltiples videos/episodios.
Ideal para series completas o carpetas de contenido.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
import json

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from voxsync.pipeline import run_pipeline

console = Console()

SUPPORTED_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".mp3", ".wav", ".m4a", ".flac"}


def find_media_files(folder: Path, recursive: bool = True) -> List[Path]:
    """Encuentra todos los archivos de video/audio en una carpeta."""
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"La carpeta no existe: {folder}")

    files = []
    pattern = "**/*" if recursive else "*"
    for path in folder.glob(pattern):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)

    return sorted(files)


def process_batch(
    input_folder: Path,
    output_folder: Path,
    source_lang: str = "en",
    target_lang: str = "es",
    model_size: str = "large-v3",
    device: str = "auto",
    recursive: bool = True,
    keep_audio: bool = False,
) -> Dict[str, Any]:
    """
    Procesa todos los videos/audios de una carpeta.

    Returns:
        Resumen del procesamiento (exitosos, fallidos, rutas, etc.)
    """
    files = find_media_files(input_folder, recursive=recursive)

    if not files:
        console.print("[yellow]No se encontraron archivos de media en la carpeta.[/yellow]")
        return {"total": 0, "success": 0, "failed": 0, "results": []}

    console.print(f"[bold]Se encontraron {len(files)} archivos para procesar.[/bold]\n")

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    results = []
    success = 0
    failed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Procesando lote...", total=len(files))

        for media_path in files:
            progress.update(task, description=f"Procesando: {media_path.name}")

            # Cada archivo tiene su propia subcarpeta de salida
            file_output = output_folder / media_path.stem

            try:
                result = run_pipeline(
                    input_path=str(media_path),
                    output_dir=file_output,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    model_size=model_size,
                    device=device,
                    keep_audio=keep_audio,
                )
                results.append({
                    "file": str(media_path),
                    "status": "success",
                    "output": result,
                })
                success += 1
            except Exception as e:
                results.append({
                    "file": str(media_path),
                    "status": "error",
                    "error": str(e),
                })
                failed += 1
                console.print(f"[red]✗ Error en {media_path.name}: {e}[/red]")

            progress.advance(task)

    # Resumen
    table = Table(title="Resumen del lote")
    table.add_column("Métrica", style="cyan")
    table.add_column("Valor", style="green")
    table.add_row("Total archivos", str(len(files)))
    table.add_row("Exitosos", str(success))
    table.add_row("Fallidos", str(failed))
    console.print(table)

    # Guardar log del lote
    log_path = output_folder / "batch_log.json"
    summary = {
        "total": len(files),
        "success": success,
        "failed": failed,
        "results": results,
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    console.print(f"\nLog del lote guardado en: [cyan]{log_path}[/cyan]")
    return summary
