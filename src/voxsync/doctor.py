"""
Diagnóstico del sistema: verifica dependencias y configuración
antes de procesar videos (muy útil para soporte comercial).
"""

from __future__ import annotations

from typing import List, Tuple
import shutil
import os

from rich.console import Console
from rich.table import Table

console = Console()


def _check(name: str, ok: bool, detail: str = "") -> Tuple[str, str, str]:
    status = "[green]✓ OK[/green]" if ok else "[red]✗ FALTA[/red]"
    return name, status, detail


def run_doctor() -> bool:
    """
    Ejecuta todas las comprobaciones.
    Returns True si el sistema está listo para usar VoxSync.
    """
    rows: List[Tuple[str, str, str]] = []
    critical_ok = True

    # Python
    import sys

    py_ok = sys.version_info >= (3, 10)
    rows.append(
        _check(
            "Python",
            py_ok,
            f"{sys.version.split()[0]} (se necesita 3.10+)",
        )
    )
    if not py_ok:
        critical_ok = False

    # FFmpeg
    ffmpeg = shutil.which("ffmpeg")
    rows.append(_check("FFmpeg", ffmpeg is not None, ffmpeg or "No está en el PATH"))
    if not ffmpeg:
        critical_ok = False

    # PyTorch
    try:
        import torch

        cuda = torch.cuda.is_available()
        device_info = f"CUDA disponible ({torch.cuda.get_device_name(0)})" if cuda else "Solo CPU"
        rows.append(_check("PyTorch", True, f"{torch.__version__} — {device_info}"))
    except ImportError:
        rows.append(_check("PyTorch", False, "pip install torch torchaudio"))
        critical_ok = False

    # WhisperX
    try:
        import whisperx  # noqa: F401

        rows.append(_check("WhisperX", True, "Instalado"))
    except ImportError:
        rows.append(_check("WhisperX", False, "pip install whisperx"))
        critical_ok = False

    # transformers
    try:
        import transformers

        rows.append(_check("Transformers", True, transformers.__version__))
    except ImportError:
        rows.append(_check("Transformers", False, "pip install transformers"))
        critical_ok = False

    # librosa
    try:
        import librosa  # noqa: F401

        rows.append(_check("Librosa", True, "Instalado (atributos de voz)"))
    except ImportError:
        rows.append(_check("Librosa", False, "pip install librosa (recomendado)"))

    # Gradio (opcional)
    try:
        import gradio  # noqa: F401

        rows.append(_check("Gradio", True, "GUI disponible"))
    except ImportError:
        rows.append(_check("Gradio", False, "pip install gradio (opcional para GUI)"))

    # HF Token
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    rows.append(
        _check(
            "HF_TOKEN",
            bool(token),
            "Configurado" if token else "Necesario para diarización (pyannote)",
        )
    )

    # yt-dlp
    try:
        import yt_dlp  # noqa: F401

        rows.append(_check("yt-dlp", True, "YouTube OK"))
    except ImportError:
        rows.append(_check("yt-dlp", False, "pip install yt-dlp"))
        critical_ok = False

    table = Table(title="VoxSync — Diagnóstico del sistema")
    table.add_column("Componente", style="cyan")
    table.add_column("Estado")
    table.add_column("Detalle", style="dim")

    for name, status, detail in rows:
        table.add_row(name, status, detail)

    console.print(table)

    if critical_ok:
        console.print("\n[bold green]✓ Sistema listo para usar VoxSync[/bold green]")
    else:
        console.print(
            "\n[bold yellow]⚠ Hay componentes críticos faltantes. "
            "Revisá los marcados en rojo.[/bold yellow]"
        )

    return critical_ok
