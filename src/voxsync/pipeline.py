"""
Pipeline principal de VoxSync.
Orquesta: descarga → audio → transcripción + diarización → atributos → traducción → exportación.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from voxsync.downloader import download_if_needed
from voxsync.audio import extract_audio
from voxsync.transcriber import transcribe_and_diarize
from voxsync.speaker_attrs import enrich_speakers
from voxsync.translator import translate_segments
from voxsync.formatter import export_results

console = Console()


def run_pipeline(
    input_path: str,
    output_dir: Path,
    source_lang: str = "en",
    target_lang: str = "es",
    model_size: str = "large-v3",
    device: str = "auto",
    keep_audio: bool = False,
) -> Dict[str, Any]:
    """
    Ejecuta el pipeline completo de VoxSync.

    Returns:
        Diccionario con rutas de los archivos generados y metadatos.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:

        # 1. Descargar si es URL
        task = progress.add_task("Preparando entrada...", total=None)
        video_path = download_if_needed(input_path, output_dir)
        progress.update(task, completed=True)

        # 2. Extraer audio
        task = progress.add_task("Extrayendo audio...", total=None)
        audio_path = extract_audio(video_path, output_dir)
        progress.update(task, completed=True)

        # 3. Transcribir + diarizar
        task = progress.add_task("Transcribiendo y diarizando (esto puede tardar)...", total=None)
        segments = transcribe_and_diarize(
            audio_path=audio_path,
            model_size=model_size,
            device=device,
            language=source_lang,
        )
        progress.update(task, completed=True)

        # 4. Enriquecer con atributos de hablante (género / edad)
        task = progress.add_task("Analizando atributos de voz...", total=None)
        segments = enrich_speakers(segments, audio_path)
        progress.update(task, completed=True)

        # 5. Traducir
        task = progress.add_task(f"Traduciendo {source_lang} → {target_lang}...", total=None)
        segments = translate_segments(segments, source_lang=source_lang, target_lang=target_lang)
        progress.update(task, completed=True)

        # 6. Exportar resultados
        task = progress.add_task("Generando archivos de salida...", total=None)
        results = export_results(
            segments=segments,
            output_dir=output_dir,
            base_name=video_path.stem,
            source_lang=source_lang,
            target_lang=target_lang,
        )
        progress.update(task, completed=True)

    # Limpieza opcional
    if not keep_audio and audio_path.exists() and audio_path.suffix == ".wav":
        try:
            audio_path.unlink()
        except Exception:
            pass

    results["output_dir"] = str(output_dir)
    return results
