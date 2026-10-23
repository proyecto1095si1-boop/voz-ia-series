"""
Pipeline principal de VoxSync.
Orquesta: descarga → audio → transcripción + diarización → atributos →
traducción → post-procesado (hilo de conversación) → exportación.
"""

from pathlib import Path
from typing import Any, Dict

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from voxsync.downloader import download_if_needed
from voxsync.audio import extract_audio
from voxsync.transcriber import transcribe_and_diarize
from voxsync.speaker_attrs import enrich_speakers
from voxsync.translator import translate_segments
from voxsync.postprocess import smooth_conversation
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
    bilingual: bool = False,
    show_speaker: bool = True,
) -> Dict[str, Any]:
    """Ejecuta el pipeline completo de VoxSync."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:

        task = progress.add_task("Preparando entrada...", total=None)
        video_path = download_if_needed(input_path, output_dir)
        progress.update(task, completed=True)

        task = progress.add_task("Extrayendo audio...", total=None)
        audio_path = extract_audio(video_path, output_dir)
        progress.update(task, completed=True)

        task = progress.add_task("Transcribiendo y diarizando (esto puede tardar)...", total=None)
        segments = transcribe_and_diarize(
            audio_path=audio_path,
            model_size=model_size,
            device=device,
            language=source_lang,
        )
        progress.update(task, completed=True)

        task = progress.add_task("Analizando atributos de voz (género + edad)...", total=None)
        segments = enrich_speakers(segments, audio_path, device=device)
        progress.update(task, completed=True)

        task = progress.add_task(f"Traduciendo {source_lang} → {target_lang}...", total=None)
        segments = translate_segments(segments, source_lang=source_lang, target_lang=target_lang)
        progress.update(task, completed=True)

        task = progress.add_task("Suavizando diálogos y uniendo segmentos...", total=None)
        segments = smooth_conversation(segments)
        progress.update(task, completed=True)

        task = progress.add_task("Generando archivos de salida...", total=None)
        results = export_results(
            segments=segments,
            output_dir=output_dir,
            base_name=video_path.stem,
            source_lang=source_lang,
            target_lang=target_lang,
            bilingual=bilingual,
            show_speaker=show_speaker,
        )
        progress.update(task, completed=True)

    if not keep_audio and audio_path.exists() and audio_path.suffix == ".wav":
        try:
            audio_path.unlink()
        except Exception:
            pass

    results["output_dir"] = str(output_dir)
    results["segments_count"] = len(segments)
    return results
