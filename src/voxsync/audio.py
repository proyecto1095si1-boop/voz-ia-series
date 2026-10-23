"""
Extracción y normalización de audio.
"""

from pathlib import Path
import subprocess
import shutil


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "FFmpeg no está instalado o no se encuentra en el PATH.\n"
            "Instálalo desde https://ffmpeg.org/download.html"
        )


def extract_audio(video_path: Path, output_dir: Path, sample_rate: int = 16000) -> Path:
    """
    Extrae el audio de un video y lo convierte a WAV mono 16kHz
    (formato ideal para Whisper y diarización).
    """
    check_ffmpeg()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_path = output_dir / f"{video_path.stem}.wav"

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        str(audio_path),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Error al extraer audio:\n{result.stderr}")

    return audio_path.resolve()
