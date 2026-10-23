"""
Módulo de descarga de videos (YouTube y otros soportados por yt-dlp).
"""

from pathlib import Path
from urllib.parse import urlparse

import yt_dlp


def is_url(path: str) -> bool:
    try:
        result = urlparse(path)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def download_if_needed(input_path: str, output_dir: Path) -> Path:
    """
    Si la entrada es una URL, descarga el video.
    Si es un archivo local, lo devuelve tal cual.
    """
    path = Path(input_path)

    if path.exists() and path.is_file():
        return path.resolve()

    if not is_url(input_path):
        raise FileNotFoundError(f"No se encontró el archivo: {input_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": str(output_dir / "%(title)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(input_path, download=True)
        filename = ydl.prepare_filename(info)

    return Path(filename).resolve()
