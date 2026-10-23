"""
Transcripción + diarización usando WhisperX (local).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


def get_device(preferred: str = "auto") -> str:
    if preferred == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return preferred


def transcribe_and_diarize(
    audio_path: Path,
    model_size: str = "large-v3",
    device: str = "auto",
    language: Optional[str] = None,
    batch_size: int = 16,
) -> List[Dict[str, Any]]:
    """
    Transcribe el audio y realiza diarización de hablantes.

    Returns:
        Lista de segmentos con:
        - start, end (segundos)
        - text
        - speaker (ej: SPEAKER_00)
    """
    try:
        import whisperx
    except ImportError as e:
        raise ImportError(
            "WhisperX no está instalado. Ejecuta:\n"
            "pip install whisperx"
        ) from e

    device = get_device(device)
    compute_type = "float16" if device == "cuda" else "int8"

    # 1. Cargar modelo de transcripción
    model = whisperx.load_model(
        model_size,
        device=device,
        compute_type=compute_type,
        language=language,
    )

    # 2. Transcribir
    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio, batch_size=batch_size)

    # 3. Alinear (mejora timestamps)
    model_a, metadata = whisperx.load_align_model(
        language_code=result.get("language", language or "en"),
        device=device,
    )
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    # 4. Diarización
    # Nota: pyannote requiere token de Hugging Face (gratis)
    # El usuario debe configurar HF_TOKEN en el entorno.
    try:
        diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=True,  # usa HF_TOKEN del entorno
            device=device,
        )
        diarize_segments = diarize_model(str(audio_path))
        result = whisperx.assign_word_speakers(diarize_segments, result)
    except Exception as e:
        # Si falla la diarización, continuamos sin ella
        print(f"[Advertencia] Diarización no disponible: {e}")
        for seg in result["segments"]:
            seg["speaker"] = "SPEAKER_00"

    # Normalizar salida
    segments: List[Dict[str, Any]] = []
    for seg in result["segments"]:
        segments.append({
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "text": seg["text"].strip(),
            "speaker": seg.get("speaker", "SPEAKER_00"),
        })

    return segments
