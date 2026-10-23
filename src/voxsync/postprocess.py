"""
Post-procesado de segmentos para mejorar el hilo de la conversación.

- Une segmentos consecutivos del mismo hablante
- Limpia espacios y puntuación
- Evita cortes artificiales en el diálogo
"""

from typing import Any, Dict, List
import re


def _clean_text(text: str) -> str:
    """Limpieza básica del texto."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    # Evitar espacios antes de puntuación
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return text


def merge_consecutive_segments(
    segments: List[Dict[str, Any]],
    max_gap: float = 1.2,
) -> List[Dict[str, Any]]:
    """
    Une segmentos consecutivos del mismo hablante si el hueco entre ellos
    es menor a `max_gap` segundos.

    Esto evita que una misma frase quede cortada en varios subtítulos
    y mantiene mejor el hilo de la conversación.
    """
    if not segments:
        return []

    merged: List[Dict[str, Any]] = []
    current = dict(segments[0])
    current["text"] = _clean_text(current.get("text", ""))

    for seg in segments[1:]:
        same_speaker = seg.get("speaker") == current.get("speaker")
        gap = seg["start"] - current["end"]

        if same_speaker and gap <= max_gap:
            # Unir
            current["end"] = seg["end"]
            extra = _clean_text(seg.get("text", ""))
            if extra:
                if not current["text"].endswith(extra):
                    current["text"] = (current["text"] + " " + extra).strip()

            if "translation" in seg and seg["translation"]:
                if "translation" not in current or not current["translation"]:
                    current["translation"] = seg["translation"]
                else:
                    current["translation"] = (
                        current["translation"] + " " + seg["translation"]
                    ).strip()
        else:
            merged.append(current)
            current = dict(seg)
            current["text"] = _clean_text(current.get("text", ""))

    merged.append(current)
    return merged


def smooth_conversation(
    segments: List[Dict[str, Any]],
    max_gap: float = 1.2,
) -> List[Dict[str, Any]]:
    """
    Aplica todas las mejoras de post-procesado orientadas a mantener
    el hilo de la conversación.
    """
    segments = merge_consecutive_segments(segments, max_gap=max_gap)

    for seg in segments:
        seg["text"] = _clean_text(seg.get("text", ""))
        if "translation" in seg and seg["translation"]:
            seg["translation"] = _clean_text(seg["translation"])

    return segments
