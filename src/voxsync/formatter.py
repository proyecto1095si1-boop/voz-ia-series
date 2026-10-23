"""
Exportación de resultados a SRT, VTT, JSON y texto legible.
"""

from pathlib import Path
from typing import Any, Dict, List
import json
from datetime import timedelta


def _format_timestamp_srt(seconds: float) -> str:
    """Convierte segundos a formato SRT: 00:00:00,000"""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int((seconds - total_seconds) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_timestamp_vtt(seconds: float) -> str:
    """Convierte segundos a formato VTT: 00:00:00.000"""
    return _format_timestamp_srt(seconds).replace(",", ".")


def _speaker_label(seg: Dict[str, Any]) -> str:
    """Genera una etiqueta legible del hablante."""
    spk = seg.get("speaker", "SPEAKER_00")
    gender = seg.get("gender", "unknown")
    age = seg.get("age_group", "unknown")

    gender_map = {
        "male": "Hombre",
        "female": "Mujer",
        "unknown": "?",
    }
    age_map = {
        "child": "Niño/a",
        "teen": "Adolescente",
        "adult": "Adulto",
        "senior": "Mayor",
        "unknown": "?",
    }

    g = gender_map.get(gender, "?")
    a = age_map.get(age, "?")

    return f"{spk} ({g}, {a})"


def export_results(
    segments: List[Dict[str, Any]],
    output_dir: Path,
    base_name: str,
    source_lang: str = "en",
    target_lang: str = "es",
) -> Dict[str, str]:
    """
    Genera los archivos de salida estándar.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # ---------- SRT (traducción) ----------
    srt_path = output_dir / f"{base_name}.{target_lang}.srt"
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            start = _format_timestamp_srt(seg["start"])
            end = _format_timestamp_srt(seg["end"])
            text = seg.get("translation") or seg["text"]
            label = _speaker_label(seg)
            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"[{label}] {text}\n\n")
    results["srt_path"] = str(srt_path)

    # ---------- VTT ----------
    vtt_path = output_dir / f"{base_name}.{target_lang}.vtt"
    with open(vtt_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for i, seg in enumerate(segments, 1):
            start = _format_timestamp_vtt(seg["start"])
            end = _format_timestamp_vtt(seg["end"])
            text = seg.get("translation") or seg["text"]
            label = _speaker_label(seg)
            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"[{label}] {text}\n\n")
    results["vtt_path"] = str(vtt_path)

    # ---------- JSON completo ----------
    json_path = output_dir / f"{base_name}.json"
    payload = {
        "source_lang": source_lang,
        "target_lang": target_lang,
        "segments": segments,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    results["json_path"] = str(json_path)

    # ---------- Texto legible (guion) ----------
    txt_path = output_dir / f"{base_name}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"VoxSync — Transcripción y traducción\n")
        f.write(f"Idioma origen: {source_lang}  |  Idioma destino: {target_lang}\n")
        f.write("=" * 60 + "\n\n")

        for seg in segments:
            start = _format_timestamp_srt(seg["start"])
            label = _speaker_label(seg)
            original = seg["text"]
            translated = seg.get("translation", "")
            g_conf = seg.get("gender_confidence")
            a_conf = seg.get("age_confidence")

            f.write(f"[{start}] {label}\n")
            if g_conf is not None or a_conf is not None:
                conf_parts = []
                if g_conf is not None:
                    conf_parts.append(f"género {g_conf:.0%}")
                if a_conf is not None:
                    conf_parts.append(f"edad {a_conf:.0%}")
                f.write(f"  Confianza : {", ".join(conf_parts)}\n")
            f.write(f"  Original : {original}\n")
            if translated and translated != original:
                f.write(f"  Traducido: {translated}\n")
            f.write("\n")

    results["txt_path"] = str(txt_path)

    return results
