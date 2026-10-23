"""
Detección aproximada de atributos de hablante (género y rango de edad).

Nota: Esta es una aproximación basada en características acústicas.
Para uso profesional se puede mejorar con modelos más avanzados.
"""

from pathlib import Path
from typing import Any, Dict, List
import numpy as np

try:
    import librosa
except ImportError:
    librosa = None


def _estimate_pitch_and_energy(y: np.ndarray, sr: int) -> Dict[str, float]:
    """Calcula pitch medio y energía RMS."""
    # Pitch (f0) con pyin
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
    )
    f0_clean = f0[~np.isnan(f0)] if f0 is not None else np.array([])

    mean_pitch = float(np.mean(f0_clean)) if len(f0_clean) > 0 else 0.0

    # Energía
    rms = librosa.feature.rms(y=y)[0]
    mean_energy = float(np.mean(rms))

    return {
        "mean_pitch": mean_pitch,
        "mean_energy": mean_energy,
    }


def _classify_voice(mean_pitch: float) -> Dict[str, str]:
    """
    Clasificación muy simple basada en pitch.
    Esto es solo una heurística inicial.
    """
    if mean_pitch == 0.0:
        return {"gender": "unknown", "age_group": "unknown"}

    # Rangos aproximados (Hz)
    # Niños: pitch más alto
    # Mujeres adultas: ~165-255 Hz
    # Hombres adultos: ~85-180 Hz
    # Personas mayores: pitch suele bajar un poco

    if mean_pitch > 250:
        gender = "female"
        age_group = "child"
    elif mean_pitch > 180:
        gender = "female"
        age_group = "adult"
    elif mean_pitch > 140:
        gender = "male"
        age_group = "adult"
    elif mean_pitch > 100:
        gender = "male"
        age_group = "adult"
    else:
        gender = "male"
        age_group = "senior"

    return {"gender": gender, "age_group": age_group}


def enrich_speakers(
    segments: List[Dict[str, Any]],
    audio_path: Path,
) -> List[Dict[str, Any]]:
    """
    Añade a cada segmento una estimación de género y grupo de edad
    basándose en el hablante dominante.

    Estrategia simple:
    - Agrupa segmentos por speaker
    - Extrae un fragmento representativo de cada speaker
    - Estima pitch → clasifica
    """
    if librosa is None:
        # Si no hay librosa, dejamos los atributos como unknown
        for seg in segments:
            seg["gender"] = "unknown"
            seg["age_group"] = "unknown"
        return segments

    # Agrupar por speaker
    speakers = {}
    for seg in segments:
        spk = seg.get("speaker", "SPEAKER_00")
        if spk not in speakers:
            speakers[spk] = []
        speakers[spk].append(seg)

    # Cargar audio completo una sola vez
    y, sr = librosa.load(str(audio_path), sr=16000, mono=True)

    speaker_attrs: Dict[str, Dict[str, str]] = {}

    for spk, segs in speakers.items():
        # Tomar el segmento más largo del hablante como muestra
        longest = max(segs, key=lambda s: s["end"] - s["start"])
        start_sample = int(longest["start"] * sr)
        end_sample = int(longest["end"] * sr)

        # Limitar a máximo 8 segundos para no sobrecargar
        max_samples = 8 * sr
        if end_sample - start_sample > max_samples:
            end_sample = start_sample + max_samples

        chunk = y[start_sample:end_sample]

        if len(chunk) < sr * 0.5:  # menos de 0.5s
            speaker_attrs[spk] = {"gender": "unknown", "age_group": "unknown"}
            continue

        feats = _estimate_pitch_and_energy(chunk, sr)
        attrs = _classify_voice(feats["mean_pitch"])
        speaker_attrs[spk] = attrs

    # Aplicar a todos los segmentos
    for seg in segments:
        spk = seg.get("speaker", "SPEAKER_00")
        attrs = speaker_attrs.get(spk, {"gender": "unknown", "age_group": "unknown"})
        seg["gender"] = attrs["gender"]
        seg["age_group"] = attrs["age_group"]

    return segments
