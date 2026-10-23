"""
Detección de atributos de hablante (género y rango de edad).

Estrategia en capas (100% local):

1. Género: modelo wav2vec2 fine-tuned (Apache-2.0, apto para uso comercial)
   Modelo: alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech

2. Edad / niño:
   - Características acústicas enriquecidas (pitch, rango, centroide espectral, etc.)
   - Heurísticas calibradas + señal de "child" si el pitch es muy alto

3. Fallback: si el modelo de género no está disponible, usa solo acústica.

Cada hablante se analiza con varios fragmentos (no solo el más largo)
para mayor robustez.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np

try:
    import librosa
except ImportError:
    librosa = None

try:
    import torch
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
except ImportError:
    torch = None
    AutoFeatureExtractor = None
    AutoModelForAudioClassification = None


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

GENDER_MODEL_ID = "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"
SAMPLE_RATE = 16000
MIN_CHUNK_SEC = 0.8
MAX_CHUNK_SEC = 6.0
MAX_CHUNKS_PER_SPEAKER = 4

# Cache del modelo de género (se carga una sola vez)
_gender_model = None
_gender_extractor = None
_gender_device = None


# ---------------------------------------------------------------------------
# Modelo de género (ML)
# ---------------------------------------------------------------------------

def _load_gender_model(device: str = "auto"):
    """Carga el clasificador de género (lazy, con cache)."""
    global _gender_model, _gender_extractor, _gender_device

    if _gender_model is not None:
        return _gender_model, _gender_extractor, _gender_device

    if torch is None or AutoModelForAudioClassification is None:
        return None, None, None

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        _gender_extractor = AutoFeatureExtractor.from_pretrained(GENDER_MODEL_ID)
        _gender_model = AutoModelForAudioClassification.from_pretrained(GENDER_MODEL_ID)
        _gender_model.to(device)
        _gender_model.eval()
        _gender_device = device
        return _gender_model, _gender_extractor, _gender_device
    except Exception as e:
        warnings.warn(f"No se pudo cargar el modelo de género ({e}). Se usará heurística.")
        return None, None, None


def _predict_gender_ml(audio: np.ndarray, sr: int = SAMPLE_RATE) -> Tuple[str, float]:
    """
    Predice género con el modelo wav2vec2.
    Returns: (gender, confidence)  gender in {male, female, unknown}
    """
    model, extractor, device = _load_gender_model()
    if model is None:
        return "unknown", 0.0

    # Asegurar mono float32
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)
    audio = audio.astype(np.float32)

    # Limitar longitud (el modelo se entrenó con ~5s)
    max_len = int(MAX_CHUNK_SEC * sr)
    if len(audio) > max_len:
        # Tomar el centro (suele tener más voz estable)
        start = (len(audio) - max_len) // 2
        audio = audio[start : start + max_len]

    inputs = extractor(
        audio,
        sampling_rate=sr,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    # id2label del modelo: 0=female, 1=male
    id2label = model.config.id2label
    pred_id = int(np.argmax(probs))
    label = id2label.get(pred_id, id2label.get(str(pred_id), "unknown")).lower()
    conf = float(probs[pred_id])

    if label not in ("male", "female"):
        # Algunos checkpoints usan otras etiquetas
        if "f" in label:
            label = "female"
        elif "m" in label:
            label = "male"
        else:
            label = "unknown"

    return label, conf


# ---------------------------------------------------------------------------
# Características acústicas (edad + fallback de género)
# ---------------------------------------------------------------------------

def _extract_acoustic_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    """Extrae un conjunto rico de features para clasificar edad/género."""
    if librosa is None or len(y) < sr * 0.3:
        return {}

    features: Dict[str, float] = {}

    # --- Pitch (f0) ---
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),  # ~65 Hz
            fmax=librosa.note_to_hz("C7"),  # ~2093 Hz
            sr=sr,
        )
        f0_clean = f0[~np.isnan(f0)] if f0 is not None else np.array([])
        if len(f0_clean) >= 5:
            features["pitch_mean"] = float(np.mean(f0_clean))
            features["pitch_median"] = float(np.median(f0_clean))
            features["pitch_std"] = float(np.std(f0_clean))
            features["pitch_min"] = float(np.min(f0_clean))
            features["pitch_max"] = float(np.max(f0_clean))
            features["pitch_range"] = features["pitch_max"] - features["pitch_min"]
        else:
            features["pitch_mean"] = 0.0
            features["pitch_median"] = 0.0
            features["pitch_std"] = 0.0
            features["pitch_range"] = 0.0
    except Exception:
        features["pitch_mean"] = 0.0
        features["pitch_median"] = 0.0
        features["pitch_std"] = 0.0
        features["pitch_range"] = 0.0

    # --- Energía ---
    try:
        rms = librosa.feature.rms(y=y)[0]
        features["energy_mean"] = float(np.mean(rms))
        features["energy_std"] = float(np.std(rms))
    except Exception:
        features["energy_mean"] = 0.0
        features["energy_std"] = 0.0

    # --- Centroide espectral (brillo de la voz) ---
    try:
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features["centroid_mean"] = float(np.mean(cent))
        features["centroid_std"] = float(np.std(cent))
    except Exception:
        features["centroid_mean"] = 0.0
        features["centroid_std"] = 0.0

    # --- Rolloff espectral ---
    try:
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features["rolloff_mean"] = float(np.mean(rolloff))
    except Exception:
        features["rolloff_mean"] = 0.0

    # --- Zero-crossing rate (indica fricativas / ruido) ---
    try:
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features["zcr_mean"] = float(np.mean(zcr))
    except Exception:
        features["zcr_mean"] = 0.0

    return features


def _classify_age_from_features(feats: Dict[str, float], gender: str) -> Tuple[str, float]:
    """
    Estima grupo de edad a partir de features acústicas.

    Grupos: child | teen | adult | senior | unknown

    Returns: (age_group, confidence aproximada 0-1)
    """
    pitch = feats.get("pitch_median") or feats.get("pitch_mean") or 0.0
    pitch_std = feats.get("pitch_std", 0.0)
    centroid = feats.get("centroid_mean", 0.0)

    if pitch <= 0:
        return "unknown", 0.0

    # Niños: pitch muy alto, a menudo más variable
    if pitch >= 280 or (pitch >= 250 and pitch_std > 40):
        return "child", 0.75

    # Adolescentes / voces jóvenes altas
    if pitch >= 220 and gender == "female":
        return "teen", 0.55
    if pitch >= 200 and gender == "male":
        # Hombres jóvenes raramente tan altos; puede ser teen o mujer mal clasificada
        return "teen", 0.45

    # Personas mayores: pitch más bajo y a veces menos energía / más inestable
    if gender == "male" and pitch < 110:
        return "senior", 0.60
    if gender == "female" and pitch < 150 and centroid < 1800:
        return "senior", 0.50

    # Adulto por defecto
    return "adult", 0.70


def _classify_gender_from_features(feats: Dict[str, float]) -> Tuple[str, float]:
    """Fallback de género solo con pitch (menos preciso)."""
    pitch = feats.get("pitch_median") or feats.get("pitch_mean") or 0.0
    if pitch <= 0:
        return "unknown", 0.0

    # Umbrales clásicos (con zona de incertidumbre)
    if pitch >= 180:
        return "female", 0.65 if pitch >= 195 else 0.50
    if pitch <= 155:
        return "male", 0.65 if pitch <= 140 else 0.50
    return "unknown", 0.30


# ---------------------------------------------------------------------------
# Selección de fragmentos por hablante
# ---------------------------------------------------------------------------

def _collect_speaker_chunks(
    y: np.ndarray,
    sr: int,
    segs: List[Dict[str, Any]],
) -> List[np.ndarray]:
    """
    Extrae hasta MAX_CHUNKS_PER_SPEAKER fragmentos de audio del hablante,
    priorizando los más largos y con suficiente duración.
    """
    # Ordenar por duración descendente
    ranked = sorted(segs, key=lambda s: s["end"] - s["start"], reverse=True)

    chunks: List[np.ndarray] = []
    for seg in ranked:
        if len(chunks) >= MAX_CHUNKS_PER_SPEAKER:
            break

        duration = seg["end"] - seg["start"]
        if duration < MIN_CHUNK_SEC:
            continue

        start = int(seg["start"] * sr)
        end = int(seg["end"] * sr)
        end = min(end, len(y))
        start = max(0, start)

        # Limitar longitud del chunk
        max_samples = int(MAX_CHUNK_SEC * sr)
        if end - start > max_samples:
            # Centro del segmento
            mid = (start + end) // 2
            start = max(0, mid - max_samples // 2)
            end = start + max_samples

        chunk = y[start:end]
        if len(chunk) >= int(MIN_CHUNK_SEC * sr):
            chunks.append(chunk)

    return chunks


def _analyze_speaker(
    chunks: List[np.ndarray],
    sr: int = SAMPLE_RATE,
) -> Dict[str, Any]:
    """
    Analiza varios chunks de un hablante y devuelve atributos consolidados.
    """
    if not chunks:
        return {
            "gender": "unknown",
            "age_group": "unknown",
            "gender_confidence": 0.0,
            "age_confidence": 0.0,
            "method": "none",
        }

    # --- Género (ML prioritario) ---
    gender_votes: List[Tuple[str, float]] = []
    for chunk in chunks:
        g, c = _predict_gender_ml(chunk, sr)
        if g != "unknown" and c > 0.4:
            gender_votes.append((g, c))

    if gender_votes:
        # Voto ponderado por confianza
        scores = {"male": 0.0, "female": 0.0}
        for g, c in gender_votes:
            scores[g] = scores.get(g, 0.0) + c
        gender = max(scores, key=scores.get)
        gender_conf = scores[gender] / sum(scores.values())
        method = "ml"
    else:
        # Fallback acústico: promediar features
        all_feats = [_extract_acoustic_features(c, sr) for c in chunks]
        all_feats = [f for f in all_feats if f]
        if all_feats:
            avg_feats = {
                k: float(np.mean([f[k] for f in all_feats if k in f]))
                for k in all_feats[0]
            }
            gender, gender_conf = _classify_gender_from_features(avg_feats)
            method = "acoustic"
        else:
            gender, gender_conf, method = "unknown", 0.0, "none"

    # --- Edad (siempre acústica enriquecida) ---
    all_feats = [_extract_acoustic_features(c, sr) for c in chunks]
    all_feats = [f for f in all_feats if f and (f.get("pitch_mean") or 0) > 0]

    if all_feats:
        avg_feats = {
            k: float(np.mean([f[k] for f in all_feats if k in f]))
            for k in all_feats[0]
        }
        age_group, age_conf = _classify_age_from_features(avg_feats, gender)

        # Si el pitch es extremadamente alto, forzar child aunque el género diga otra cosa
        pitch = avg_feats.get("pitch_median") or avg_feats.get("pitch_mean") or 0
        if pitch >= 280:
            age_group = "child"
            age_conf = max(age_conf, 0.8)
    else:
        age_group, age_conf = "unknown", 0.0

    return {
        "gender": gender,
        "age_group": age_group,
        "gender_confidence": round(gender_conf, 3),
        "age_confidence": round(age_conf, 3),
        "method": method,
    }


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def enrich_speakers(
    segments: List[Dict[str, Any]],
    audio_path: Path,
    device: str = "auto",
) -> List[Dict[str, Any]]:
    """
    Añade a cada segmento:
      - gender: male | female | unknown
      - age_group: child | teen | adult | senior | unknown
      - gender_confidence, age_confidence
      - speaker_label legible (opcional, se usa en formatter)

    Usa el modelo ML de género cuando está disponible y features acústicas
    para la edad. Analiza varios fragmentos por hablante.
    """
    # Valores por defecto si no hay librosa ni modelo
    default = {
        "gender": "unknown",
        "age_group": "unknown",
        "gender_confidence": 0.0,
        "age_confidence": 0.0,
        "method": "none",
    }

    if librosa is None and (torch is None or AutoModelForAudioClassification is None):
        for seg in segments:
            seg.update(default)
        return segments

    # Precargar modelo de género (si falla, seguimos con acústica)
    _load_gender_model(device=device)

    # Agrupar por speaker
    speakers: Dict[str, List[Dict[str, Any]]] = {}
    for seg in segments:
        spk = seg.get("speaker", "SPEAKER_00")
        speakers.setdefault(spk, []).append(seg)

    # Cargar audio una sola vez
    try:
        y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    except Exception:
        for seg in segments:
            seg.update(default)
        return segments

    speaker_attrs: Dict[str, Dict[str, Any]] = {}

    for spk, segs in speakers.items():
        chunks = _collect_speaker_chunks(y, sr, segs)
        attrs = _analyze_speaker(chunks, sr)
        speaker_attrs[spk] = attrs

    # Aplicar a todos los segmentos
    for seg in segments:
        spk = seg.get("speaker", "SPEAKER_00")
        attrs = speaker_attrs.get(spk, default)
        seg["gender"] = attrs["gender"]
        seg["age_group"] = attrs["age_group"]
        seg["gender_confidence"] = attrs.get("gender_confidence", 0.0)
        seg["age_confidence"] = attrs.get("age_confidence", 0.0)
        seg["attr_method"] = attrs.get("method", "none")

    return segments
