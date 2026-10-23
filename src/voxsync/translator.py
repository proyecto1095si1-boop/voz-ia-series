"""
Traducción local usando modelos NLLB / MarianMT de Hugging Face.
"""

from typing import Any, Dict, List, Optional

# Cache simple de modelos para no recargar en cada llamada
_model_cache = {}


def _get_translator(source_lang: str, target_lang: str):
    """
    Carga (o reutiliza) un pipeline de traducción.
    Usamos Helsinki-NLP MarianMT cuando está disponible,
    o NLLB como fallback más general.
    """
    key = f"{source_lang}-{target_lang}"
    if key in _model_cache:
        return _model_cache[key]

    try:
        from transformers import pipeline
    except ImportError as e:
        raise ImportError(
            "transformers no está instalado. Ejecuta:\n"
            "pip install transformers sentencepiece"
        ) from e

    # Mapeo de pares comunes a modelos MarianMT (más ligeros y rápidos)
    marian_models = {
        ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
        ("es", "en"): "Helsinki-NLP/opus-mt-es-en",
        ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
        ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
        ("en", "pt"): "Helsinki-NLP/opus-mt-en-ROMANCE",
        ("es", "pt"): "Helsinki-NLP/opus-mt-es-pt",
    }

    model_name = marian_models.get((source_lang, target_lang))

    if model_name is None:
        # Fallback a NLLB-200 (más pesado pero soporta muchos idiomas)
        model_name = "facebook/nllb-200-distilled-600M"

    translator = pipeline(
        "translation",
        model=model_name,
        device=-1,  # CPU por defecto; se puede mejorar después
    )

    _model_cache[key] = translator
    return translator


def translate_segments(
    segments: List[Dict[str, Any]],
    source_lang: str = "en",
    target_lang: str = "es",
) -> List[Dict[str, Any]]:
    """
    Traduce el texto de cada segmento manteniendo el resto de metadatos.
    """
    if source_lang == target_lang:
        for seg in segments:
            seg["translation"] = seg["text"]
        return segments

    translator = _get_translator(source_lang, target_lang)

    # Traducir en lotes para mayor eficiencia
    texts = [seg["text"] for seg in segments]

    # Algunos modelos tienen límite de longitud; procesamos de a pocos
    batch_size = 16
    translations = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        results = translator(batch, max_length=512)
        for r in results:
            # MarianMT devuelve "translation_text", NLLB similar
            translations.append(r.get("translation_text", r.get("generated_text", "")))

    for seg, translated in zip(segments, translations):
        seg["translation"] = translated.strip()

    return segments
