# VoxSync

**Transcripción, diarización y traducción profesional de diálogos · 100% local**

VoxSync es una herramienta de grado profesional diseñada para canales de YouTube, productoras y plataformas de streaming que necesitan transcribir y traducir episodios, videos y contenido audiovisual con alta precisión, manteniendo el hilo de la conversación y detectando características de los hablantes.

---

## 🎯 ¿Para quién es?

- Canales de YouTube de habla inglesa que quieren llegar a público hispanohablante (y viceversa)
- Productoras y estudios de doblaje / subtitulado
- Empresas de streaming y contenido (Netflix, plataformas independientes, etc.)
- Cualquier equipo que necesite transcripciones precisas, diarizadas y traducidas de forma local y sin costos por API

---

## ✨ Características principales

- **100% local** — Sin APIs de pago. Todo corre en tu máquina.
- Transcripción de alta calidad con **WhisperX**
- **Diarización de hablantes** (quién habla y cuándo)
- Detección aproximada de **género** y **rango de edad** (niño/niña, adulto, persona mayor)
- Traducción de calidad profesional (inglés ↔ español y otros idiomas) con modelos locales (NLLB)
- Mantiene el **hilo cronológico** de la conversación
- Soporte para videos locales y enlaces de YouTube
- Exportación en formatos estándar: **SRT**, **VTT**, **JSON** y texto legible
- Diseñado para ser empaquetado y vendido como producto profesional

---

## 🚀 Estado del proyecto

Actualmente en desarrollo activo (fase de producción inicial).

---

## 📁 Estructura del proyecto

```
voxsync/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── src/
│   └── voxsync/
│       ├── __init__.py
│       ├── cli.py
│       ├── pipeline.py
│       ├── downloader.py
│       ├── audio.py
│       ├── transcriber.py
│       ├── speaker_attrs.py
│       ├── translator.py
│       └── formatter.py
├── models/          # Modelos se descargan aquí
└── examples/
```

---

## 💻 Requisitos del sistema

- Python 3.10+
- FFmpeg instalado en el sistema
- GPU recomendada (CUDA) para velocidad, pero funciona en CPU
- ~8–16 GB de RAM (dependiendo del modelo de Whisper)

---

## 📝 Licencia y comercialización

Este software está pensado para uso comercial.  
La licencia se definirá en una etapa posterior (posible licencia comercial / dual).

---

## 🛠️ Roadmap inicial

- [x] Nombre comercial y estructura base
- [ ] Pipeline de descarga + extracción de audio
- [ ] Transcripción + diarización con WhisperX
- [ ] Detección de atributos de hablante
- [ ] Traducción local con NLLB
- [ ] Generación de subtítulos (SRT/VTT)
- [ ] CLI profesional
- [ ] Empaquetado para distribución

---

**VoxSync** — Diálogos claros. Idiomas conectados. 100% bajo tu control.
