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
- Detección de **género** (modelo ML) y **edad** (niño / adolescente / adulto / mayor)
- Traducción de calidad profesional (inglés ↔ español y otros idiomas) con modelos locales
- Mantiene el **hilo cronológico** de la conversación (une segmentos del mismo hablante)
- Soporte para videos locales y enlaces de YouTube
- **Modo lote** para series completas
- **Interfaz gráfica** (Gradio) para usuarios no técnicos
- Exportación: **SRT**, **VTT**, **JSON** y texto legible
- Diseñado para empaquetarse y venderse como producto profesional

---

## 💻 Requisitos del sistema

- Python **3.10+**
- **FFmpeg** instalado y disponible en el PATH
- GPU con CUDA recomendada (funciona también en CPU)
- 8–16 GB de RAM (según el modelo de Whisper)
- Cuenta gratuita en Hugging Face (necesaria para la diarización con pyannote)

---

## 🚀 Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/proyecto1095si1-boop/voz-ia-series.git
cd voz-ia-series

# 2. Crear entorno virtual (recomendado)
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Instalar el paquete en modo editable
pip install -e .

# 5. Configurar token de Hugging Face (gratis)
# Crea una cuenta en https://huggingface.co y genera un token
# Luego acepta las condiciones de pyannote en:
# https://huggingface.co/pyannote/speaker-diarization-3.1
export HF_TOKEN="tu_token_aqui"   # Linux/macOS
# set HF_TOKEN=tu_token_aqui      # Windows CMD
```

---

## 💬 Uso

### Interfaz gráfica (recomendado para no técnicos)

```bash
voxsync gui
# Abre http://127.0.0.1:7860 en el navegador
```

Opciones:

```bash
voxsync gui --port 8080          # otro puerto
voxsync gui --share              # enlace público temporal
```

### Un video / YouTube

```bash
voxsync process "https://www.youtube.com/watch?v=XXXX" -s en -t es
voxsync process "./episodio01.mp4" -s en -t es -o ./resultados
voxsync process "./video.mp4" --model medium --device cpu
```

### Serie completa (modo lote)

```bash
voxsync batch "./series/Season01" -o "./subtitulos/S01" -s en -t es
```

### Opciones principales

| Opción | Descripción | Default |
|--------|-------------|--------|
| `--source-lang` / `-s` | Idioma de origen | `en` |
| `--target-lang` / `-t` | Idioma de destino | `es` |
| `--model` | Tamaño Whisper | `large-v3` |
| `--device` | `auto`, `cpu` o `cuda` | `auto` |
| `--output` / `-o` | Carpeta de salida | `output` |
| `--keep-audio` | Conservar el WAV | `False` |

---

## 📂 Archivos de salida

- `nombre.es.srt` — Subtítulos SRT (traducidos)
- `nombre.es.vtt` — Subtítulos WebVTT
- `nombre.json` — Datos completos (timestamps, hablantes, género, edad, confianza, texto)
- `nombre.txt` — Guion legible con hablantes y diálogos

---

## 📁 Estructura del proyecto

```
voxsync/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── src/voxsync/
│   ├── cli.py              # process | batch | gui | version
│   ├── gui.py              # Interfaz Gradio
│   ├── pipeline.py
│   ├── batch.py
│   ├── downloader.py
│   ├── audio.py
│   ├── transcriber.py
│   ├── speaker_attrs.py    # Género (ML) + edad
│   ├── translator.py
│   ├── postprocess.py      # Hilo de conversación
│   ├── formatter.py
│   └── config.py
└── examples/
```

---

## 🛠️ Roadmap

- [x] Nombre comercial y estructura base
- [x] Pipeline de descarga + extracción de audio
- [x] Transcripción + diarización con WhisperX
- [x] Detección de género (ML) y edad mejorada
- [x] Traducción local
- [x] Post-procesado del hilo de conversación
- [x] Generación de subtítulos (SRT/VTT)
- [x] CLI profesional (`process`, `batch`, `gui`)
- [x] Interfaz gráfica (Gradio)
- [x] Modo batch (varios episodios)
- [ ] Empaquetado para distribución (instalador)
- [ ] Documentación comercial y licencia definitiva

---

## 📝 Licencia y comercialización

Este software está pensado para **uso comercial**.  
La licencia definitiva se definirá en una etapa posterior (posible licencia comercial o dual).

---

**VoxSync** — Diálogos claros. Idiomas conectados. 100% bajo tu control.
