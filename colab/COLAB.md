# VoxSync Colab

## Abrir notebook
https://colab.research.google.com/github/proyecto1095si1-boop/voz-ia-series/blob/main/colab/VoxSync_Colab.ipynb

## Pipeline (v5)
Archivo: `colab/voxsync_colab_pipeline.py`

### Features
- Faster-Whisper (mas rapido)
- Traduccion MarianMT local (sin rate limit de Google)
- Diarizacion pyannote (opcional, HF_TOKEN)
- 1 voz TTS fija por hablante (Edge-TTS ES)
- Demucs: conserva musica/ambiente
- Merge de frases del mismo hablante
- **Checkpoint/resume** si falla a mitad
- Soft subtitles en el MP4 (sin bajar calidad de video)
- MODE=fast | quality
- Lote de episodios (INPUT_FOLDER)

### Uso rapido
```python
import os
# os.environ["HF_TOKEN"] = "hf_..."
!wget -q -O /content/voxsync_colab_pipeline.py https://raw.githubusercontent.com/proyecto1095si1-boop/voz-ia-series/main/colab/voxsync_colab_pipeline.py
%run /content/voxsync_colab_pipeline.py
```

### Config util
```python
MODE = "quality"   # o "fast"
RESUME = True      # reanudar TTS/Demucs
EMBED_SOFT_SUBS = True
```
