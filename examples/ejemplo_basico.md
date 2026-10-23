# Ejemplos de uso de VoxSync

## 1. Procesar un video de YouTube (inglés → español)

```bash
voxsync process "https://www.youtube.com/watch?v=XXXX" \
  --source-lang en \
  --target-lang es \
  --output ./resultados/demo
```

## 2. Procesar un episodio local de una serie

```bash
voxsync process "./series/TheOffice/S01E01.mp4" \
  -s en \
  -t es \
  -o "./series/TheOffice/subtitulos" \
  --model large-v3
```

## 3. Usar un modelo más rápido (para pruebas)

```bash
voxsync process "./video_corto.mp4" --model small --device cpu
```

## 4. Traducir de español a inglés

```bash
voxsync process "./entrevista.mp4" -s es -t en
```

## 5. Procesar una carpeta completa (modo lote / serie entera)

```bash
# Procesa todos los videos de una carpeta (y subcarpetas)
voxsync batch "./series/TheOffice/Season01" \
  -o "./subtitulos/TheOffice/S01" \
  -s en \
  -t es \
  --model large-v3
```

## Resultado esperado

Después de ejecutar el comando, en la carpeta de salida encontrarás:

```
resultados/demo/
├── NombreDelVideo.es.srt     ← Subtítulos listos para usar
├── NombreDelVideo.es.vtt
├── NombreDelVideo.json       ← Datos completos (para integraciones)
└── NombreDelVideo.txt        ← Guion legible con hablantes
```

### Ejemplo de contenido del .txt

```
[00:01:23,450] SPEAKER_00 (Hombre, Adulto)
  Original : Hey, how are you doing today?
  Traducido: Hola, ¿cómo estás hoy?

[00:01:25,120] SPEAKER_01 (Mujer, Adulto)
  Original : I'm good, thanks. What about you?
  Traducido: Estoy bien, gracias. ¿Y tú?
```

Los segmentos del mismo hablante que están muy juntos se unen automáticamente para mantener el hilo de la conversación.
