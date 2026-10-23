# VoxSync Colab pipeline - doblaje mejorado
# Mejoras: Demucs (fondo), TTS estirado, video -c:v copy

YOUTUBE_URL = ""
SOURCE_LANG = "en"
TARGET_LANG = "es"
WHISPER_MODEL = "small"
DUB_AUDIO = True
KEEP_BACKGROUND = True
BURN_SUBTITLES = False
VOICE_FEMALE = "es-ES-ElviraNeural"
VOICE_MALE = "es-ES-AlvaroNeural"
VOICE_SENIOR_M = "es-ES-AlvaroNeural"
VOICE_SENIOR_F = "es-ES-ElviraNeural"
VOICE_CHILD = "es-ES-ElviraNeural"
CROSSFADE_MS = 40

import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "openai-whisper", "yt-dlp", "transformers>=4.40,<5", "sentencepiece",
    "sacremoses", "edge-tts", "librosa", "soundfile", "demucs"])
subprocess.run(["apt", "-y", "-qq", "install", "ffmpeg"], check=False)

from pathlib import Path
import numpy as np
import json as _json
import shutil
import whisper
import torch
import librosa
import soundfile as sf
import edge_tts
import asyncio
from collections import Counter
from transformers import MarianMTModel, MarianTokenizer

work = Path("/content/voxsync")
work.mkdir(exist_ok=True)

if YOUTUBE_URL.strip():
    print("Descargando YouTube...")
    subprocess.run([
        "yt-dlp", "-f", "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
        "-o", str(work / "%(title)s.%(ext)s"), "--no-playlist", YOUTUBE_URL.strip()
    ], check=True)
    cands = []
    for ext in (".mp4", ".webm", ".mkv"):
        cands.extend(work.glob("*" + ext))
    video_path = max(cands, key=lambda p: p.stat().st_mtime)
else:
    from google.colab import files
    print("Sube un video...")
    uploaded = files.upload()
    name = list(uploaded.keys())[0]
    video_path = work / name
    video_path.write_bytes(uploaded[name])
print("Video:", video_path)

audio_path = work / "audio.wav"
audio_16k = work / "audio_16k.wav"
subprocess.run(["ffmpeg", "-y", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le",
                "-ar", "44100", "-ac", "2", str(audio_path)], check=True, capture_output=True)
subprocess.run(["ffmpeg", "-y", "-i", str(audio_path), "-ac", "1", "-ar", "16000",
                str(audio_16k)], check=True, capture_output=True)

print("Transcribiendo...")
model = whisper.load_model(WHISPER_MODEL)
result = model.transcribe(str(audio_16k), language=SOURCE_LANG or None, verbose=False)
segments = result["segments"]
print(len(segments), "segmentos")

model_name = "Helsinki-NLP/opus-mt-en-es"
if SOURCE_LANG == "es" and TARGET_LANG == "en":
    model_name = "Helsinki-NLP/opus-mt-es-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
mt_model = MarianMTModel.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
mt_model = mt_model.to(device).eval()

def translate_batch(texts, max_length=256):
    texts = [t if (t and t.strip()) else "." for t in texts]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = mt_model.generate(**inputs, max_length=max_length)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

texts = [(s.get("text") or "").strip() for s in segments]
translations = list(texts) if SOURCE_LANG == TARGET_LANG else []
if SOURCE_LANG != TARGET_LANG:
    for i in range(0, len(texts), 16):
        batch = texts[i:i+16]
        out = translate_batch(batch)
        for j, t in enumerate(batch):
            translations.append(out[j] if t else "")
        print(f"  trad {min(i+16, len(texts))}/{len(texts)}")
for seg, tr in zip(segments, translations):
    seg["translation"] = tr
print("Traduccion OK")

y16, sr16 = librosa.load(str(audio_16k), sr=16000)

def classify_voice(start, end):
    i0 = max(0, int(start * sr16))
    i1 = min(len(y16), int(end * sr16))
    if i1 - i0 < sr16 * 0.15:
        return "adult_male"
    chunk = y16[i0:i1]
    try:
        f0, _, _ = librosa.pyin(chunk, fmin=60, fmax=400, sr=sr16)
        f0_clean = f0[~np.isnan(f0)] if f0 is not None else np.array([])
    except Exception:
        f0_clean = np.array([])
    if len(f0_clean) < 3:
        return "adult_male"
    med = float(np.median(f0_clean))
    if med >= 220:
        return "child"
    if med >= 165:
        return "adult_female"
    if med >= 135:
        return "adult_female"
    if med >= 95:
        return "adult_male"
    return "senior_male"

for seg in segments:
    seg["voice_type"] = classify_voice(seg["start"], seg["end"])
print("Distribucion:", Counter(s["voice_type"] for s in segments))

bg_path = work / "background.wav"
if DUB_AUDIO and KEEP_BACKGROUND:
    print("Separando fondo (Demucs)...")
    demucs_out = work / "demucs_out"
    if demucs_out.exists():
        shutil.rmtree(demucs_out)
    subprocess.run(["python", "-m", "demucs", "--two-stems", "vocals", "-n", "htdemucs",
                    "-o", str(demucs_out), str(audio_path)], check=True)
    found_bg = list(demucs_out.rglob("no_vocals.wav"))
    if not found_bg:
        raise RuntimeError("Demucs no genero no_vocals.wav")
    shutil.copy(found_bg[0], bg_path)
    print("Fondo:", bg_path)
elif DUB_AUDIO:
    probe = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format",
                            str(video_path)], capture_output=True, text=True, check=True)
    duration = float(_json.loads(probe.stdout)["format"]["duration"])
    sf.write(str(bg_path), np.zeros((int(duration * 44100), 2), dtype=np.float32), 44100)

def pick_voice(voice_type):
    if voice_type == "adult_female":
        return VOICE_FEMALE, "+0Hz", "+0%"
    if voice_type == "child":
        return VOICE_CHILD, "+15Hz", "+8%"
    if voice_type == "senior_male":
        return VOICE_SENIOR_M, "-20Hz", "-6%"
    if voice_type == "senior_female":
        return VOICE_SENIOR_F, "-12Hz", "-4%"
    return VOICE_MALE, "+0Hz", "+0%"

async def synth_one(text, out_mp3, voice, rate, pitch):
    text = (text or "").strip()
    if not text:
        return False
    await edge_tts.Communicate(text, voice=voice, rate=rate, pitch=pitch).save(str(out_mp3))
    return True

def fit_to_duration(wav_in, wav_out, target_sec, sr=44100):
    y, _ = librosa.load(str(wav_in), sr=sr, mono=True)
    if len(y) < 10 or target_sec <= 0.05:
        sf.write(str(wav_out), y, sr)
        return
    cur = len(y) / sr
    rate = float(np.clip(cur / target_sec, 0.7, 1.45))
    y2 = librosa.effects.time_stretch(y, rate=rate)
    n = int(target_sec * sr)
    if len(y2) > n:
        y2 = y2[:n]
    else:
        y2 = np.pad(y2, (0, n - len(y2)))
    fade = min(int(sr * CROSSFADE_MS / 1000), len(y2) // 4)
    if fade > 2:
        y2[:fade] *= np.linspace(0, 1, fade)
        y2[-fade:] *= np.linspace(1, 0, fade)
    sf.write(str(wav_out), y2.astype(np.float32), sr)

dub_path = None
if DUB_AUDIO:
    tts_dir = work / "tts_segs"
    tts_dir.mkdir(exist_ok=True)
    print("TTS + ajuste de duracion...")

    async def run_tts():
        for i, seg in enumerate(segments):
            voice, pitch, rate = pick_voice(seg.get("voice_type", "adult_male"))
            mp3 = tts_dir / f"seg_{i:04d}.mp3"
            wav_raw = tts_dir / f"seg_{i:04d}_raw.wav"
            wav_fit = tts_dir / f"seg_{i:04d}_fit.wav"
            ok = await synth_one(seg.get("translation") or seg.get("text"), mp3, voice, rate, pitch)
            if not ok:
                seg["tts_fit"] = None
                continue
            subprocess.run(["ffmpeg", "-y", "-i", str(mp3), "-ar", "44100", "-ac", "1", str(wav_raw)],
                           check=True, capture_output=True)
            fit_to_duration(wav_raw, wav_fit, max(0.08, float(seg["end"] - seg["start"])), 44100)
            seg["tts_fit"] = str(wav_fit)
            if (i + 1) % 20 == 0:
                print(f"  TTS {i+1}/{len(segments)}")

    try:
        asyncio.get_event_loop().run_until_complete(run_tts())
    except RuntimeError:
        asyncio.run(run_tts())

    bg, sr = sf.read(str(bg_path))
    if bg.ndim == 1:
        bg = np.stack([bg, bg], axis=1)
    mix = bg.astype(np.float32).copy()
    for i, seg in enumerate(segments):
        path = seg.get("tts_fit")
        if not path or not Path(path).exists():
            continue
        voice, _ = sf.read(str(path))
        if voice.ndim > 1:
            voice = voice.mean(axis=1)
        voice = voice.astype(np.float32)
        start = int(seg["start"] * sr)
        end = min(start + len(voice), len(mix))
        if start >= len(mix):
            continue
        voice = voice[: end - start]
        stereo = np.stack([voice, voice], axis=1)
        mix[start:end] *= 0.35
        mix[start:end] += stereo * 0.95
    peak = np.max(np.abs(mix)) + 1e-8
    if peak > 0.99:
        mix *= 0.99 / peak
    dub_path = work / "audio_doblado.wav"
    sf.write(str(dub_path), mix, sr)
    print("Audio final:", dub_path)

def ts(seconds):
    total = int(float(seconds))
    ms = int((float(seconds) - total) * 1000)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

base = video_path.stem
srt_path = work / f"{base}.es.srt"
with open(srt_path, "w", encoding="utf-8") as f:
    for i, seg in enumerate(segments, 1):
        text = (seg.get("translation") or seg["text"]).strip()
        f.write(f"{i}\n{ts(seg['start'])} --> {ts(seg['end'])}\n{text}\n\n")

out_video = work / f"{base}_ES_DOBLADO.mp4"
audio_input = str(dub_path) if dub_path else str(audio_path)
if BURN_SUBTITLES:
    srt_esc = str(srt_path).replace("\\", "/").replace(":", "\\:")
    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-i", audio_input,
           "-map", "0:v:0", "-map", "1:a:0",
           "-vf", f"subtitles={srt_esc}:force_style='FontSize=18,Outline=1'",
           "-c:v", "libx264", "-preset", "medium", "-crf", "18",
           "-c:a", "aac", "-b:a", "192k", "-shortest", str(out_video)]
else:
    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-i", audio_input,
           "-map", "0:v:0", "-map", "1:a:0",
           "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(out_video)]
print("Renderizando...")
subprocess.run(cmd, check=True, capture_output=True)
print("Video:", out_video)

from google.colab import files
files.download(str(srt_path))
files.download(str(out_video))
if dub_path:
    files.download(str(dub_path))
print("Listo: fondo conservado + voces fluidas + video sin recomprimir")
