# VoxSync Colab pipeline v4.1
# Fixes: pyannote token= + nest_asyncio for Colab

YOUTUBE_URL = ""
INPUT_FOLDER = ""
SOURCE_LANG = "en"
TARGET_LANG = "es"
WHISPER_MODEL = "small"

DUB_AUDIO = True
KEEP_BACKGROUND = True
BURN_SUBTITLES = False
DUCK_BACKGROUND = 0.30
TTS_GAIN = 0.95
CROSSFADE_MS = 45
USE_DIARIZATION = True
MERGE_SAME_SPEAKER = True
MERGE_MAX_GAP = 0.45
MERGE_MAX_CHARS = 280

VOICE_POOL = {
    "adult_female": ["es-MX-DaliaNeural", "es-ES-ElviraNeural", "es-AR-ElenaNeural"],
    "adult_male": ["es-MX-JorgeNeural", "es-ES-AlvaroNeural", "es-AR-TomasNeural"],
    "senior_male": ["es-ES-AlvaroNeural"],
    "senior_female": ["es-ES-ElviraNeural"],
    "child": ["es-MX-DaliaNeural"],
}

import os, subprocess, sys, shutil, asyncio
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "openai-whisper", "yt-dlp", "transformers>=4.40,<5", "sentencepiece",
    "sacremoses", "edge-tts", "librosa", "soundfile", "demucs", "nest_asyncio"])
if USE_DIARIZATION:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pyannote.audio"])
subprocess.run(["apt", "-y", "-qq", "install", "ffmpeg"], check=False)

import nest_asyncio
nest_asyncio.apply()

from pathlib import Path
import numpy as np, json as _json, whisper, torch, librosa, soundfile as sf, edge_tts
from collections import Counter, defaultdict
from transformers import MarianMTModel, MarianTokenizer

work = Path("/content/voxsync")
work.mkdir(exist_ok=True)

def get_video_list():
    if INPUT_FOLDER.strip():
        folder = Path(INPUT_FOLDER)
        vids = sorted([p for ext in ("*.mp4","*.mkv","*.webm","*.mov") for p in folder.glob(ext)])
        if not vids:
            raise SystemExit(f"No hay videos en {folder}")
        print(f"Modo LOTE: {len(vids)} videos")
        return vids
    if YOUTUBE_URL.strip():
        print("Descargando YouTube...")
        subprocess.run(["yt-dlp", "-f", "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
            "-o", str(work / "%(title)s.%(ext)s"), "--no-playlist", YOUTUBE_URL.strip()], check=True)
        cands = []
        for ext in (".mp4", ".webm", ".mkv"):
            cands.extend(work.glob("*" + ext))
        return [max(cands, key=lambda p: p.stat().st_mtime)]
    from google.colab import files
    print("Sube un video (o varios)...")
    uploaded = files.upload()
    paths = []
    for name, data in uploaded.items():
        p = work / name
        p.write_bytes(data)
        paths.append(p)
    return paths

print("Cargando Whisper:", WHISPER_MODEL)
whisper_model = whisper.load_model(WHISPER_MODEL)
model_name = "Helsinki-NLP/opus-mt-es-en" if SOURCE_LANG == "es" and TARGET_LANG == "en" else "Helsinki-NLP/opus-mt-en-es"
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

def run_diarization(wav_path):
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("  Sin HF_TOKEN: sin diarizacion")
        return None
    try:
        from pyannote.audio import Pipeline
        print("  Diarizando...")
        try:
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)
        except TypeError:
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
        diar = pipeline(str(wav_path))
        turns = [{"start": float(t.start), "end": float(t.end), "speaker": spk}
                 for t, _, spk in diar.itertracks(yield_label=True)]
        print("  hablantes:", sorted(set(x["speaker"] for x in turns)))
        return turns
    except Exception as e:
        print("  Diarizacion fallo:", e)
        return None

def assign_speaker(seg, turns):
    if not turns:
        return "SPEAKER_00"
    mid = (float(seg["start"]) + float(seg["end"])) / 2.0
    best, best_ov = None, 0.0
    for t in turns:
        ov = max(0.0, min(seg["end"], t["end"]) - max(seg["start"], t["start"]))
        if ov > best_ov:
            best_ov, best = ov, t["speaker"]
        if t["start"] <= mid <= t["end"]:
            return t["speaker"]
    return best or "SPEAKER_00"

def classify_voice(y16, sr16, start, end):
    i0, i1 = max(0, int(start * sr16)), min(len(y16), int(end * sr16))
    if i1 - i0 < sr16 * 0.15:
        return "adult_male", 80.0
    chunk = y16[i0:i1]
    try:
        f0, _, _ = librosa.pyin(chunk, fmin=60, fmax=400, sr=sr16)
        f0_clean = f0[~np.isnan(f0)] if f0 is not None else np.array([])
    except Exception:
        f0_clean = np.array([])
    if len(f0_clean) < 3:
        return "adult_male", 80.0
    med = float(np.median(f0_clean))
    if med >= 220: return "child", med
    if med >= 165: return "adult_female", med
    if med >= 135: return "adult_female", med
    if med >= 95: return "adult_male", med
    return "senior_male", med

def merge_same_speaker(segments):
    if not segments: return segments
    out, cur = [], dict(segments[0])
    for seg in segments[1:]:
        gap = float(seg["start"]) - float(cur["end"])
        same = seg.get("speaker") == cur.get("speaker")
        new_text = ((cur.get("translation") or cur.get("text") or "") + " " +
                    (seg.get("translation") or seg.get("text") or "")).strip()
        if same and gap <= MERGE_MAX_GAP and len(new_text) <= MERGE_MAX_CHARS:
            cur["end"] = seg["end"]
            cur["text"] = ((cur.get("text") or "") + " " + (seg.get("text") or "")).strip()
            cur["translation"] = new_text
        else:
            out.append(cur)
            cur = dict(seg)
    out.append(cur)
    print(f"  merge: {len(segments)} -> {len(out)} segmentos")
    return out

def pitch_rate_for(vtype):
    if vtype == "child": return "+18Hz", "+10%"
    if vtype == "senior_male": return "-22Hz", "-8%"
    if vtype == "senior_female": return "-12Hz", "-5%"
    return "+0Hz", "+0%"

async def synth_one(text, out_mp3, voice, rate, pitch):
    text = (text or "").strip()
    if not text: return False
    await edge_tts.Communicate(text, voice=voice, rate=rate, pitch=pitch).save(str(out_mp3))
    return True

def fit_to_duration(wav_in, wav_out, target_sec, sr=44100):
    y, _ = librosa.load(str(wav_in), sr=sr, mono=True)
    if len(y) < 10 or target_sec <= 0.05:
        sf.write(str(wav_out), y, sr); return
    rate = float(np.clip((len(y)/sr) / target_sec, 0.72, 1.40))
    y2 = librosa.effects.time_stretch(y, rate=rate)
    n = int(target_sec * sr)
    y2 = y2[:n] if len(y2) > n else np.pad(y2, (0, n - len(y2)))
    fade = min(int(sr * CROSSFADE_MS / 1000), max(2, len(y2) // 4))
    if fade > 2:
        y2[:fade] *= np.linspace(0, 1, fade)
        y2[-fade:] *= np.linspace(1, 0, fade)
    sf.write(str(wav_out), y2.astype(np.float32), sr)

def ts(seconds):
    total = int(float(seconds)); ms = int((float(seconds) - total) * 1000)
    h, rem = divmod(total, 3600); m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)

def process_one(video_path, out_root):
    video_path = Path(video_path)
    print("\n==========", video_path.name, "==========")
    ep_dir = out_root / video_path.stem
    ep_dir.mkdir(parents=True, exist_ok=True)
    audio_path = ep_dir / "audio.wav"
    audio_16k = ep_dir / "audio_16k.wav"
    subprocess.run(["ffmpeg", "-y", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le",
                    "-ar", "44100", "-ac", "2", str(audio_path)], check=True, capture_output=True)
    subprocess.run(["ffmpeg", "-y", "-i", str(audio_path), "-ac", "1", "-ar", "16000",
                    str(audio_16k)], check=True, capture_output=True)
    result = whisper_model.transcribe(str(audio_16k), language=SOURCE_LANG or None, verbose=False)
    segments = result["segments"]
    print("  segmentos:", len(segments))
    diar_turns = run_diarization(audio_16k) if USE_DIARIZATION else None
    for seg in segments:
        seg["speaker"] = assign_speaker(seg, diar_turns)
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
    y16, sr16 = librosa.load(str(audio_16k), sr=16000)
    for seg in segments:
        lab, med = classify_voice(y16, sr16, seg["start"], seg["end"])
        seg["voice_type_raw"], seg["pitch_hz"] = lab, med
    speaker_types = defaultdict(list)
    for seg in segments:
        speaker_types[seg["speaker"]].append(seg["voice_type_raw"])
    speaker_voice, used = {}, set()
    for spk, types in speaker_types.items():
        vtype = Counter(types).most_common(1)[0][0]
        pool = list(VOICE_POOL.get(vtype, VOICE_POOL["adult_male"]))
        choice = next((v for v in pool if v not in used), pool[0])
        used.add(choice)
        speaker_voice[spk] = {"voice_type": vtype, "tts_voice": choice}
        print(f"  {spk} -> {vtype} -> {choice}")
    for seg in segments:
        info = speaker_voice[seg["speaker"]]
        seg["voice_type"], seg["tts_voice"] = info["voice_type"], info["tts_voice"]
    if MERGE_SAME_SPEAKER:
        segments = merge_same_speaker(segments)
    bg_path = ep_dir / "background.wav"
    if DUB_AUDIO and KEEP_BACKGROUND:
        print("  Demucs...")
        demucs_out = ep_dir / "demucs_out"
        if demucs_out.exists(): shutil.rmtree(demucs_out)
        subprocess.run(["python", "-m", "demucs", "--two-stems", "vocals", "-n", "htdemucs",
                        "-o", str(demucs_out), str(audio_path)], check=True)
        found_bg = list(demucs_out.rglob("no_vocals.wav"))
        if not found_bg: raise RuntimeError("Demucs fallo")
        shutil.copy(found_bg[0], bg_path)
        print("  Fondo OK")
    elif DUB_AUDIO:
        probe = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format",
                                str(video_path)], capture_output=True, text=True, check=True)
        duration = float(_json.loads(probe.stdout)["format"]["duration"])
        sf.write(str(bg_path), np.zeros((int(duration * 44100), 2), dtype=np.float32), 44100)
    dub_path = None
    if DUB_AUDIO:
        tts_dir = ep_dir / "tts_segs"; tts_dir.mkdir(exist_ok=True)
        print("  TTS...")
        async def run_tts():
            for i, seg in enumerate(segments):
                voice = seg.get("tts_voice") or VOICE_POOL["adult_male"][0]
                pitch, rate = pitch_rate_for(seg.get("voice_type", "adult_male"))
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
                if (i + 1) % 10 == 0:
                    print(f"  TTS {i+1}/{len(segments)}")
        run_async(run_tts())
        bg, sr = sf.read(str(bg_path))
        if bg.ndim == 1: bg = np.stack([bg, bg], axis=1)
        mix = bg.astype(np.float32).copy()
        for seg in segments:
            path = seg.get("tts_fit")
            if not path or not Path(path).exists(): continue
            voice, _ = sf.read(str(path))
            if voice.ndim > 1: voice = voice.mean(axis=1)
            voice = voice.astype(np.float32)
            start = int(seg["start"] * sr)
            end = min(start + len(voice), len(mix))
            if start >= len(mix): continue
            voice = voice[: end - start]
            stereo = np.stack([voice, voice], axis=1)
            mix[start:end] *= float(DUCK_BACKGROUND)
            mix[start:end] += stereo * float(TTS_GAIN)
        peak = np.max(np.abs(mix)) + 1e-8
        if peak > 0.99: mix *= 0.99 / peak
        dub_path = ep_dir / "audio_doblado.wav"
        sf.write(str(dub_path), mix, sr)
        print("  Audio doblado OK")
    base = video_path.stem
    srt_path = ep_dir / f"{base}.es.srt"
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            text = (seg.get("translation") or seg["text"]).strip()
            f.write(f"{i}\n{ts(seg['start'])} --> {ts(seg['end'])}\n[{seg.get('speaker','')}] {text}\n\n")
    json_path = ep_dir / f"{base}.meta.json"
    with open(json_path, "w", encoding="utf-8") as f:
        _json.dump({"speakers": speaker_voice, "segments": [
            {"start": s["start"], "end": s["end"], "speaker": s.get("speaker"),
             "text": s.get("text"), "translation": s.get("translation"),
             "voice_type": s.get("voice_type"), "tts_voice": s.get("tts_voice")}
            for s in segments]}, f, ensure_ascii=False, indent=2)
    out_video = ep_dir / f"{base}_ES_DOBLADO.mp4"
    audio_input = str(dub_path) if dub_path else str(audio_path)
    if BURN_SUBTITLES:
        srt_esc = str(srt_path).replace("\\", "/").replace(":", "\\:")
        cmd = ["ffmpeg", "-y", "-i", str(video_path), "-i", audio_input, "-map", "0:v:0", "-map", "1:a:0",
               "-vf", f"subtitles={srt_esc}:force_style='FontSize=18,Outline=1'",
               "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-c:a", "aac", "-b:a", "192k", "-shortest", str(out_video)]
    else:
        cmd = ["ffmpeg", "-y", "-i", str(video_path), "-i", audio_input, "-map", "0:v:0", "-map", "1:a:0",
               "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(out_video)]
    print("  Renderizando video...")
    subprocess.run(cmd, check=True, capture_output=True)
    print("  OK:", out_video)
    return out_video, srt_path, json_path, dub_path

videos = get_video_list()
out_root = work / "output"; out_root.mkdir(exist_ok=True)
results = []
for vp in videos:
    try:
        results.append(process_one(vp, out_root))
    except Exception as e:
        import traceback
        print("ERROR en", vp, ":", e)
        traceback.print_exc()

from google.colab import files
for out_video, srt_path, json_path, dub_path in results:
    files.download(str(srt_path))
    files.download(str(out_video))
    files.download(str(json_path))
print(f"Listo v4.1: {len(results)} video(s) procesados")
