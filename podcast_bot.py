#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Podcast Summary Bot
-------------------
YouTube podcast link  ->  Hindi summary video(s) with English captions  ->  YouTube upload.
Runs on GitHub Actions (.github/workflows/podcast_summary.yml). Uses only free tiers:
Gemini (3 keys, round-robin), Groq (Whisper + LLM fallback), edge-tts, ffmpeg.
"""
import asyncio
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import wave
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

# ------------------------------------------------------------------ config
WORK = Path("work")
DONE_FILE = Path("done.json")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GROQ_LLM = os.getenv("GROQ_LLM_MODEL", "llama-3.3-70b-versatile")
GROQ_STT = os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo")
VOICE = os.getenv("TTS_VOICE", "hi-IN-MadhurNeural")
TTS_RATE = os.getenv("TTS_RATE", "+0%")
WORDS_PER_PART = int(os.getenv("WORDS_PER_PART", "900"))  # ~6 min of Hindi speech

W, H, FPS = 1280, 720, 25
SEC_PER_IMG = 8
PAUSE = 0.25  # seconds of silence between voice segments
CAPTION_WORDS = 6

GEMINI_KEYS = [
    k for k in (
        os.getenv("GEMINI_API_KEY_1"),
        os.getenv("GEMINI_API_KEY_2"),
        os.getenv("GEMINI_API_KEY_3"),
    ) if k
]
_key_i = 0


def log(msg):
    print(msg, flush=True)


def run(cmd, cwd=None):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {p.stderr[-600:]}")
    return p.stdout


# ------------------------------------------------------------------ telegram
def tg(msg):
    tok, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not tok or not chat:
        return
    for attempt in range(3):  # retry: Telegram sometimes times out from cloud runners
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{tok}/sendMessage",
                data={"chat_id": chat, "text": msg[:3900], "disable_web_page_preview": True},
                timeout=12,
            )
            if r.status_code == 200:
                return
            log(f"telegram HTTP {r.status_code}: {r.text[:150]}")
            if r.status_code in (400, 401, 403, 404):
                return  # wrong token / chat id: retrying will not help
            if r.status_code == 429:  # rate limited: wait as long as Telegram asks
                try:
                    wait = int(r.json().get("parameters", {}).get("retry_after", 5))
                except Exception:
                    wait = 5
                if wait > 10:
                    return  # do not stall the whole run for a status message
                time.sleep(wait + 1)
                continue
        except Exception as e:  # never let alerts crash the bot
            log(f"telegram failed (try {attempt + 1}): {e}")
        time.sleep(3 * (attempt + 1))


def run_url():
    s, r, i = os.getenv("GITHUB_SERVER_URL"), os.getenv("GITHUB_REPOSITORY"), os.getenv("GITHUB_RUN_ID")
    return f"{s}/{r}/actions/runs/{i}" if s and r and i else ""


# ------------------------------------------------------------------ LLM layer
def gemini(prompt, json_mode=True):
    """Round-robin over all Gemini keys; rotate to the next key on any failure."""
    global _key_i
    if not GEMINI_KEYS:
        raise RuntimeError("no Gemini keys set")
    last = None
    for _ in range(len(GEMINI_KEYS) * 2):
        key = GEMINI_KEYS[_key_i % len(GEMINI_KEYS)]
        body = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.7}}
        if json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"
        try:
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
                params={"key": key}, json=body, timeout=180,
            )
        except requests.RequestException as e:
            last = str(e)
            _key_i += 1
            time.sleep(3)
            continue
        _key_i += 1  # spread load across keys even on success
        if r.status_code == 200:
            try:
                return r.json()["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError, ValueError):
                last = "empty response: " + r.text[:200]
                continue
        last = f"{r.status_code} {r.text[:200]}"
        time.sleep(5 if r.status_code in (429, 500, 503) else 1)
    raise RuntimeError("Gemini failed: " + str(last))


def groq_llm(prompt, json_mode=True):
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY missing")
    body = {"model": GROQ_LLM, "messages": [{"role": "user", "content": prompt}], "temperature": 0.7}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    last = None
    for a in range(3):
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"}, json=body, timeout=180,
        )
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        last = f"{r.status_code} {r.text[:200]}"
        time.sleep(10 * (a + 1))
    raise RuntimeError("Groq LLM failed: " + str(last))


def llm(prompt, json_mode=True):
    try:
        return gemini(prompt, json_mode)
    except Exception as e:
        log(f"Gemini failed ({e}); falling back to Groq")
        return groq_llm(prompt, json_mode)


def parse_json(text):
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.S)
        if m:
            return json.loads(m.group(0))
        raise


def llm_json(prompt):
    last = None
    for _ in range(3):
        try:
            return parse_json(llm(prompt, True))
        except Exception as e:
            last = e
            log(f"bad JSON from LLM, retrying: {e}")
            time.sleep(3)
    raise RuntimeError(f"LLM did not return valid JSON: {last}")


# ------------------------------------------------------------------ yt-dlp
def ytdlp(args):
    base = ["yt-dlp", "--no-playlist"]
    if Path("cookies.txt").exists():
        base += ["--cookies", "cookies.txt"]
    err = ""
    for extra in (["--js-runtimes", "node"], []):
        p = subprocess.run(base + extra + args, capture_output=True, text=True)
        if p.returncode == 0:
            return p.stdout
        err = p.stderr
        low = err.lower()
        if extra and ("no such option" in low or "unrecognized" in low):
            continue  # old yt-dlp without --js-runtimes
        break
    keep = [ln.strip() for ln in err.splitlines()
            if ln.startswith(("ERROR", "WARNING")) or "cookie" in ln.lower()]
    raise RuntimeError("yt-dlp failed: " + (" || ".join(keep) or err)[:900])


def cookie_report():
    """Print a safe summary of cookies.txt (cookie NAMES only, never values) and repair lost tabs."""
    p = Path("cookies.txt")
    if not p.exists():
        log("cookies.txt: NOT FOUND (YT_COOKIES secret is missing or empty)")
        return
    raw = p.read_text(encoding="utf-8", errors="ignore").replace("\r\n", "\n")
    out, names, doms, fixed, bad = [], set(), set(), 0, 0
    for ln in raw.split("\n"):
        s = ln.strip()
        if not s or (s.startswith("#") and not s.startswith("#HttpOnly_")):
            out.append(ln)
            continue
        parts = ln.split("\t")
        if len(parts) < 7:
            ws = s.split()
            if len(ws) == 7:  # tabs were turned into spaces while pasting
                parts, fixed = ws, fixed + 1
            else:
                bad += 1
                continue
        out.append("\t".join(parts))
        doms.add(parts[0].replace("#HttpOnly_", ""))
        names.add(parts[5])
    if not out or not out[0].startswith("# Netscape"):
        out.insert(0, "# Netscape HTTP Cookie File")
    p.write_text("\n".join(out) + "\n", encoding="utf-8")
    keys = ["SID", "HSID", "SSID", "SAPISID", "__Secure-1PSID", "__Secure-3PSID", "LOGIN_INFO"]
    log(f"cookies.txt: {len(names)} cookies, domains={sorted(doms)[:6]}, repaired_lines={fixed}, bad_lines={bad}")
    log("cookies.txt login cookies present: " + ", ".join(k for k in keys if k in names) +
        " | missing: " + ", ".join(k for k in keys if k not in names))


def extract_id(url):
    m = re.search(r"(?:v=|youtu\.be/|shorts/|embed/|live/)([A-Za-z0-9_-]{11})", url)
    if m:
        return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url.strip()):
        return url.strip()
    raise ValueError(f"cannot find a YouTube video id in: {url}")


def get_meta(url, vid):
    try:
        info = json.loads(ytdlp(["--dump-single-json", "--skip-download", url]))
        return {
            "title": info.get("title") or "Podcast",
            "channel": info.get("uploader") or info.get("channel") or "",
            "duration": info.get("duration") or 0,
            "thumb": info.get("thumbnail"),
        }
    except Exception as e:
        log(f"yt-dlp metadata failed ({e}); trying oEmbed")
    try:
        r = requests.get("https://www.youtube.com/oembed", params={"url": url, "format": "json"}, timeout=30).json()
        return {"title": r.get("title", "Podcast"), "channel": r.get("author_name", ""), "duration": 0,
                "thumb": f"https://i.ytimg.com/vi/{vid}/maxresdefault.jpg"}
    except Exception:
        return {"title": "Podcast", "channel": "", "duration": 0, "thumb": f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg"}


# ------------------------------------------------------------------ transcript
def transcript_api(vid):
    from youtube_transcript_api import YouTubeTranscriptApi
    try:
        api = YouTubeTranscriptApi()
        tl = api.list(vid)
        try:
            t = tl.find_transcript(["hi", "en"])
        except Exception:
            t = next(iter(tl))
        return " ".join(s.text for s in t.fetch())
    except AttributeError:  # old library version
        data = YouTubeTranscriptApi.get_transcript(vid, languages=["hi", "en"])
        return " ".join(s["text"] for s in data)


def transcript_whisper(url):
    ytdlp(["-f", "bestaudio/best", "-o", str(WORK / "audio_src.%(ext)s"), url])
    src = next(WORK.glob("audio_src.*"))
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar", "16000", "-b:a", "32k",
         "-f", "segment", "-segment_time", "600", "-reset_timestamps", "1", str(WORK / "aseg_%03d.mp3")])
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY missing for Whisper fallback")
    texts = []
    for seg in sorted(WORK.glob("aseg_*.mp3")):
        for attempt in range(4):
            with open(seg, "rb") as f:
                r = requests.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {key}"},
                    files={"file": (seg.name, f, "audio/mpeg")},
                    data={"model": GROQ_STT, "response_format": "json"},
                    timeout=300,
                )
            if r.status_code == 200:
                texts.append(r.json().get("text", ""))
                break
            log(f"whisper {seg.name}: {r.status_code} {r.text[:150]}")
            time.sleep(20 * (attempt + 1))
        else:
            raise RuntimeError(f"Whisper failed for {seg.name}")
    return " ".join(texts)


def vtt_to_text(raw):
    import html
    out, prev = [], ""
    for line in raw.splitlines():
        line = line.strip()
        if not line or "-->" in line or line.startswith(("WEBVTT", "Kind:", "Language:", "NOTE")):
            continue
        line = html.unescape(re.sub(r"<[^>]+>", "", line)).strip()
        if line and line != prev:  # auto-captions repeat rolling lines
            out.append(line)
            prev = line
    return " ".join(out)


def transcript_ytdlp_subs(url):
    ytdlp(["--skip-download", "--write-subs", "--write-auto-subs", "--sub-langs", "hi.*,en.*",
           "--sub-format", "vtt", "-o", str(WORK / "subs.%(ext)s"), url])
    files = sorted(WORK.glob("subs.*.vtt"), key=lambda p: (".hi" not in p.name, ".en" not in p.name, p.name))
    if not files:
        raise RuntimeError("no subtitle files downloaded")
    return vtt_to_text(files[0].read_text(encoding="utf-8", errors="ignore"))


def get_text(url, vid):
    errors = []
    for name, fn in (("YouTube transcript API", lambda: transcript_api(vid)),
                     ("yt-dlp subtitles", lambda: transcript_ytdlp_subs(url)),
                     ("Groq Whisper", lambda: transcript_whisper(url))):
        try:
            log(f"trying transcript source: {name}")
            t = fn()
            if len(t) > 500:
                log(f"transcript OK via {name}: {len(t)} chars")
                return t
            errors.append(f"{name}: too short")
        except Exception as e:
            log(f"{name} failed: {str(e)[:300]}")
            errors.append(f"{name}: {str(e)[:120]}")
    hint = ""
    if not Path("cookies.txt").exists():
        hint = ("\n\nYouTube ने GitHub का IP ब्लॉक किया है। रेपो में YT_COOKIES secret जोड़ें "
                "(cookies.txt का पूरा कंटेंट) और दोबारा चलाएँ।")
    raise RuntimeError("transcript नहीं मिला। " + " | ".join(errors) + hint)


def make_notes(text):
    chunks = [text[i:i + 14000] for i in range(0, len(text), 14000)]
    if len(chunks) == 1:
        return text
    notes = []
    for i, c in enumerate(chunks, 1):
        p = (f"This is section {i} of {len(chunks)} of a podcast transcript. Extract the key points as 8-12 concise "
             f"English bullet points. Keep names, numbers and examples. Output only the bullets.\n\nTRANSCRIPT:\n{c}")
        notes.append(f"[Section {i}]\n" + llm(p, json_mode=False).strip())
        time.sleep(2)
    return "\n\n".join(notes)


# ------------------------------------------------------------------ images
def probe_duration(path):
    return float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                      "-of", "default=nw=1:nk=1", str(path)]).strip())


def extract_frames(video, n_candidates=36, n_keep=8):
    from PIL import Image, ImageChops, ImageFilter, ImageStat
    dur = probe_duration(video)
    cands = []
    for i in range(n_candidates):
        t = dur * (i + 0.5) / n_candidates
        f = WORK / f"cand_{i:02d}.jpg"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", str(video),
                        "-frames:v", "1", "-q:v", "2", str(f)], capture_output=True)
        if not f.exists():
            continue
        try:
            g = Image.open(f).convert("L")
        except Exception:
            continue
        mid = g.resize((480, 270))
        mean = ImageStat.Stat(mid).mean[0]
        if mean < 25 or mean > 235:
            continue  # black / blown-out frame
        sharp = ImageStat.Stat(mid.filter(ImageFilter.FIND_EDGES)).stddev[0]
        cands.append((f, sharp, g.resize((32, 18))))

    def diff(a, b):
        return ImageStat.Stat(ImageChops.difference(a, b)).mean[0]

    chosen = []
    for f, s, small in sorted(cands, key=lambda x: -x[1]):
        if all(diff(small, c[2]) > 8 for c in chosen):
            chosen.append((f, s, small))
        if len(chosen) >= n_keep:
            break
    if len(chosen) < 5:  # frames too similar (e.g. static podcast set): relax
        for f, s, small in sorted(cands, key=lambda x: -x[1]):
            if f not in [c[0] for c in chosen]:
                chosen.append((f, s, small))
            if len(chosen) >= 5:
                break
    return sorted([c[0] for c in chosen], key=lambda p: p.name)


def get_images(url, meta):
    try:
        ytdlp(["-f", "bv*[height<=720]/b[height<=720]/b", "-o", str(WORK / "src.%(ext)s"), url])
        video = next(p for p in WORK.glob("src.*") if p.suffix.lower() in (".mp4", ".webm", ".mkv", ".mov"))
        frames = extract_frames(video)
        if frames:
            log(f"picked {len(frames)} frames from the video")
            try:
                video.unlink()  # free disk
            except OSError:
                pass
            return frames
    except Exception as e:
        log(f"frame extraction failed: {e}")
    log("using the YouTube thumbnail as the only image")
    r = requests.get(meta["thumb"], timeout=30)
    r.raise_for_status()
    p = WORK / "thumb_src.jpg"
    p.write_bytes(r.content)
    return [p]


def make_clips(images):
    clips = []
    imgs = list(images)
    if len(imgs) == 1:
        imgs = imgs * 2  # one zoom-in + one zoom-out so it is not monotonous
    frames = SEC_PER_IMG * FPS
    for idx, img in enumerate(imgs):
        out = WORK / f"clip_{idx}.mp4"
        z = "min(zoom+0.0005,1.12)" if idx % 2 == 0 else "if(lte(on,1),1.12,max(zoom-0.0005,1.0))"
        vf = (f"scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,"
              f"zoompan=z='{z}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s={W}x{H}:fps={FPS},format=yuv420p")
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(img), "-vf", vf, "-c:v", "libx264",
             "-preset", "veryfast", "-crf", "23", "-t", str(SEC_PER_IMG), str(out)])
        clips.append(out)
    return clips


# ------------------------------------------------------------------ audio + captions
async def tts_all(segments, pdir):
    import edge_tts
    paths = []
    for i, seg in enumerate(segments):
        mp3 = pdir / f"s{i:03d}.mp3"
        for a in range(4):
            try:
                await edge_tts.Communicate(seg["hi"], VOICE, rate=TTS_RATE).save(str(mp3))
                if mp3.exists() and mp3.stat().st_size > 500:
                    break
            except Exception as e:
                log(f"tts retry {a + 1} (segment {i}): {e}")
                await asyncio.sleep(2 * (a + 1))
        else:
            raise RuntimeError(f"TTS failed for segment {i}")
        paths.append(mp3)
    return paths


def build_audio(segments, pdir):
    """Returns (audio_wav_path, timeline[(start, end)], total_seconds)."""
    mp3s = asyncio.run(tts_all(segments, pdir))
    sr = 44100
    out_path = pdir / "audio.wav"
    out = wave.open(str(out_path), "wb")
    out.setnchannels(1)
    out.setsampwidth(2)
    out.setframerate(sr)
    silence = b"\x00\x00" * int(sr * PAUSE)
    t, timeline = 0.0, []
    for i, mp3 in enumerate(mp3s):
        wav = pdir / f"s{i:03d}.wav"
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3), "-ar", str(sr), "-ac", "1",
             "-c:a", "pcm_s16le", str(wav)])
        w = wave.open(str(wav), "rb")
        n = w.getnframes()
        data = w.readframes(n)
        w.close()
        dur = n / sr
        timeline.append((t, t + dur))
        out.writeframes(data)
        out.writeframes(silence)
        t += dur + PAUSE
    out.writeframes(b"\x00\x00" * sr)  # 1s tail
    out.close()
    return out_path, timeline, t + 1.0


def split_caption(text, max_words=CAPTION_WORDS):
    words = text.split()
    chunks = [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]
    if len(chunks) > 1 and len(chunks[-1].split()) <= 2:
        last = chunks.pop()
        chunks[-1] += " " + last
    return chunks


def srt_ts(t):
    ms_total = int(round(t * 1000))
    h, rem = divmod(ms_total, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def build_srt(segments, timeline, path):
    lines, n = [], 1
    for seg, (s, e) in zip(segments, timeline):
        chunks = split_caption(seg.get("en") or "")
        if not chunks:
            continue
        total = sum(len(c) for c in chunks)
        cur = s
        for c in chunks:
            d = (e - s) * len(c) / total
            lines.append(f"{n}\n{srt_ts(cur)} --> {srt_ts(cur + d)}\n{c}\n")
            cur += d
            n += 1
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def build_video(clips, pdir, total_dur, offset):
    need = math.ceil((total_dur + 1) / SEC_PER_IMG)
    with open(pdir / "list.txt", "w") as f:
        for k in range(need):
            f.write(f"file '{clips[(k + offset) % len(clips)].resolve()}'\n")
    style = "FontName=DejaVu Sans,FontSize=15,Bold=1,Outline=2,Shadow=0,MarginV=35,Alignment=2"
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", "list.txt", "-i", "audio.wav",
         "-vf", f"subtitles=cap.srt:force_style='{style}'",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", "video.mp4"], cwd=pdir)
    return pdir / "video.mp4"


# ------------------------------------------------------------------ thumbnail
def make_thumb(img_path, text, out):
    from PIL import Image, ImageDraw, ImageEnhance, ImageFont
    im = Image.open(img_path).convert("RGB")
    tw, th = im.size
    target = W / H
    if tw / th > target:
        nw = int(th * target)
        im = im.crop(((tw - nw) // 2, 0, (tw - nw) // 2 + nw, th))
    else:
        nh = int(tw / target)
        im = im.crop((0, (th - nh) // 2, tw, (th - nh) // 2 + nh))
    im = ImageEnhance.Brightness(im.resize((W, H))).enhance(0.7)
    font = None
    for fp in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
               "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"):
        if Path(fp).exists():
            font = ImageFont.truetype(fp, 120)
            break
    if font is None:
        font = ImageFont.load_default()
    d = ImageDraw.Draw(im)
    words = (text or "PODCAST SUMMARY").upper().split()
    mid = (len(words) + 1) // 2
    lines = [" ".join(words[:mid]), " ".join(words[mid:])] if len(words) > 2 else [" ".join(words)]
    y = H // 2 - 70 * len(lines)
    for ln in lines:
        d.text((60, y), ln, font=font, fill=(255, 221, 0), stroke_width=6, stroke_fill=(0, 0, 0))
        y += 140
    im.save(out, "JPEG", quality=90)


# ------------------------------------------------------------------ YouTube upload
def yt_service():
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    info = load_yt_info()
    creds = Credentials(
        None,
        refresh_token=info["refresh_token"],
        token_uri=info["token_uri"],
        client_id=info["client_id"],
        client_secret=info["client_secret"],
    )
    creds.refresh(Request())
    return build("youtube", "v3", credentials=creds)


def load_yt_info():
    """Accepts the full token.json (YT_TOKEN) and client_secrets.json (YT_CLIENT_SECRETS) pasted as-is,
    or the three separate values YT_REFRESH_TOKEN / YT_CLIENT_ID / YT_CLIENT_SECRET."""
    info = {"refresh_token": None, "client_id": None, "client_secret": None,
            "token_uri": "https://oauth2.googleapis.com/token"}
    tok = (os.getenv("YT_TOKEN") or "").strip()
    if tok:
        t = json.loads(tok)
        for k in ("refresh_token", "client_id", "client_secret", "token_uri"):
            info[k] = t.get(k) or info[k]
    cs = (os.getenv("YT_CLIENT_SECRETS") or "").strip()
    if cs:
        c = json.loads(cs)
        c = c.get("installed") or c.get("web") or c
        info["client_id"] = info["client_id"] or c.get("client_id")
        info["client_secret"] = info["client_secret"] or c.get("client_secret")
        info["token_uri"] = c.get("token_uri") or info["token_uri"]
    info["refresh_token"] = info["refresh_token"] or os.getenv("YT_REFRESH_TOKEN") or None
    info["client_id"] = info["client_id"] or os.getenv("YT_CLIENT_ID") or None
    info["client_secret"] = info["client_secret"] or os.getenv("YT_CLIENT_SECRET") or None
    missing = [k for k in ("refresh_token", "client_id", "client_secret") if not info[k]]
    if missing:
        raise RuntimeError("YouTube credentials missing: " + ", ".join(missing)
                           + " (set YT_TOKEN and YT_CLIENT_SECRETS secrets)")
    return info


def upload(yt, path, title, desc, tags, privacy, publish_at=None, thumb=None):
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload
    clean_tags, total = [], 0
    for t in tags:
        t = str(t).strip()
        if t and total + len(t) < 450:
            clean_tags.append(t)
            total += len(t)
    body = {
        "snippet": {"title": title[:100], "description": desc[:4900], "tags": clean_tags, "categoryId": "22",
                    "defaultLanguage": "hi", "defaultAudioLanguage": "hi"},
        "status": {"privacyStatus": privacy, "selfDeclaredMadeForKids": False},
    }
    if publish_at:
        body["status"]["privacyStatus"] = "private"
        body["status"]["publishAt"] = publish_at
    media = MediaFileUpload(str(path), chunksize=8 * 1024 * 1024, resumable=True, mimetype="video/mp4")
    req = yt.videos().insert(part="snippet,status", body=body, media_body=media)
    resp, tries = None, 0
    while resp is None:
        try:
            _, resp = req.next_chunk()
        except HttpError as e:
            if e.resp.status in (500, 502, 503, 504) and tries < 5:
                tries += 1
                time.sleep(5 * tries)
                continue
            raise
    vid = resp["id"]
    if thumb:
        try:
            yt.thumbnails().set(videoId=vid, media_body=MediaFileUpload(str(thumb))).execute()
        except Exception as e:
            log(f"thumbnail not set (channel may need verification): {e}")
    pl = os.getenv("YT_PLAYLIST_ID")
    if pl:
        try:
            yt.playlistItems().insert(part="snippet", body={"snippet": {
                "playlistId": pl, "resourceId": {"kind": "youtube#video", "videoId": vid}}}).execute()
        except Exception as e:
            log(f"playlist add failed: {e}")
    return vid


# ------------------------------------------------------------------ prompts
def outline_prompt(meta, notes, n):
    return (
        "You are the editor of a Hindi YouTube channel that makes summary videos of long podcasts.\n"
        f"Podcast title: {meta['title']}\nPodcast channel: {meta['channel']}\n\n"
        f"Key points / transcript of the podcast (in order):\n{notes}\n\n"
        f"Split the summary into exactly {n} part(s), by topic, in chronological order. Each part must stand on its own.\n"
        "Return JSON only, in this shape:\n"
        '{"video_title_hi": "...", "description_hi": "...", "tags": ["..."], '
        '"parts": [{"part": 1, "title_hi": "...", "thumb_text": "...", "focus": "...", "points": ["..."]}]}\n'
        "Rules: titles in Devanagari Hindi, catchy but truthful, max 70 characters. description_hi is 3-5 lines of Hindi. "
        "10-15 tags (mix Hindi and English). thumb_text is 2-4 ENGLISH words. points are 6-12 English bullets per part "
        "listing exactly what that part covers."
    )


def script_prompt(meta, notes, k, n, part):
    if n == 1:
        role = ("Start with a strong hook. Within the first segments mention that this is a summary of the podcast "
                f"\"{meta['title']}\" from {meta['channel']} and invite viewers to watch the original. "
                "End by thanking viewers and asking them to subscribe.")
    elif k == 1:
        role = ("Start with a strong hook. Within the first segments mention that this is a summary of the podcast "
                f"\"{meta['title']}\" from {meta['channel']} and invite viewers to watch the original. "
                "End with a short teaser for the next part.")
    elif k < n:
        role = "Start with a 2-sentence recap of the previous part. End with a short teaser for the next part."
    else:
        role = ("Start with a 2-sentence recap of the previous part. End with the key takeaway, thank the viewers "
                "and ask them to subscribe.")
    return (
        f"Write the Hindi voice-over script for part {k} of {n} of a podcast summary video.\n"
        f"Podcast: {meta['title']} ({meta['channel']})\n"
        f"This part's focus: {part.get('focus', '')}\n"
        "This part's key points:\n- " + "\n- ".join(part.get("points", [])) + "\n\n"
        f"Source material (use it only for facts about this part's focus):\n{notes}\n\n"
        "Rules:\n"
        f"- About {WORDS_PER_PART} words of Hindi (Devanagari), simple spoken style, no markdown.\n"
        "- Split into short segments of 1-2 sentences (max 30 words each).\n"
        "- Each segment has \"hi\" (Devanagari) and \"en\" (accurate, natural English translation of that same segment).\n"
        f"- {role}\n"
        "- Do not invent facts that are not in the source. Brand and person names may stay in Latin letters.\n"
        'Return JSON only: {"segments": [{"hi": "...", "en": "..."}]}'
    )


# ------------------------------------------------------------------ done list
def load_done():
    try:
        return json.loads(DONE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


# ------------------------------------------------------------------ main
def main():
    url = os.environ["VIDEO_URL"].strip()
    n_parts = int(os.getenv("NUM_PARTS", "2") or 2)
    privacy = os.getenv("PRIVACY", "public") or "public"
    delay_h = float(os.getenv("PART2_DELAY_HOURS", "0") or 0)
    force = os.getenv("FORCE", "false").lower() == "true"

    vid = extract_id(url)
    done = load_done()
    if vid in done and not force:
        tg(f"ℹ️ यह वीडियो पहले बन चुका है, छोड़ दिया:\n{url}")
        log("already processed; use force=true to redo")
        return

    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True)
    cookie_report()

    meta = get_meta(url, vid)
    log(f"podcast: {meta['title']} | {meta['channel']} | {meta['duration']}s")
    tg(f"▶️ Podcast summary शुरू: {meta['title']}\n{run_url()}")

    text = get_text(url, vid)
    notes = make_notes(text)

    plan = llm_json(outline_prompt(meta, notes, n_parts))
    parts = plan.get("parts") or []
    if not parts:
        raise RuntimeError("outline has no parts")
    n_parts = min(n_parts, len(parts))

    images = get_images(url, meta)
    clips = make_clips(images)
    yt = yt_service()

    results = []
    for k in range(1, n_parts + 1):
        part = parts[k - 1]
        log(f"--- part {k}/{n_parts}: {part.get('title_hi')}")
        script = llm_json(script_prompt(meta, notes, k, n_parts, part))
        segs = [s for s in script.get("segments", []) if (s.get("hi") or "").strip()]
        if len(segs) < 5:
            raise RuntimeError(f"part {k}: script too short ({len(segs)} segments)")

        pdir = WORK / f"part{k}"
        pdir.mkdir()
        _, timeline, total = build_audio(segs, pdir)
        build_srt(segs, timeline, pdir / "cap.srt")
        video = build_video(clips, pdir, total, offset=(k - 1) * 3)
        log(f"part {k} video: {total / 60:.1f} min")

        thumb = pdir / "thumb.jpg"
        make_thumb(images[(k - 1) % len(images)], part.get("thumb_text"), thumb)

        base_title = part.get("title_hi") or plan.get("video_title_hi") or meta["title"]
        title = f"{base_title} | भाग {k}" if n_parts > 1 else base_title
        desc = (
            (plan.get("description_hi") or "") +
            (f"\n\n📌 भाग {k}/{n_parts}" if n_parts > 1 else "") +
            f"\n\n🎙️ मूल पॉडकास्ट: {meta['title']} — {meta['channel']}\n{url}\n"
            "(यह मूल पॉडकास्ट का हिंदी सारांश है। पूरी बातचीत के लिए ऊपर दिया ओरिजिनल वीडियो देखें।)"
        )
        publish_at = None
        if k >= 2 and delay_h > 0:
            publish_at = (datetime.now(timezone.utc) + timedelta(hours=delay_h * (k - 1))).strftime(
                "%Y-%m-%dT%H:%M:%S.000Z")
        yid = upload(yt, video, title, desc, plan.get("tags", []), privacy, publish_at, thumb)
        results.append({"part": k, "id": yid, "title": title, "scheduled": publish_at})
        log(f"uploaded: https://youtu.be/{yid}")
        shutil.rmtree(pdir, ignore_errors=True)  # free disk

    done[vid] = {"url": url, "title": meta["title"], "date": datetime.now(timezone.utc).isoformat(),
                 "uploads": results}
    DONE_FILE.write_text(json.dumps(done, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [f"✅ Podcast summary तैयार: {meta['title']}"]
    for r in results:
        extra = f" (शेड्यूल: {r['scheduled']})" if r["scheduled"] else ""
        lines.append(f"भाग {r['part']}: https://youtu.be/{r['id']}{extra}")
    tg("\n".join(lines))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        traceback.print_exc()
        tg(f"❌ Podcast bot फेल हुआ\n{os.getenv('VIDEO_URL', '')}\n\n{type(exc).__name__}: {str(exc)[-700:]}\n{run_url()}")
        sys.exit(1)
