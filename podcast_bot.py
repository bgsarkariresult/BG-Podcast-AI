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

# Comma-separated list, tried in order; a model that returns 404 (retired) is skipped automatically.
GEMINI_MODELS = [m.strip() for m in (os.getenv("GEMINI_MODEL") or "gemini-3.6-flash,gemini-3.8-flash").split(",")
                 if m.strip()]
# Groq model names change often; leave these unset to auto-pick from Groq's live model list.
GROQ_LLM = os.getenv("GROQ_LLM_MODEL") or ""
GROQ_STT = os.getenv("GROQ_STT_MODEL") or ""
VOICE = os.getenv("TTS_VOICE", "hi-IN-MadhurNeural")
TTS_RATE = os.getenv("TTS_RATE", "+0%")
WORDS_PER_PART = int(os.getenv("WORDS_PER_PART", "900"))  # ~6 min of Hindi speech
MIN_PART_SEC = int(os.getenv("MIN_PART_SEC", "240"))  # a part shorter than this is regenerated once (long is fine)

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
class ModelNotFound(Exception):
    pass


def _gemini_one(prompt, json_mode, model):
    """Round-robin over all Gemini keys; rotate to the next key on any failure."""
    global _key_i
    if not GEMINI_KEYS:
        raise RuntimeError("no Gemini keys set")
    last = None
    for attempt in range(len(GEMINI_KEYS) * 4):
        key = GEMINI_KEYS[_key_i % len(GEMINI_KEYS)]
        body = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.7}}
        if json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"
        try:
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
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
        last = f"{r.status_code} {r.text[:300]}"
        log(f"Gemini {model} key#{(_key_i - 1) % len(GEMINI_KEYS) + 1}: HTTP {r.status_code} {r.text[:200]!r}")
        if r.status_code == 404:
            raise ModelNotFound(f"{model}: {r.text[:200]}")
        if r.status_code == 429 and 'limit": 0' in r.text:  # this project has no free quota for this model
            raise ModelNotFound(f"{model}: no free quota ({r.text[:160]})")
        wait = 5
        if r.status_code == 429:
            m = re.search(r'retryDelay"?\s*:\s*"?(\d+)', r.text)
            wait = min(int(m.group(1)) + 2, 70) if m else 15
            if attempt < len(GEMINI_KEYS) - 1:  # other keys are still fresh: switch quickly
                wait = 2
        elif r.status_code not in (500, 503):
            wait = 1
        time.sleep(wait)
    raise RuntimeError(f"Gemini ({model}) failed: " + str(last))


def gemini(prompt, json_mode=True):
    if not GEMINI_KEYS:
        raise RuntimeError("no Gemini keys set")
    last = None
    for model in GEMINI_MODELS:
        try:
            return _gemini_one(prompt, json_mode, model)
        except ModelNotFound as e:
            log(f"Gemini model not available, trying next: {str(e)[:160]}")
            last = e
        except RuntimeError as e:
            last = e
    raise RuntimeError(f"Gemini failed on all models {GEMINI_MODELS}: {last}")


_GROQ_MODELS = None


def groq_models():
    global _GROQ_MODELS
    if _GROQ_MODELS is None:
        _GROQ_MODELS = []
        try:
            r = requests.get("https://api.groq.com/openai/v1/models",
                             headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"}, timeout=30)
            if r.status_code == 200:
                _GROQ_MODELS = [m["id"] for m in r.json().get("data", [])]
        except Exception as e:
            log(f"could not list Groq models: {e}")
        log(f"Groq models available: {_GROQ_MODELS}")
    return _GROQ_MODELS


def groq_llm_model():
    if GROQ_LLM:
        return GROQ_LLM
    bad = ("whisper", "tts", "guard", "orpheus", "playai", "embed", "safeguard")
    ids = [m for m in groq_models() if not any(b in m.lower() for b in bad)]
    for pat in ("llama-3.3-70b", "70b", "gpt-oss-120b", "llama", "qwen", "gpt-oss"):
        for m in ids:
            if pat in m.lower():
                return m
    return ids[0] if ids else "llama-3.3-70b-versatile"


def groq_stt_model():
    if GROQ_STT:
        return GROQ_STT
    ids = [m for m in groq_models() if "whisper" in m.lower()]
    for pat in ("large-v3-turbo", "large-v3", "whisper"):
        for m in ids:
            if pat in m.lower():
                return m
    return ids[0] if ids else "whisper-large-v3-turbo"


def groq_llm(prompt, json_mode=True):
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY missing")
    body = {"model": groq_llm_model(), "messages": [{"role": "user", "content": prompt}], "temperature": 0.7}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    last = None
    for a in range(4):
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"}, json=body, timeout=180,
        )
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        last = f"{r.status_code} {r.text[:200]}"
        if r.status_code == 413:  # over Groq's tokens-per-minute limit: shrink the prompt and retry
            m = re.search(r"Limit (\d+), Requested (\d+)", r.text)
            ratio = int(m.group(1)) / int(m.group(2)) * 0.75 if m else 0.6
            prompt = shrink_prompt(prompt, ratio)
            body["messages"][0]["content"] = prompt
            log(f"Groq 413: shrinking prompt to {len(prompt)} chars and retrying")
            continue
        time.sleep(10 * (a + 1))
    raise RuntimeError("Groq LLM failed: " + str(last))


def shrink_prompt(prompt, ratio):
    """Emergency trim for the Groq fallback: keep the start (instructions) and the end (output format)."""
    n = max(2000, int(len(prompt) * ratio))
    head = int(n * 0.65)
    return prompt[:head] + "\n...[trimmed to fit the model limit]...\n" + prompt[-(n - head):]


def gemini_retry(prompt, json_mode=True, rounds=3):
    """Gemini 503 ("high demand") and 429 are usually temporary: wait and retry before giving up."""
    err = None
    for rnd in range(rounds):
        try:
            return gemini(prompt, json_mode)
        except Exception as e:
            err = e
            log(f"Gemini round {rnd + 1}/{rounds} failed: {str(e)[:400]}")
            if not GEMINI_KEYS:
                break
            if rnd < rounds - 1:
                time.sleep(40 * (rnd + 1))
    raise err if err else RuntimeError("Gemini unavailable")


def llm(prompt, json_mode=True):
    try:
        return gemini_retry(prompt, json_mode)
    except Exception as err:
        log("Gemini unavailable; falling back to Groq")
        try:
            return groq_llm(prompt, json_mode)
        except Exception as e2:
            raise RuntimeError(f"Gemini: {str(err)[:380]} || Groq: {str(e2)[:380]}")


GROQ_PIECE = 6000  # chars per Groq request, so one request stays under the free 8,000 tokens-per-minute limit


def llm_notes(prompt_for, text):
    """Bullet-point notes for one transcript chunk. Gemini first; if it stays down, Groq processes the SAME chunk
    in small pieces (nothing is trimmed away), waiting between pieces to respect Groq's per-minute token limit."""
    try:
        return gemini_retry(prompt_for(text), json_mode=False)
    except Exception as err:
        log(f"Gemini unavailable for notes ({str(err)[:200]}); using Groq in small pieces")
    pieces = [text[i:i + GROQ_PIECE] for i in range(0, len(text), GROQ_PIECE)]
    out = []
    for j, piece in enumerate(pieces, 1):
        out.append(groq_llm(prompt_for(piece), json_mode=False).strip())
        log(f"Groq notes piece {j}/{len(pieces)} done")
        if j < len(pieces):
            time.sleep(25)
    return "\n".join(out)


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
    p = subprocess.run(base + js_args() + args, capture_output=True, text=True)
    if p.returncode == 0:
        return p.stdout
    log("---- yt-dlp stderr (tail) ----\n" + p.stderr[-2500:] + "\n---- end ----")
    lines = [ln.strip() for ln in p.stderr.splitlines()]
    errs = [ln for ln in lines if ln.startswith("ERROR")]
    warns = [ln for ln in lines if ln.startswith("WARNING") or "cookie" in ln.lower()]
    raise RuntimeError("yt-dlp failed: " + " || ".join(errs + warns)[:1200])


_JS_ARGS = None


def js_args():
    """YouTube needs a JavaScript runtime; yt-dlp only auto-enables deno, so pass node explicitly."""
    global _JS_ARGS
    if _JS_ARGS is None:
        _JS_ARGS = []
        node = shutil.which("node")
        if node:
            helptext = subprocess.run(["yt-dlp", "--help"], capture_output=True, text=True).stdout
            if "--js-runtimes" in helptext:
                _JS_ARGS = ["--js-runtimes", "node"]
                if "--remote-components" in helptext:  # lets yt-dlp fetch the n-challenge solver if needed
                    _JS_ARGS += ["--remote-components", "ejs:github"]
        log(f"yt-dlp JS runtime args: {_JS_ARGS or 'NONE'} (node at {node})")
    return _JS_ARGS


COOKIES_LOGGED_IN = False


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
    log("cookie names in file: " + ", ".join(sorted(names)))
    global COOKIES_LOGGED_IN
    COOKIES_LOGGED_IN = any(k in names for k in keys)
    if not COOKIES_LOGGED_IN:
        log("WARNING: cookies.txt has NO login cookies -> these are anonymous cookies, YouTube will still block")


def extract_id(url):
    m = re.search(r"(?:v=|youtu\.be/|shorts/|embed/|live/)([A-Za-z0-9_-]{11})", url)
    if m:
        return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url.strip()):
        return url.strip()
    raise ValueError(f"cannot find a YouTube video id in: {url}")


def get_meta(url, vid):
    try:
        info = json.loads(ytdlp(["--dump-single-json", "--skip-download", "--ignore-no-formats-error", url]))
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
         "-f", "segment", "-segment_time", "300", "-reset_timestamps", "1", str(WORK / "aseg_%03d.mp3")])
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY missing for Whisper fallback")
    segs = sorted(WORK.glob("aseg_*.mp3"))
    lang = os.getenv("WHISPER_LANG") or ""  # optional language hint, e.g. "hi" or "en"
    texts = []
    for seg in segs:
        text = ""
        for attempt in range(6):
            data = {"model": groq_stt_model(), "response_format": "json", "temperature": "0"}
            if lang:
                data["language"] = lang
            with open(seg, "rb") as f:
                r = requests.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {key}"},
                    files={"file": (seg.name, f, "audio/mpeg")},
                    data=data, timeout=300,
                )
            if r.status_code == 200:
                text = r.json().get("text", "")
                if len(text) < 150 and attempt == 0:  # near-empty 5-min chunk: try once more
                    log(f"whisper {seg.name}: only {len(text)} chars, retrying once")
                    continue
                break
            log(f"whisper {seg.name}: {r.status_code} {r.text[:150]}")
            try:  # honour Groq's rate-limit wait time (free tier has an audio-seconds-per-hour cap)
                wait = int(float(r.headers.get("retry-after", ""))) + 2
            except ValueError:
                wait = 20 * (attempt + 1)
            time.sleep(min(wait, 600))
        else:
            raise RuntimeError(f"Whisper failed for {seg.name}")
        texts.append(text)
        log(f"whisper {seg.name}: {len(text)} chars")
    total = sum(len(t) for t in texts)
    log(f"whisper total: {total} chars for ~{len(segs) * 5} min of audio "
        f"(~{total // max(1, len(segs) * 5)} chars/min)")
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
    for attempt in range(3):
        try:
            ytdlp(["--skip-download", "--ignore-no-formats-error", "--write-subs", "--write-auto-subs",
                   "--sub-langs", "hi-orig,hi,en-orig,en", "--sleep-subtitles", "2",
                   "--sub-format", "vtt", "-o", str(WORK / "subs.%(ext)s"), url])
            break
        except RuntimeError as e:
            if "429" in str(e) and attempt < 2:  # YouTube rate limit: wait and retry
                wait = 45 * (attempt + 1)
                log(f"subtitles got 429, waiting {wait}s before retry {attempt + 2}/3")
                time.sleep(wait)
                continue
            raise
    order = ["hi-orig", "hi", "en-orig", "en"]

    def rank(p):
        lang = p.name[len("subs."):-len(".vtt")]
        return order.index(lang) if lang in order else 99

    files = sorted(WORK.glob("subs.*.vtt"), key=rank)
    if not files:
        raise RuntimeError("no subtitle files downloaded")
    return vtt_to_text(files[0].read_text(encoding="utf-8", errors="ignore"))


CACHE_DIR = Path("cache")


def cache_get(vid):
    p = CACHE_DIR / f"{vid}.txt"
    if p.exists():
        t = p.read_text(encoding="utf-8", errors="ignore")
        if len(t) > 2000:
            return t
    return None


def cache_put(vid, text):
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        (CACHE_DIR / f"{vid}.txt").write_text(text, encoding="utf-8")
    except OSError as e:
        log(f"could not write transcript cache: {e}")


def get_text(url, vid, force=False):
    if not force:
        cached = cache_get(vid)
        if cached:
            log(f"transcript from cache: {len(cached)} chars (use force=true to fetch again)")
            return cached
    errors = []
    for name, fn in (("YouTube transcript API", lambda: transcript_api(vid)),
                     ("yt-dlp subtitles", lambda: transcript_ytdlp_subs(url)),
                     ("Groq Whisper", lambda: transcript_whisper(url))):
        try:
            log(f"trying transcript source: {name}")
            t = fn()
            if len(t) > 500:
                log(f"transcript OK via {name}: {len(t)} chars")
                cache_put(vid, t)
                return t
            errors.append(f"{name}: too short")
        except Exception as e:
            log(f"{name} failed: {str(e)[:600]}")
            errors.append(f"{name}: {str(e)[:220]}")
    if not Path("cookies.txt").exists():
        hint = ("\n\nYouTube ने GitHub का IP ब्लॉक किया है। रेपो में YT_COOKIES secret जोड़ें "
                "(cookies.txt का पूरा कंटेंट) और दोबारा चलाएँ।")
    elif not COOKIES_LOGGED_IN:
        hint = ("\n\ncookies.txt में लॉगिन वाली cookies नहीं हैं (सिर्फ़ अनाम cookies)। "
                "नए Google अकाउंट से इन्कॉग्निटो में लॉगिन करके नई cookies एक्सपोर्ट करें।")
    elif any("Sign in to confirm" in e for e in errors[1:]):
        hint = ("\n\ncookies लॉगिन वाली हैं, फिर भी YouTube बॉट-जाँच लगा रहा है: cookies पुरानी/अमान्य हो सकती हैं "
                "(नई निकालें) या GitHub का IP सख़्ती से ब्लॉक है (तब self-hosted runner चाहिए)।")
    else:
        hint = ("\n\nbॉट-ब्लॉक का एरर नहीं है। yt-dlp को वीडियो के format नहीं मिले "
                "(JS challenge solver की दिक्कत)। पूरा लॉग देखें।")
    raise RuntimeError("transcript नहीं मिला।" + hint + "\n\n" + " | ".join(errors))


def make_notes(text):
    size = 30000 if GEMINI_KEYS else 9000  # Gemini takes big chunks; Groq's free limits need small ones
    chunks = [text[i:i + size] for i in range(0, len(text), size)]
    if len(chunks) == 1:
        return text
    notes = []
    for i, c in enumerate(chunks, 1):
        def prompt_for(t, i=i):
            return (f"This is part of section {i} of {len(chunks)} of a podcast transcript. Extract the key points as "
                    f"8-12 concise English bullet points. Keep names, numbers and examples. Output only the "
                    f"bullets.\n\nTRANSCRIPT:\n{t}")
        notes.append(f"[Section {i}]\n" + llm_notes(prompt_for, c).strip())
        time.sleep(6)  # stay under the free-tier requests-per-minute limit
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
        "Rules: titles are in easy everyday Hinglish written in Devanagari script (simple words people use daily, "
        "English words like 'पॉडकास्ट', 'इंडिया', 'मनी' are fine; no heavy Sanskritized Hindi), catchy but truthful, "
        "max 70 characters. description_hi is 3-5 lines in the same easy Hinglish (Devanagari). "
        "10-15 tags (mix Hindi and English). thumb_text is 2-4 ENGLISH words. points are 6-12 English bullets per part "
        "listing exactly what that part covers."
    )


STYLE_HINGLISH = (
    "LANGUAGE STYLE (very important): write in easy, everyday Hinglish, the way a friendly YouTuber talks - simple "
    "Hindi sentences mixed with common English words - but ALWAYS in Devanagari script. Use the English words people "
    "really use in daily talk (written in Devanagari) instead of heavy or formal Hindi. Examples: 'मेन पॉइंट' not "
    "'मुख्य बिंदु'; 'स्पीड' not 'रफ़्तार'; 'टॉपिक' not 'विषय'; 'प्रॉब्लम' not 'समस्या'; 'आइडिया' not 'विचार'; "
    "'इकोनॉमी' not 'अर्थव्यवस्था'; 'रिलेशनशिप' not 'संबंध'; 'सक्सेस' not 'सफलता'; 'इंपॉर्टेंट' not 'महत्वपूर्ण'. "
    "Avoid difficult Sanskritized words (अत्यंत, तत्पश्चात, परंतु, किंतु, अतः, इत्यादि); say 'बहुत', 'फिर', 'लेकिन', "
    "'इसलिए' instead. Keep sentences short and clear so a common viewer understands on the first listen. "
    "Write EVERY English word, and names of people, countries and brands, in Devanagari as they are pronounced "
    "(e.g. 'इंडिया', 'चाइना', 'पॉडकास्ट'). Do NOT use Roman/Latin letters inside the 'hi' field."
)


def script_prompt(meta, notes, k, n, part, words=None):
    words = words or WORDS_PER_PART
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
        f"- At least {words} words in total (a longer, complete explanation is fine; do not rush or skip important points).\n"
        "- Split into short segments of 1-2 sentences (max 30 words each).\n"
        "- Each segment has \"hi\" (easy Hinglish in Devanagari) and \"en\" (accurate, natural English translation "
        "of that same segment, in plain English).\n"
        f"- {role}\n"
        "- Do not invent facts that are not in the source.\n"
        f"- {STYLE_HINGLISH}\n"
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

    text = get_text(url, vid, force)
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
        pdir = WORK / f"part{k}"
        pdir.mkdir()
        words = WORDS_PER_PART
        for attempt in range(2):
            script = llm_json(script_prompt(meta, notes, k, n_parts, part, words))
            segs = [s for s in script.get("segments", []) if (s.get("hi") or "").strip()]
            if len(segs) < 5:
                raise RuntimeError(f"part {k}: script too short ({len(segs)} segments)")
            _, timeline, total = build_audio(segs, pdir)
            # Long parts are fine; only a part that is too SHORT gets one retry with a bigger word target.
            if total >= MIN_PART_SEC or attempt == 1:
                break
            words = int(words * max(1.3, MIN_PART_SEC * 1.15 / total))
            log(f"part {k} is only {total / 60:.1f} min (< {MIN_PART_SEC / 60:.0f} min); "
                f"regenerating with ~{words} words")
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
        results.append({"part": k, "id": yid, "title": title, "scheduled": publish_at,
                        "minutes": round(total / 60, 1)})
        log(f"uploaded: https://youtu.be/{yid}")
        shutil.rmtree(pdir, ignore_errors=True)  # free disk

    done[vid] = {"url": url, "title": meta["title"], "date": datetime.now(timezone.utc).isoformat(),
                 "uploads": results}
    DONE_FILE.write_text(json.dumps(done, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [f"✅ Podcast summary तैयार: {meta['title']}"]
    for r in results:
        extra = f" (शेड्यूल: {r['scheduled']})" if r["scheduled"] else ""
        lines.append(f"भाग {r['part']} ({r['minutes']} मिनट): https://youtu.be/{r['id']}{extra}")
    tg("\n".join(lines))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        traceback.print_exc()
        tg(f"❌ Podcast bot फेल हुआ\n{os.getenv('VIDEO_URL', '')}\n\n{type(exc).__name__}: {str(exc)[:900]}\n{run_url()}")
        sys.exit(1)
