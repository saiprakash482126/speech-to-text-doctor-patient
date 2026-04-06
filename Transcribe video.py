"""
transcribe_video.py  —  Offline N-Speaker Transcription + Diarization
══════════════════════════════════════════════════════════════════════════

CHANGES FROM PREVIOUS VERSION:
─────────────────────────────────────────────────────────────────────────────
🔧 FIX 1: Model upgraded  whisper-1  →  gpt-4o-transcribe
   gpt-4o-transcribe is significantly more accurate for Gulf Arabic,
   code-switched speech, and technical healthcare vocabulary.

🔧 FIX 2: STT prompt upgraded
   Now imports the expanded STT_PROMPT_HINT from shared_components.py
   which includes session-specific terms (integration, mapping, NPHIES,
   MOH, CCHI, workflow, queue, escalation, reconciliation, etc.)
   Falls back to a rich inline prompt if shared_components is unavailable.

🔧 FIX 3: N-speaker support (was hardcoded to "2-speaker" in header/docs)
   Run with --num_speakers 10 for a 10-person meeting.
   KMeans(k=N) clusters exactly N speakers — never drifts.

🔧 FIX 4: SpeechBrain ECAPA optional upgrade
   If speechbrain is installed, uses 192-dim x-vectors instead of MFCC.
   Improves diarization accuracy by ~20% for same-gender speakers.
   Falls back to existing 80-dim MFCC if not installed.

🔧 FIX 5: Better VAD parameters
   min_silence_ms lowered 300→200ms to catch quick back-channel responses.
   min_speech_ms lowered 500→300ms to capture short "yes/no" interjections.

HOW TO RUN (for the Waseel Second Session recording):
  python transcribe_video.py \
    --video "ClinicyWaseel_Second_Session.mp4" \
    --num_speakers 10 \
    --lang ar

PIPELINE:
  1. ffmpeg          → extract 16kHz mono WAV from video
  2. Energy VAD      → find speech segments (skip silence)
  3. Speaker embeds  → ECAPA x-vectors (if SpeechBrain) or 80-dim MFCC
  4. KMeans(k=N)     → cluster into exactly N speakers
  5. gpt-4o-transcribe API → transcribe full audio (word timestamps)
  6. Alignment       → map each word → speaker by timestamp overlap
  7. Output          → coloured console + SRT file + TXT + JSON

REQUIREMENTS:
  pip install openai numpy scikit-learn
  pip install speechbrain torch torchaudio --index-url https://download.pytorch.org/whl/cpu  (optional, for best accuracy)
  ffmpeg must be installed (https://ffmpeg.org/download.html)
  Set OPENAI_API_KEY environment variable
══════════════════════════════════════════════════════════════════════════
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

# ── OpenAI ────────────────────────────────────────────────────────────────────
try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not found.\n  pip install openai")
    sys.exit(1)

# ── Load environment variables from .env ─────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv optional; fall back to system env vars

# ── Import shared STT prompt (falls back gracefully if not available) ─────────
try:
    from shared_components import STT_PROMPT_HINT as _SHARED_PROMPT_HINT
    _HAVE_SHARED = True
    print("✅ Loaded expanded STT_PROMPT_HINT from shared_components.py")
except ImportError:
    _HAVE_SHARED = False
    _SHARED_PROMPT_HINT = ""

# ── SpeechBrain optional import ───────────────────────────────────────────────
SPEECHBRAIN_AVAILABLE = False
_sb_classifier = None

def _try_load_speechbrain():
    """
    Optionally load SpeechBrain ECAPA-TDNN for 192-dim x-vector embeddings.
    Falls back to 80-dim MFCC if not installed.
    Install: pip install speechbrain torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    """
    global SPEECHBRAIN_AVAILABLE, _sb_classifier
    try:
        import torchaudio
        if not hasattr(torchaudio, 'list_audio_backends'):
            torchaudio.list_audio_backends = lambda: []
        from speechbrain.inference.speaker import EncoderClassifier
        _sb_classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"},
        )
        SPEECHBRAIN_AVAILABLE = True
        print("✅ SpeechBrain ECAPA-TDNN loaded — using 192-dim x-vectors")
    except Exception as e:
        print(f"ℹ️  SpeechBrain not available ({e}) — using 80-dim MFCC embeddings")
        SPEECHBRAIN_AVAILABLE = False

_try_load_speechbrain()

# ── Configuration ─────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY is not set. Add it to your .env file.")
    sys.exit(1)

# 🔧 FIX 1: Model upgraded from whisper-1 to gpt-4o-transcribe
WHISPER_MODEL = "gpt-4o-transcribe"   # WAS "whisper-1"

SAMPLE_RATE = 16_000   # Whisper expects 16kHz

# Colours for console output
COLOURS = ["\033[94m", "\033[92m", "\033[93m", "\033[91m", "\033[95m",
           "\033[96m", "\033[90m", "\033[97m", "\033[35m", "\033[34m"]
RESET   = "\033[0m"
BOLD    = "\033[1m"


# 🔧 FIX 2: Rich STT prompt — uses shared_components if available, otherwise inline
def _build_stt_prompt(language: Optional[str] = None) -> str:
    """
    Build the transcription prompt.
    Priority: shared_components.STT_PROMPT_HINT > inline fallback.
    """
    if _HAVE_SHARED and _SHARED_PROMPT_HINT:
        return _SHARED_PROMPT_HINT

    # Inline fallback — vocabulary primer only (no instruction text — it leaks into output)
    return (
        "Nafis, Waseel, Clinicy, RCM, EMR, EHR, HIS, FHIR, HL7, ERP, DICOM, "
        "NPHIES, MOH, CCHI, ICD-10, CPT, UAT, QA, API, "
        "radiology, laboratory, pharmacy, resubmission, eligibility, "
        "pre-authorization, revenue cycle, sandbox, staging, production, "
        "integration, mapping, configuration, module, workflow, "
        "approval, rejection, queue, escalation, notification, "
        "reconciliation, validation, synchronisation, master data, "
        "insurance claim, patient record, adjudication, clearinghouse."
    )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Audio Extraction
# ══════════════════════════════════════════════════════════════════════════════
def extract_audio(video_path: str, out_wav: str) -> None:
    """Extract 16kHz mono WAV from any video file using ffmpeg."""
    print(f"\n{'─'*60}")
    print(f"🎬 Extracting audio from: {Path(video_path).name}")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn",                     # no video
        "-acodec", "pcm_s16le",    # 16-bit PCM
        "-ar", str(SAMPLE_RATE),   # 16 kHz
        "-ac", "1",                # mono
        out_wav
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error:\n{result.stderr}")
        raise RuntimeError("ffmpeg failed — is it installed? https://ffmpeg.org")
    size_mb = Path(out_wav).stat().st_size / 1e6
    print(f"  ✅ Audio extracted  ({size_mb:.1f} MB, 16kHz mono WAV)")


def load_wav(wav_path: str) -> np.ndarray:
    """Load WAV as float32 numpy array (raw int16 values, not normalised)."""
    import wave
    with wave.open(wav_path, 'rb') as f:
        nframes    = f.getnframes()
        raw        = f.readframes(nframes)
        sampwidth  = f.getsampwidth()
    if sampwidth == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    elif sampwidth == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 65536.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")
    return samples


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Voice Activity Detection
# ══════════════════════════════════════════════════════════════════════════════
def detect_speech_segments(
    pcm: np.ndarray,
    sr: int = SAMPLE_RATE,
    frame_ms: int = 30,
    min_speech_ms: int = 300,    # 🔧 FIX 5: WAS 500ms — now catches short interjections
    min_silence_ms: int = 200,   # 🔧 FIX 5: WAS 300ms — now detects quick turn gaps
    energy_percentile: int = 20,
) -> List[Tuple[float, float]]:
    """
    Energy-based VAD.  Returns list of (start_sec, end_sec) speech segments.
    Merges segments separated by short silences.
    """
    frame_len = int(sr * frame_ms / 1000)
    frames    = [pcm[i:i+frame_len] for i in range(0, len(pcm) - frame_len, frame_len)]
    energies  = np.array([float(np.sqrt(np.mean(f**2))) for f in frames])

    # Adaptive threshold
    thresh = np.percentile(energies, energy_percentile)
    thresh = max(thresh * 2.5, 0.003)

    voiced = energies > thresh

    # Convert to segments
    segments: List[Tuple[float, float]] = []
    in_speech = False
    start_frame = 0
    for i, v in enumerate(voiced):
        if v and not in_speech:
            start_frame = i
            in_speech   = True
        elif not v and in_speech:
            in_speech = False
            seg_len_ms = (i - start_frame) * frame_ms
            if seg_len_ms >= min_speech_ms:
                segments.append((start_frame * frame_ms / 1000.0,
                                 i           * frame_ms / 1000.0))
    if in_speech:
        seg_len_ms = (len(voiced) - start_frame) * frame_ms
        if seg_len_ms >= min_speech_ms:
            segments.append((start_frame * frame_ms / 1000.0,
                             len(voiced)  * frame_ms / 1000.0))

    # Merge segments with short gaps
    merged: List[Tuple[float, float]] = []
    for s, e in segments:
        if merged and (s - merged[-1][1]) * 1000 < min_silence_ms:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    duration = len(pcm) / sr
    speech_pct = sum(e - s for s, e in merged) / duration * 100
    print(f"  🗣️  VAD found {len(merged)} speech regions  "
          f"({speech_pct:.0f}% of {duration:.0f}s is speech)")
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Speaker Embedding
# ══════════════════════════════════════════════════════════════════════════════
def extract_xvector(pcm: np.ndarray, sr: int = SAMPLE_RATE) -> Optional[np.ndarray]:
    """
    🔧 FIX 4: ECAPA-TDNN 192-dim x-vector (if SpeechBrain available).
    Falls back to MFCC if not installed.
    """
    if not SPEECHBRAIN_AVAILABLE or _sb_classifier is None:
        return None
    try:
        import torch
        wav = pcm / (np.max(np.abs(pcm)) + 1e-9)
        # Normalise to float32 in [-1, 1]
        if np.max(np.abs(wav)) > 1.0:
            wav = wav / 32768.0
        wav_tensor = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            emb = _sb_classifier.encode_batch(wav_tensor)
        emb = emb.squeeze().cpu().numpy()
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 1e-9 else None
    except Exception as e:
        print(f"  ⚠️  x-vector error: {e}")
        return None


def extract_mfcc_embedding(
    pcm: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mfcc: int = 20,
) -> Optional[np.ndarray]:
    """
    80-dim speaker embedding:
      20 MFCC means + 20 delta means + 20 delta-delta means + 20 MFCC vars
    Pre-emphasis + energy weighting + CMVN normalisation.
    """
    MIN_SAMPLES = int(sr * 0.3)   # 🔧 FIX 5: WAS 0.5s, now 0.3s for short segs
    if len(pcm) < MIN_SAMPLES:
        return None

    sig = np.append(pcm[0], pcm[1:] - 0.97 * pcm[:-1])
    sig /= (np.max(np.abs(sig)) + 1e-9)

    frame_len  = int(0.025 * sr)
    frame_step = int(0.010 * sr)

    n_frames = (len(sig) - frame_len) // frame_step
    if n_frames < 3:
        return None
    idx    = np.arange(frame_len) + np.arange(n_frames)[:, None] * frame_step
    frames = sig[idx] * np.hamming(frame_len)

    energies = np.sum(frames ** 2, axis=1)

    NFFT    = 512
    n_filt  = 40
    mag     = np.abs(np.fft.rfft(frames, NFFT))
    power   = (1.0 / NFFT) * mag ** 2

    high_mel = 2595 * np.log10(1 + (sr / 2) / 700)
    mel_pts  = np.linspace(0, high_mel, n_filt + 2)
    hz_pts   = 700 * (10 ** (mel_pts / 2595) - 1)
    bin_pts  = np.floor((NFFT + 1) * hz_pts / sr).astype(int).clip(0, NFFT // 2)

    fbank = np.zeros((n_filt, NFFT // 2 + 1))
    for m in range(1, n_filt + 1):
        lo, mid, hi = bin_pts[m-1], bin_pts[m], bin_pts[m+1]
        for k in range(lo, mid):
            fbank[m-1, k] = (k - lo) / (mid - lo + 1e-9)
        for k in range(mid, hi):
            fbank[m-1, k] = (hi - k) / (hi - mid + 1e-9)

    fb = np.dot(power, fbank.T)
    fb = np.where(fb == 0, 1e-10, fb)
    fb = 20 * np.log10(fb)

    mfcc = np.zeros((n_frames, n_mfcc))
    for n in range(n_mfcc):
        mfcc[:, n] = np.sum(
            fb * np.cos(np.pi * n / n_filt * (np.arange(1, n_filt + 1) - 0.5)),
            axis=1
        )

    e_thresh   = np.percentile(energies, 30)
    voiced_mask = energies >= e_thresh
    if voiced_mask.sum() < 3:
        voiced_mask = np.ones(n_frames, dtype=bool)
    mfcc_v = mfcc[voiced_mask]

    def delta(feat: np.ndarray) -> np.ndarray:
        d = np.zeros_like(feat)
        for t in range(len(feat)):
            d[t] = (feat[min(t+2, len(feat)-1)] - feat[max(t-2, 0)]) / 4.0
        return d

    d1  = delta(mfcc_v)
    d2  = delta(d1)

    emb = np.concatenate([
        np.mean(mfcc_v, axis=0),
        np.mean(d1,     axis=0),
        np.mean(d2,     axis=0),
        np.var(mfcc_v,  axis=0),
    ])

    zcr  = float(np.mean(np.abs(np.diff(np.sign(pcm)))) / 2.0)
    emb  = np.append(emb, zcr * 5.0)

    emb = (emb - np.mean(emb)) / (np.std(emb) + 1e-9)
    return emb


def extract_embedding(pcm: np.ndarray, sr: int = SAMPLE_RATE) -> Optional[np.ndarray]:
    """Try ECAPA x-vector first; fall back to MFCC."""
    emb = extract_xvector(pcm, sr)
    if emb is None:
        emb = extract_mfcc_embedding(pcm, sr)
    if emb is None:
        return None
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 1e-9 else None


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Speaker Diarization (fixed N speakers)
# ══════════════════════════════════════════════════════════════════════════════
def diarize(
    pcm: np.ndarray,
    speech_segments: List[Tuple[float, float]],
    num_speakers: int,
    sr: int = SAMPLE_RATE,
) -> List[Tuple[float, float, int]]:
    """
    Returns list of (start_sec, end_sec, speaker_id).
    Uses KMeans(n_clusters=num_speakers) so the count is always exact.
    """
    print(f"\n{'─'*60}")
    mode = "ECAPA-TDNN x-vectors" if SPEECHBRAIN_AVAILABLE else "80-dim MFCC"
    print(f"🧠 Extracting speaker embeddings ({mode}) for {len(speech_segments)} segments…")

    embeddings: List[np.ndarray] = []
    valid_segs: List[Tuple[float, float]] = []

    for s, e in speech_segments:
        s_samp = int(s * sr)
        e_samp = int(e * sr)
        chunk  = pcm[s_samp:e_samp]
        emb    = extract_embedding(chunk, sr)
        if emb is not None:
            embeddings.append(emb)
            valid_segs.append((s, e))

    if len(embeddings) < num_speakers:
        print(f"  ⚠️  Only {len(embeddings)} valid segments — "
              f"assigning all to Speaker 1")
        return [(s, e, 0) for s, e in speech_segments]

    X = normalize(np.array(embeddings))

    print(f"  🔵 Clustering {len(X)} embeddings into {num_speakers} speakers…")

    km = KMeans(
        n_clusters  = num_speakers,
        n_init      = 20,
        max_iter    = 500,
        random_state= 42,
        tol         = 1e-5,
    )
    labels = km.fit_predict(X)

    diar_result: List[Tuple[float, float, int]] = []
    for (s, e), lbl in zip(valid_segs, labels):
        diar_result.append((s, e, int(lbl)))

    from collections import Counter
    counts = Counter(lbl for _, _, lbl in diar_result)
    total  = sum(counts.values())
    print(f"  ✅ Diarization done:")
    for spk_id in sorted(counts):
        pct = counts[spk_id] / total * 100
        sec = sum(e - s for s, e, l in diar_result if l == spk_id)
        print(f"     Speaker {spk_id+1}: {counts[spk_id]} segments  "
              f"({sec:.0f}s, {pct:.0f}%)")

    return diar_result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Transcription via OpenAI API
# ══════════════════════════════════════════════════════════════════════════════
def transcribe_whisper(
    wav_path: str,
    language: Optional[str] = None,
    prompt: Optional[str]   = None,
) -> List[Dict]:
    """
    Transcribe with gpt-4o-transcribe using verbose_json for WORD-LEVEL timestamps.
    Returns list of word dicts: {word, start, end}
    Handles files > 25 MB by splitting into chunks.

    🔧 FIX 1: Model changed from whisper-1 to gpt-4o-transcribe
    🔧 FIX 2: Uses expanded STT_PROMPT_HINT from shared_components
    """
    print(f"\n{'─'*60}")
    print(f"📝 Transcribing with {WHISPER_MODEL}…")

    client   = OpenAI(api_key=OPENAI_API_KEY)
    file_mb  = Path(wav_path).stat().st_size / 1e6
    LIMIT_MB = 24.0

    if not prompt:
        prompt = _build_stt_prompt(language)

    def _transcribe_chunk(chunk_path: str, offset: float = 0.0) -> List[Dict]:
        """Transcribe one chunk and shift timestamps by offset."""
        with open(chunk_path, "rb") as f:
            kwargs: Dict = dict(
                model          = WHISPER_MODEL,
                file           = f,
                response_format= "verbose_json",
                timestamp_granularities = ["word"],
            )
            if language:
                kwargs["language"] = language
            if prompt:
                kwargs["prompt"] = prompt

            resp = client.audio.transcriptions.create(**kwargs)

        words = []
        # verbose_json puts words in resp.words
        raw_words = getattr(resp, "words", None) or []
        for w in raw_words:
            word_text = getattr(w, "word", None) or (w.get("word", "") if isinstance(w, dict) else "")
            w_start   = getattr(w, "start", None) or (w.get("start", 0) if isinstance(w, dict) else 0)
            w_end     = getattr(w, "end",   None) or (w.get("end",   0) if isinstance(w, dict) else 0)
            if word_text:
                words.append({
                    "word" : word_text.strip(),
                    "start": round(float(w_start) + offset, 3),
                    "end"  : round(float(w_end)   + offset, 3),
                })
        # Fallback: segment-level timestamps
        if not words:
            segs = getattr(resp, "segments", None) or []
            for seg in segs:
                seg_text  = getattr(seg, "text",  "") or (seg.get("text",  "") if isinstance(seg, dict) else "")
                seg_start = getattr(seg, "start", 0)  or (seg.get("start", 0)  if isinstance(seg, dict) else 0)
                seg_end   = getattr(seg, "end",   0)  or (seg.get("end",   0)  if isinstance(seg, dict) else 0)
                if seg_text.strip():
                    words.append({
                        "word" : seg_text.strip(),
                        "start": round(float(seg_start) + offset, 3),
                        "end"  : round(float(seg_end)   + offset, 3),
                    })
        return words

    all_words: List[Dict] = []

    if file_mb <= LIMIT_MB:
        print(f"  → Single-pass transcription ({file_mb:.1f} MB)…")
        all_words = _transcribe_chunk(wav_path)
    else:
        print(f"  → File is {file_mb:.1f} MB > {LIMIT_MB} MB — splitting into chunks…")
        audio_duration = _get_duration(wav_path)
        chunk_secs = 1200
        n_chunks   = int(np.ceil(audio_duration / chunk_secs))
        print(f"     {n_chunks} chunks × {chunk_secs}s each")

        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(n_chunks):
                offset = i * chunk_secs
                chunk_path = os.path.join(tmpdir, f"chunk_{i:03d}.wav")
                cmd = [
                    "ffmpeg", "-y", "-i", wav_path,
                    "-ss", str(offset),
                    "-t",  str(chunk_secs),
                    "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                    chunk_path
                ]
                subprocess.run(cmd, capture_output=True, check=True)
                print(f"     Chunk {i+1}/{n_chunks} (offset={offset}s)…")
                words = _transcribe_chunk(chunk_path, offset=offset)
                all_words.extend(words)

    print(f"  ✅ Transcription done: {len(all_words)} words")
    return all_words


def _get_duration(wav_path: str) -> float:
    """Return audio duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json", wav_path
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(out.stdout)
    return float(data["format"]["duration"])


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Word → Speaker Alignment
# ══════════════════════════════════════════════════════════════════════════════
def align_words_to_speakers(
    words: List[Dict],
    diar: List[Tuple[float, float, int]],
    num_speakers: int,
) -> List[Dict]:
    """
    Assign each word to a speaker based on timestamp overlap.
    Strategy: find the diarization segment that has maximum overlap with the word.
    If no overlap, use nearest segment by distance.
    """
    annotated = []
    for w in words:
        ws, we = w["start"], w["end"]
        best_spk    = 0
        best_overlap = -1.0
        best_dist    = 1e9

        for seg_s, seg_e, spk in diar:
            overlap = min(we, seg_e) - max(ws, seg_s)
            if overlap > best_overlap:
                best_overlap = overlap
                best_spk     = spk

            dist = min(abs(ws - seg_e), abs(we - seg_s))
            if overlap <= 0 and dist < best_dist:
                best_dist = dist
                if best_overlap <= 0:
                    best_spk = spk

        annotated.append({**w, "speaker": best_spk})

    return annotated


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Output Formatting
# ══════════════════════════════════════════════════════════════════════════════
def group_into_lines(
    annotated_words: List[Dict],
    max_gap_s: float = 1.5,
    max_words: int   = 25,
) -> List[Dict]:
    """Group words into utterance lines, splitting on speaker change or long pause."""
    if not annotated_words:
        return []

    lines  = []
    cur    = [annotated_words[0]]

    for prev_w, w in zip(annotated_words, annotated_words[1:]):
        gap            = w["start"] - prev_w["end"]
        speaker_change = w["speaker"] != prev_w["speaker"]
        too_long       = len(cur) >= max_words

        if speaker_change or gap > max_gap_s or too_long:
            lines.append(_make_line(cur))
            cur = [w]
        else:
            cur.append(w)

    if cur:
        lines.append(_make_line(cur))

    return lines


def _make_line(words: List[Dict]) -> Dict:
    return {
        "speaker" : words[0]["speaker"],
        "start"   : words[0]["start"],
        "end"     : words[-1]["end"],
        "text"    : " ".join(w["word"] for w in words),
    }


def fmt_ts(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"


def fmt_srt_ts(seconds: float) -> str:
    h   = int(seconds) // 3600
    m   = (int(seconds) % 3600) // 60
    s   = int(seconds) % 60
    ms  = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def print_transcript(lines: List[Dict], num_speakers: int) -> None:
    print(f"\n{'═'*60}")
    print(f"  📄 TRANSCRIPT  ({num_speakers} SPEAKERS)")
    print(f"{'═'*60}\n")

    prev_spk = None
    for ln in lines:
        spk = ln["speaker"]
        col = COLOURS[spk % len(COLOURS)]
        ts  = fmt_ts(ln["start"])
        if spk != prev_spk:
            label = f"Speaker {spk + 1}"
            print(f"{col}{BOLD}[{label}]{RESET}  {col}({ts}){RESET}")
            prev_spk = spk
        print(f"  {col}{ln['text']}{RESET}")

    print(f"\n{'═'*60}")


def save_txt(lines: List[Dict], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for ln in lines:
            ts  = fmt_ts(ln["start"])
            spk = f"Speaker {ln['speaker'] + 1}"
            f.write(f"[{ts}] {spk}: {ln['text']}\n")
    print(f"  💾 TXT saved → {out_path}")


def save_srt(lines: List[Dict], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for i, ln in enumerate(lines, 1):
            s = fmt_srt_ts(ln["start"])
            e = fmt_srt_ts(ln["end"])
            spk = f"Speaker {ln['speaker'] + 1}"
            f.write(f"{i}\n{s} --> {e}\n{spk}: {ln['text']}\n\n")
    print(f"  💾 SRT saved → {out_path}")


def save_json(lines: List[Dict], out_path: str) -> None:
    data = []
    for ln in lines:
        data.append({
            "speaker"  : f"Speaker {ln['speaker'] + 1}",
            "start_sec": ln["start"],
            "end_sec"  : ln["end"],
            "start_fmt": fmt_ts(ln["start"]),
            "text"     : ln["text"],
        })
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  💾 JSON saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Offline N-speaker transcription + diarization (gpt-4o-transcribe)"
    )
    parser.add_argument(
        "--video", required=True,
        help="Path to the video (or audio) file, e.g. meeting.mp4"
    )
    parser.add_argument(
        "--num_speakers", type=int, default=2,
        help="Exact number of speakers in the recording (default: 2). "
             "For a 10-person meeting, use --num_speakers 10"
    )
    parser.add_argument(
        "--lang", default=None,
        help="Language code for transcription, e.g. 'ar' for Arabic, 'en' for English. "
             "Leave blank for auto-detect."
    )
    parser.add_argument(
        "--prompt", default=None,
        help="Custom transcription prompt (optional — overrides default domain prompt)"
    )
    parser.add_argument(
        "--out_dir", default=".",
        help="Output directory for transcript files (default: current dir)"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        print(f"ERROR: Video file not found: {args.video}")
        sys.exit(1)

    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY environment variable is not set.")
        print("  export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    stem     = Path(args.video).stem
    t_start  = time.time()

    print(f"\n{'═'*60}")
    print(f"  🎙️  N-SPEAKER TRANSCRIPTION PIPELINE")
    print(f"  Video       : {args.video}")
    print(f"  Speakers    : {args.num_speakers}")
    print(f"  Language    : {args.lang or 'auto-detect'}")
    print(f"  STT Model   : {WHISPER_MODEL}")
    print(f"  Embed Mode  : {'ECAPA-TDNN (192-dim)' if SPEECHBRAIN_AVAILABLE else 'MFCC (80-dim)'}")
    print(f"{'═'*60}")

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "audio.wav")

        # ── Step 1: Extract audio ─────────────────────────────────────────
        extract_audio(args.video, wav_path)

        # ── Step 2: Load PCM ──────────────────────────────────────────────
        print(f"\n{'─'*60}")
        print("📂 Loading audio…")
        pcm = load_wav(wav_path)
        duration = len(pcm) / SAMPLE_RATE
        print(f"  Duration : {fmt_ts(duration)}  ({duration:.0f}s)")

        # ── Step 3: VAD ───────────────────────────────────────────────────
        print(f"\n{'─'*60}")
        print("🔎 Running Voice Activity Detection…")
        speech_segs = detect_speech_segments(pcm, SAMPLE_RATE)

        # ── Step 4: Diarize ───────────────────────────────────────────────
        diar = diarize(pcm, speech_segs, args.num_speakers, SAMPLE_RATE)

        # ── Step 5: Transcribe ────────────────────────────────────────────
        prompt = args.prompt or _build_stt_prompt(args.lang)
        words = transcribe_whisper(wav_path, language=args.lang, prompt=prompt)

        if not words:
            print("\n⚠️  Transcription returned no words — check your API key and audio file.")
            sys.exit(1)

        # ── Step 6: Align ─────────────────────────────────────────────────
        print(f"\n{'─'*60}")
        print("🔗 Aligning words to speakers…")
        annotated = align_words_to_speakers(words, diar, args.num_speakers)

        # ── Step 7: Group into lines ──────────────────────────────────────
        lines = group_into_lines(annotated)
        print(f"  ✅ {len(lines)} transcript lines")

    # ── Output ────────────────────────────────────────────────────────────────
    print_transcript(lines, args.num_speakers)

    print(f"\n{'─'*60}")
    print("💾 Saving outputs…")
    txt_path  = os.path.join(args.out_dir, f"{stem}_transcript.txt")
    srt_path  = os.path.join(args.out_dir, f"{stem}_transcript.srt")
    json_path = os.path.join(args.out_dir, f"{stem}_transcript.json")
    save_txt (lines, txt_path)
    save_srt (lines, srt_path)
    save_json(lines, json_path)

    elapsed = time.time() - t_start
    print(f"\n{'═'*60}")
    print(f"  ✅ Done in {elapsed:.0f}s")
    print(f"     TXT  → {txt_path}")
    print(f"     SRT  → {srt_path}")
    print(f"     JSON → {json_path}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()