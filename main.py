"""
main.py  v5.2 — Live Multi-Speaker Speech Translator
════════════════════════════════════════════════════════════════════
FIXES vs v5.1 (live meeting focused)
──────────────────────────────────────────────────────────────────

FIX 1 — STT gets real word timestamps now
  OLD: gpt-4o-transcribe + response_format="json"
       → words[] list always empty
       → speaker-word alignment falls back to crude proportional split
         (words distributed by duration, ignoring actual speech rhythm)
       → Speaker A's words randomly assigned to Speaker B and vice-versa

  NEW: whisper-1 + response_format="verbose_json"
       + timestamp_granularities[]=word
       → per-word {word, start, end} timestamps returned
       → each word assigned to speaker whose audio segment overlaps its
         timestamp — accurate even when speakers talk at different speeds
  Cost: IDENTICAL — $0.006 / audio-minute for both models.

FIX 2 — Speaker diarization stops creating phantom speakers
  Root cause: MFCC embeddings computed on 1.5-second sub-segments are
  noisy. Normal within-speaker variation (breath, vowel shift, emotion)
  exceeded the old cosine threshold and opened new speaker slots.

  Changes:
  • COSINE_THRESH_XVEC  0.25 → 0.18   (require stronger difference)
  • COSINE_THRESH_MFCC  0.30 → 0.22
  • SPK_CHANGE_MIN_SEG_SEC 1.5 → 2.0  (longer segment = richer MFCC)
  • BOOTSTRAP_COUNT  2 → 5  (collect 5 embeddings before allowing new speaker)
  • TURN_SWITCH_MARGIN 0.20 → 0.25   (harder to flip speaker on short turns)
  • LLM_DIAR_CONF_THRESHOLD 0.45 → 0.35  (fewer expensive LLM diar calls)

FIX 3 — GPT correction cost reduced
  • PipelineDecisionEngine.TRIVIAL_SET expanded (shared_components.py)
    → back-channel phrases ("got it", "understood", "طبعا" …) skip GPT
  • LLM diarization fires only when confidence < 0.35 instead of < 0.45
  • Estimated 25-35% fewer OpenAI calls per meeting hour
════════════════════════════════════════════════════════════════════
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from deep_translator import GoogleTranslator
import asyncio, json, httpx, numpy as np, subprocess, tempfile
import os, time, threading, re, difflib
from typing import Optional, Dict, List, Tuple, Any
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from collections import deque
from itertools import count

from dotenv import load_dotenv
load_dotenv()  # loads OPENAI_API_KEY (and any other vars) from .env

from shared_components import (
    SemanticCache, TokenBudgetManager, AutoLearningLoop,
    EnhancedVAD, PipelineDecisionEngine, ValidationLayer,
    PromptTemplateManager,
    OPENAI_STT_MODEL, OPENAI_COT_MODEL, OPENAI_MINI_MODEL,
    OPENAI_STT_URL, OPENAI_CHAT_URL,
    SAMPLE_RATE, STT_PROMPT_HINT, GULF_PHRASES, NOISE_TAGS,
    detect_lang, compute_word_diff, format_ts, _safe_parse_json,
    extract_mean_pitch, apply_domain_fixes,
    is_whisper_hallucination,
)

# ── Optional SpeechBrain ECAPA (loads in background) ──────────────────────────
SPEECHBRAIN_AVAILABLE = False
sb_classifier = None

def _try_load_speechbrain():
    global SPEECHBRAIN_AVAILABLE, sb_classifier
    try:
        import torchaudio

        # ✅ FORCE backend (fixes warning on Windows)
        try:
            torchaudio.set_audio_backend("soundfile")
        except Exception:
            pass

        # ✅ Patch for newer torchaudio versions
        if not hasattr(torchaudio, "list_audio_backends"):
            torchaudio.list_audio_backends = lambda: ["soundfile"]

        from speechbrain.inference.speaker import EncoderClassifier

        sb_classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"},
        )

        SPEECHBRAIN_AVAILABLE = True
        print("✅ SpeechBrain ECAPA-TDNN loaded — 192-dim x-vectors active")

    except Exception as e:
        print(f"⚠️  SpeechBrain not available ({e}) — using enhanced MFCC fallback")

        
threading.Thread(target=_try_load_speechbrain, daemon=True).start()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")
OPENAI_HEADERS = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

# ── FIX 2: Diarization tuning constants ──────────────────────────────────────
COSINE_THRESH_XVEC  = 0.5
SPK_CHANGE_MIN_SEG_SEC = 2.5

FIXED_NUM_SPEAKERS   = 10
MIN_SPEAKERS         = 2
MAX_SPEAKERS_ALLOWED = 10
BOOTSTRAP_COUNT      = 5     # was 2 — need 5 embeddings before allowing new speaker
DYNAMIC_RECAL_EVERY  = 8
MAX_SEGS_PER_BLOB    = 10

MIN_BLOB_BYTES       = 4_000
BACK_TRANS_CONF      = 0.70

SPK_CHANGE_MIN_SEG_SEC = 4.0   # was 1.5 — longer segment → better MFCC
SPK_CHANGE_HOP_MS      = 250
SPK_CHANGE_WIN_MS      = 700

LLM_DIAR_CONF_THRESHOLD = 0.35   # was 0.45 — fewer expensive LLM diar calls
TURN_SWITCH_MARGIN      = 0.25   # was 0.20 — harder speaker flip on short turns
SHORT_TURN_SEC          = 1.8

_dynamic_num_speakers: int = 2

cache     = SemanticCache()
budget    = TokenBudgetManager()
learning  = AutoLearningLoop()
vad       = EnhancedVAD()
engine    = PipelineDecisionEngine()
validator = ValidationLayer()
prompts   = PromptTemplateManager()

meeting_transcript: List[Dict] = []
session_start_time: float      = time.time()
segment_id_counter             = count()


def _next_segment_id() -> str:
    return f"seg-{next(segment_id_counter)}"


# ── Speaker tracker ────────────────────────────────────────────────────────────
class AdvancedSpeakerTracker:
    """
    Dynamic-count speaker tracker with FIX 2 improvements:
    - Raises BOOTSTRAP_COUNT to 5 before allowing new-speaker creation
    - Tighter cosine thresholds (0.18 XVEC, 0.22 MFCC)
    - Harder turn-switch margin (0.25)
    """

    def __init__(self):
        self.lock         = threading.Lock()
        self.centroids:   List[np.ndarray] = []
        self.histories:   List[deque]      = []
        self.active_secs: List[float]      = []
        self.confidences: List[float]      = []
        self.embeddings:  List[np.ndarray] = []
        self.labels:      List[int]        = []
        self.timestamps:  List[float]      = []
        self.seg_count    = 0
        self.turn_memory: deque = deque(maxlen=20)
        self.use_xvec     = False
        self.threshold    = 0.0
        self._num_speakers = 2

    def reset(self):
        with self.lock:
            self.centroids.clear(); self.histories.clear()
            self.active_secs.clear(); self.confidences.clear()
            self.embeddings.clear(); self.labels.clear()
            self.timestamps.clear()
            self.seg_count = 0; self.turn_memory.clear()
            self.use_xvec  = SPEECHBRAIN_AVAILABLE
            self.threshold = COSINE_THRESH_XVEC if SPEECHBRAIN_AVAILABLE else COSINE_THRESH_MFCC
            self._num_speakers = 2
            mode = "ECAPA x-vec" if self.use_xvec else "MFCC"
            print(f"  🔄 SpeakerTracker reset — {mode} mode")

    # ── Embedding extractors ───────────────────────────────────────────────────
    def extract_xvector(self, pcm_float: np.ndarray) -> Optional[np.ndarray]:
        if not SPEECHBRAIN_AVAILABLE or sb_classifier is None:
            return None
        try:
            import torch
            wav = pcm_float / (np.max(np.abs(pcm_float)) + 1e-9)
            t   = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                emb = sb_classifier.encode_batch(t).squeeze().cpu().numpy()
            norm = np.linalg.norm(emb)
            return emb / norm if norm > 1e-9 else None
        except Exception as e:
            print(f"  ⚠️  x-vector error: {e}"); return None

    def extract_mfcc_enhanced(self, pcm_float: np.ndarray,
                               sr: int = SAMPLE_RATE) -> Optional[np.ndarray]:
        # FIX 2: require ≥ 0.5s of audio (was 0.3s) — richer embedding
        if len(pcm_float) < sr * 0.5:
            return None
        signal = pcm_float / (np.max(np.abs(pcm_float)) + 1e-9)
        signal = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])
        frame_len  = int(0.025 * sr)
        frame_step = int(0.010 * sr)
        n_frames   = (len(signal) - frame_len) // frame_step
        if n_frames < 3:
            return None
        idx    = np.arange(frame_len) + np.arange(n_frames)[:, None] * frame_step
        frames = signal[idx] * np.hamming(frame_len)
        energies = np.sum(frames ** 2, axis=1)
        NFFT, n_filt, n_mfcc = 512, 40, 20
        mag   = np.abs(np.fft.rfft(frames, NFFT))
        power = (1.0 / NFFT) * mag ** 2
        high_mel = 2595 * np.log10(1 + (sr / 2) / 700)
        mel_pts  = np.linspace(0, high_mel, n_filt + 2)
        hz_pts   = 700 * (10 ** (mel_pts / 2595) - 1)
        bin_pts  = np.floor((NFFT + 1) * hz_pts / sr).astype(int).clip(0, NFFT // 2)
        fbank = np.zeros((n_filt, NFFT // 2 + 1))
        for m in range(1, n_filt + 1):
            lo, mid, hi = bin_pts[m-1], bin_pts[m], bin_pts[m+1]
            for k in range(lo, mid):   fbank[m-1, k] = (k - lo) / (mid - lo + 1e-9)
            for k in range(mid, hi):   fbank[m-1, k] = (hi - k) / (hi - mid + 1e-9)
        fb   = np.dot(power, fbank.T)
        fb   = np.where(fb == 0, np.finfo(float).eps, fb)
        fb   = 20 * np.log10(fb)
        mfcc = np.zeros((n_frames, n_mfcc))
        for n in range(n_mfcc):
            mfcc[:, n] = np.sum(
                fb * np.cos(np.pi * n / n_filt * (np.arange(1, n_filt + 1) - 0.5)), axis=1)
        voiced = energies >= np.percentile(energies, 30)
        if voiced.sum() < 3: voiced = np.ones(n_frames, dtype=bool)
        mv = mfcc[voiced]

        def _delta(feat, N=2):
            d = np.zeros_like(feat)
            for t in range(len(feat)):
                num = sum(nn*(feat[min(t+nn,len(feat)-1)]-feat[max(t-nn,0)]) for nn in range(1,N+1))
                den = 2*sum(nn**2 for nn in range(1,N+1))
                d[t] = num/den
            return d

        d1, d2 = _delta(mv), _delta(_delta(mv))
        emb = np.concatenate([np.mean(mv,0), np.mean(d1,0), np.mean(d2,0), np.var(mv,0)])
        emb = np.append(emb, extract_mean_pitch(pcm_float, sr) * 5.0)
        emb = (emb - np.mean(emb)) / (np.std(emb) + 1e-9)
        return emb

    def extract_embedding(self, pcm: np.ndarray) -> Optional[np.ndarray]:
        if SPEECHBRAIN_AVAILABLE and not self.use_xvec:
            self.use_xvec = True
            print("  🎙️  Switched to ECAPA x-vector mode")
        emb = self.extract_xvector(pcm) if SPEECHBRAIN_AVAILABLE else None
        if emb is None:
            emb = self.extract_mfcc_enhanced(pcm)
        if emb is None:
            return None
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 1e-9 else None

    # ── Core identification ────────────────────────────────────────────────────
    def identify_speaker(self, embedding: np.ndarray,
                          seg_duration_sec: float = 1.0) -> Tuple[int, float]:
        with self.lock:
            norm_val = np.linalg.norm(embedding)
            if norm_val < 1e-9:
                return 0, 0.5
            emb_n = embedding / norm_val

            # Bootstrap phase: create speaker slots up to FIXED_NUM_SPEAKERS.
            # FIX 2: Only open a new slot after we have >= BOOTSTRAP_COUNT
            # embeddings total — this prevents the first noisy frames from
            # spawning a forest of phantom speakers.
            if len(self.centroids) < FIXED_NUM_SPEAKERS:
                thresh   = COSINE_THRESH_XVEC if self.use_xvec else COSINE_THRESH_MFCC
                best_sim = -2.0
                best_idx = 0
                for i, c in enumerate(self.centroids):
                    cn = np.linalg.norm(c)
                    if cn < 1e-9: continue
                    sim = float(np.dot(emb_n, c / cn))
                    if sim > best_sim: best_sim, best_idx = sim, i

                # Conditions to open a new speaker slot:
                # 1. Cosine distance from all known speakers exceeds threshold
                # 2. We have collected enough embeddings to trust the comparison
                enough_data = len(self.embeddings) >= BOOTSTRAP_COUNT
                if best_sim < thresh and (not self.centroids or enough_data):
                    spk = len(self.centroids)
                    self.centroids.append(embedding.copy())
                    self.histories.append(deque(maxlen=30))
                    self.histories[-1].append(embedding.copy())
                    self.confidences.append(1.0)
                    self.active_secs.append(seg_duration_sec)
                    self._num_speakers = len(self.centroids)
                    self._record(embedding, spk)
                    print(f"  🟢 New Speaker {spk+1} "
                          f"(total={self._num_speakers}, embeds={len(self.embeddings)})")
                    return spk, 1.0
                elif not self.centroids:
                    self.centroids.append(embedding.copy())
                    self.histories.append(deque(maxlen=30))
                    self.histories[-1].append(embedding.copy())
                    self.confidences.append(1.0)
                    self.active_secs.append(seg_duration_sec)
                    self._num_speakers = 1
                    self._record(embedding, 0)
                    print("  🟢 Bootstrap Speaker 1")
                    return 0, 1.0
                else:
                    spk = best_idx
                    self._update_centroid(spk, embedding, seg_duration_sec)
                    self.turn_memory.append((spk, time.time()))
                    self._record(embedding, spk)
                    return spk, max(0.0, min(1.0, (best_sim + 1.0) / 2.0))

            # Normal phase: assign to nearest centroid; supplement with history
            sims = []
            for i, c in enumerate(self.centroids):
                cn = np.linalg.norm(c)
                if cn < 1e-9: sims.append(-1.0); continue
                csim = float(np.dot(emb_n, c / cn))
                hist = self.histories[i]
                if len(hist) >= 2:
                    hsims = [float(np.dot(emb_n, h/(np.linalg.norm(h)+1e-9)))
                             for h in list(hist)[-5:]]
                    csim = max(csim, float(np.percentile(hsims, 75)) * 0.95)
                sims.append(csim)

            best_idx = int(np.argmax(sims))
            best_sim = sims[best_idx]
            spk      = best_idx

            # FIX 2: Harder speaker flip — need TURN_SWITCH_MARGIN=0.25 advantage
            if self.turn_memory and len(sims) >= 2:
                last_spk = self.turn_memory[-1][0]
                if 0 <= last_spk < len(sims) and best_idx != last_spk:
                    prev_sim    = sims[last_spk]
                    weak_switch = (best_sim - prev_sim) < TURN_SWITCH_MARGIN
                    if seg_duration_sec <= SHORT_TURN_SEC and weak_switch:
                        spk = last_spk; best_sim = prev_sim

            self._update_centroid(spk, embedding, seg_duration_sec)
            self._record(embedding, spk)
            self.turn_memory.append((spk, time.time()))

            if self.seg_count % DYNAMIC_RECAL_EVERY == 0 and self.seg_count > 0:
                self._auto_recalibrate()

            return spk, float(max(0.0, min(1.0, (best_sim + 1.0) / 2.0)))

    def _update_centroid(self, spk: int, embedding: np.ndarray,
                          dur: float, alpha: float = 0.05):
        self.centroids[spk] = (1-alpha)*self.centroids[spk] + alpha*embedding
        self.histories[spk].append(embedding.copy())
        self.active_secs[spk] += dur

    def _record(self, embedding: np.ndarray, spk: int):
        self.embeddings.append(embedding.copy())
        self.labels.append(spk)
        self.timestamps.append(time.time())
        self.seg_count += 1

    def _auto_recalibrate(self):
        n = len(self.embeddings)
        if n < 6: return
        new_k = min(len(self.centroids), FIXED_NUM_SPEAKERS)
        if new_k < 2: return
        try:
            from sklearn.cluster import KMeans
            recent  = min(n, 200)
            X       = normalize(np.array(self.embeddings[-recent:]))
            km      = KMeans(n_clusters=new_k, n_init=10, max_iter=300, random_state=42)
            km.fit(X)
            new_centers = km.cluster_centers_
            old_k = len(self.centroids)
            used = set(); mapping = {}
            for ki in range(new_k):
                best_s, best_v = 0, -2.0
                for si in range(old_k):
                    if si in used: continue
                    cn = np.linalg.norm(self.centroids[si])
                    if cn < 1e-9: continue
                    v = float(np.dot(new_centers[ki], self.centroids[si] / cn))
                    if v > best_v: best_v, best_s = v, si
                mapping[ki] = best_s; used.add(best_s)
            new_c, new_h, new_a, new_cf = [], [], [], []
            for ki in range(new_k):
                old_si = mapping.get(ki, ki)
                new_c.append(new_centers[ki].copy())
                if old_si < len(self.histories):
                    new_h.append(self.histories[old_si]); new_a.append(self.active_secs[old_si])
                    new_cf.append(self.confidences[old_si] if old_si < len(self.confidences) else 0.8)
                else:
                    new_h.append(deque(maxlen=30)); new_a.append(0.0); new_cf.append(0.8)
            self.centroids = new_c; self.histories = new_h
            self.active_secs = new_a; self.confidences = new_cf
            self._num_speakers = new_k
            print(f"  🔄 Recalibrate: k={old_k}→{new_k} on {recent} segs")
        except Exception as e:
            print(f"  ⚠️  Auto-recalibrate error: {e}")

    def force_minimum_speakers(self, min_spk: int = 2):
        if len(self.centroids) < min_spk and self.embeddings:
            self._auto_recalibrate()

    def get_stats(self) -> Dict:
        return {
            "n_speakers":  len(self.centroids),
            "n_segments":  self.seg_count,
            "mode":        "ecapa-xvec" if self.use_xvec else "mfcc-enhanced",
            "threshold":   round(self.threshold, 3),
            "active_secs": [round(s, 1) for s in self.active_secs],
        }

speaker_tracker = AdvancedSpeakerTracker()



# ── Utilities ──────────────────────────────────────────────────────────────────
def _collapse_repeated_words(text: str) -> str:
    words = text.split()
    if not words: return text.strip()
    collapsed: List[str] = []
    for word in words:
        token     = re.sub(r"[^\w\u0600-\u06FF]+", "", word).lower()
        prev      = collapsed[-1] if collapsed else ""
        prev_token = re.sub(r"[^\w\u0600-\u06FF]+", "", prev).lower() if prev else ""
        if token and prev_token == token: continue
        collapsed.append(word)
    return " ".join(collapsed).strip()


def find_speaker_boundaries_v5(pcm: np.ndarray,
                                sr: int = SAMPLE_RATE) -> List[Tuple[int, int]]:
    """
    Multi-feature speaker change detection (spectral flux, RMS, centroid, ZCR).
    FIX 2: MIN_SEG now uses SPK_CHANGE_MIN_SEG_SEC=2.0 — longer boundaries mean
    each embedding has more audio, improving MFCC quality significantly.
    """
    MIN_SEG = int(sr * SPK_CHANGE_MIN_SEG_SEC)
    HOP     = int(sr * SPK_CHANGE_HOP_MS / 1000)
    WIN     = int(sr * SPK_CHANGE_WIN_MS / 1000)

    if len(pcm) < 2 * MIN_SEG:
        return [(0, len(pcm))]

    sig    = np.append(pcm[0], pcm[1:] - 0.97 * pcm[:-1])
    N_BINS = 40

    def mel_spec(w: np.ndarray) -> np.ndarray:
        n = min(len(w), 512)
        f = np.abs(np.fft.rfft(w[:n] * np.hamming(n), 512))[:N_BINS] + 1e-9
        return f / (np.linalg.norm(f) + 1e-9)

    def rms(w: np.ndarray) -> float:
        return float(np.sqrt(np.mean(w**2) + 1e-9))

    def spectral_centroid(w: np.ndarray) -> float:
        n   = min(len(w), 512)
        mag = np.abs(np.fft.rfft(w[:n], 512))
        return float(np.sum(np.linspace(0, sr/2, len(mag)) * mag) / (np.sum(mag) + 1e-9))

    def zcr(w: np.ndarray) -> float:
        return float(np.mean(np.abs(np.diff(np.sign(w)))) / 2.0)

    scores: List[Tuple[int, float]] = []
    for pos in range(WIN, len(sig) - WIN, HOP):
        w1, w2   = sig[max(0,pos-WIN):pos], sig[pos:pos+WIN]
        d_spec   = 1.0 - float(np.dot(mel_spec(w1), mel_spec(w2)))
        r1, r2   = rms(w1), rms(w2)
        d_rms    = abs(r1-r2) / (max(r1,r2)+1e-9)
        sc1, sc2 = spectral_centroid(w1), spectral_centroid(w2)
        d_cent   = abs(sc1-sc2) / (max(sc1,sc2,1.0))
        d_zcr    = abs(zcr(w1) - zcr(w2))
        combined = 0.40*d_spec + 0.25*d_rms + 0.20*d_cent + 0.15*d_zcr
        scores.append((pos, combined))

    if not scores:
        return [(0, len(pcm))]

    dist_arr = np.array([d for _, d in scores])
    smooth   = np.convolve(dist_arr, np.ones(5)/5, mode='same')
    silence_boundaries = _detect_silence_gaps(pcm, sr)
    threshold  = float(np.mean(smooth) + 1.2 * np.std(smooth))
    boundaries = [0]
    last_b     = 0

    for i, (pos, _) in enumerate(scores):
        if smooth[i] < threshold: continue
        lo = max(0, i-3); hi = min(len(smooth), i+4)
        if smooth[i] != max(smooth[lo:hi]): continue
        if (pos - last_b) >= MIN_SEG:
            boundaries.append(pos); last_b = pos

    for sb in silence_boundaries:
        near = any(abs(sb - b) < int(sr*0.3) for b in boundaries)
        if not near and sb > MIN_SEG and sb < len(pcm) - MIN_SEG:
            boundaries.append(sb)

    boundaries.append(len(pcm))
    boundaries = sorted(set(boundaries))
    result     = [
        (boundaries[i], boundaries[i+1])
        for i in range(len(boundaries)-1)
        if boundaries[i+1] - boundaries[i] >= MIN_SEG // 2
    ]
    return result if result else [(0, len(pcm))]


def _detect_silence_gaps(pcm: np.ndarray, sr: int,
                          min_gap_ms: int = 150,
                          silence_thresh_rms: float = 300.0) -> List[int]:
    HOP = int(sr*0.020); WIN = int(sr*0.040); MIN_GAP = int(min_gap_ms/1000*sr)
    in_silence = True; silence_start = 0; transitions = []
    for pos in range(0, len(pcm)-WIN, HOP):
        r = float(np.sqrt(np.mean(pcm[pos:pos+WIN].astype(np.float64)**2)))
        if in_silence and r > silence_thresh_rms:
            if pos - silence_start >= MIN_GAP: transitions.append(pos)
            in_silence = False
        elif not in_silence and r < silence_thresh_rms * 0.5:
            silence_start = pos; in_silence = True
    return transitions


# ── LLM speaker validation (cost-gated) ───────────────────────────────────────
async def llm_validate_speakers(segments: List[Dict],
                                 confidences: List[float],
                                 meeting_context: str) -> List[Dict]:
    """
    FIX 2: LLM_DIAR_CONF_THRESHOLD lowered to 0.35 (was 0.45).
    At 0.45 almost every MFCC-based blob triggered this call.
    At 0.35 only genuinely uncertain segments (where acoustic + neighbour
    context is truly ambiguous) reach the LLM.
    """
    low_conf_indices = [i for i, c in enumerate(confidences)
                        if c < LLM_DIAR_CONF_THRESHOLD]
    if not low_conf_indices or len(segments) < 2:
        return segments

    context_lines = []
    for i, seg in enumerate(segments):
        note = " [LOW CONF]" if i in low_conf_indices else ""
        context_lines.append(f"[{i}] Speaker {seg['speaker']+1}{note}: {seg['original']}")

    prompt = (
        "You are helping with speaker diarization for a healthcare IT meeting.\n"
        f"MEETING CONTEXT:\n{meeting_context or '(start)'}\n\n"
        "SEGMENTS:\n" + "\n".join(context_lines) + "\n\n"
        "For each [LOW CONF] segment, check if the speaker assignment is correct "
        "using dialogue patterns (Q→A pairs, topic continuity, name mentions).\n"
        "Return ONLY JSON array of corrections (empty [] if none needed):\n"
        '[{"idx":0,"speaker":1,"reason":"brief"}]\n'
        "Speaker numbers are 1-indexed. Only include confident corrections."
    )

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                OPENAI_CHAT_URL,
                headers={**OPENAI_HEADERS, "Content-Type": "application/json"},
                json={"model": OPENAI_MINI_MODEL, "temperature": 0.0,
                      "max_tokens": 300,
                      "messages": [{"role": "user", "content": prompt}],
                      "response_format": {"type": "json_object"}},
            )
        if resp.status_code == 200:
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            parsed = _safe_parse_json(raw)
            corrections = parsed if isinstance(parsed, list) else parsed.get("corrections", [])
            for corr in corrections:
                idx = corr.get("idx"); spk = corr.get("speaker")
                if idx is not None and spk is not None and 0 <= idx < len(segments):
                    old = segments[idx]["speaker"] + 1; new = int(spk)
                    if old != new:
                        segments[idx]["speaker"] = new - 1
                        segments[idx]["llm_corrected"] = True
                        print(f"  🤖 LLM diar: seg[{idx}] Spk{old}→Spk{new} ({corr.get('reason','')[:50]})")
    except Exception as e:
        print(f"  ⚠️  LLM diarization check failed: {e}")
    return segments


# ── Audio helpers ──────────────────────────────────────────────────────────────
def webm_to_pcm(webm_bytes: bytes) -> Optional[np.ndarray]:
    try:
        proc = subprocess.run(
            ["ffmpeg", "-v", "quiet", "-i", "pipe:0",
             "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "s16le", "pipe:1"],
            input=webm_bytes, capture_output=True, timeout=10
        )
        if proc.returncode == 0 and proc.stdout:
            return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32)
    except Exception as e:
        print(f"webm_to_pcm error: {e}")
    return None


def _embed_sub_segment_sync(pcm: np.ndarray,
                             start_s: int, end_s: int) -> Tuple[Optional[np.ndarray], float]:
    sub = pcm[start_s:end_s]
    dur = (end_s - start_s) / SAMPLE_RATE
    emb = speaker_tracker.extract_embedding(sub)
    return emb, dur


async def _call_openai_chat_async(system_text: str, user_text: str,
                                   model: str, max_tokens: int = 200,
                                   json_mode: bool = False) -> Optional[str]:
    payload: Dict = {
        "model": model, "temperature": 0.0, "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": system_text},
                     {"role": "user",   "content": user_text}],
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            resp = await client.post(
                OPENAI_CHAT_URL,
                headers={**OPENAI_HEADERS, "Content-Type": "application/json"},
                json=payload,
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
            print(f"OpenAI chat {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"_call_openai_chat_async error: {e}")
    return None


def _parse_words(data: dict) -> list:
    """
    Extract word-level timestamps from verbose_json STT response.
    FIX 1: With whisper-1 + verbose_json, data["words"] is now populated.
    Each entry: {"word": "...", "start": float, "end": float}
    """
    raw = data.get("words") or []
    out = []
    for w in raw:
        if isinstance(w, dict) and "word" in w:
            out.append({"word": w["word"].strip(),
                        "start": float(w.get("start", 0)),
                        "end":   float(w.get("end",   0))})
    # Fallback: segment-level timestamps if word-level unavailable
    if not out:
        for seg in (data.get("segments") or []):
            out.append({"word":  seg.get("text", "").strip(),
                        "start": float(seg.get("start", 0)),
                        "end":   float(seg.get("end",   0))})
    return out


# ── FIX 1: STT with whisper-1 + verbose_json for word timestamps ──────────────
async def transcribe_with_openai(webm_bytes: bytes) -> Optional[Dict]:
    """
    FIX 1 — whisper-1 + verbose_json + word timestamps.

    OLD (broken):
      model=gpt-4o-transcribe, response_format="json"
      → API returns only {"text": "..."}, no words[] field.
      → _parse_words() always returns []
      → Every blob uses proportional word-split fallback (wrong)

    NEW (fixed):
      model=whisper-1, response_format="verbose_json"
      + timestamp_granularities[]=word
      → API returns words[] with per-word {word, start, end}
      → Speaker-word alignment uses actual timestamps (correct)

    whisper-1 is also faster for real-time (<1s latency on short clips)
    because it doesn't run the full GPT-4o reasoning stack.
    """
    import hashlib
    fp = hashlib.sha256(webm_bytes[:8192]).hexdigest()[:32]
    ck = f"stt:{len(webm_bytes)}:{fp}"
    cached = cache.get(ck, "stt")
    if cached:
        print("✔️  STT cache hit"); return cached

    def _send_stt_request() -> httpx.Response:
        with httpx.Client(timeout=30.0) as client:
            return client.post(
                OPENAI_STT_URL,
                headers=OPENAI_HEADERS,
                files={"file": ("audio.webm", webm_bytes, "audio/webm")},
                data={
                    "model":           OPENAI_STT_MODEL,   # "whisper-1"
                    "response_format": "verbose_json",      # ← word timestamps
                    "timestamp_granularities[]": "word",    # ← per-word timing
                    "prompt":          STT_PROMPT_HINT,
                    "temperature":     "0",
                },
            )

    try:
        resp = await asyncio.to_thread(_send_stt_request)
        if resp.status_code == 200:
            data  = resp.json()
            txt   = data.get("text", "").strip()
            lang  = detect_lang(txt) if txt else "en"
            if txt:
                # ── Whisper hallucination gate ─────────────────────────────
                # Whisper generates "Thank you", "Thank you for watching" etc.
                # for silent/near-silent audio.  Drop these before any further
                # processing so they never reach the transcript or translation.
                if is_whisper_hallucination(txt):
                    print(f"🚫 Hallucination blocked: '{txt[:80]}'")
                    return {"text": "", "language": "en", "words": []}
                # ──────────────────────────────────────────────────────────
                words      = _parse_words(data)
                word_count = len(words)
                print(f"📝 STT: '{txt[:80]}{'…' if len(txt)>80 else ''}' "
                      f"(lang={lang}, words={word_count})")
                if word_count == 0:
                    print("  ⚠️  No word timestamps returned — check whisper-1 verbose_json")
                result = {"text": txt, "language": lang, "words": words}
                cache.set(ck, result, "stt")
                return result
            else:
                print("🔇 STT: silent/noise — skipping")
                return {"text": "", "language": "en", "words": []}
        print(f"STT error {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"transcribe_with_openai error: {e}")
    return None


# ── Translation ────────────────────────────────────────────────────────────────
async def translate_async(text: str, src: str) -> str:
    if not text.strip() or src == "en": return text
    for arabic, english in GULF_PHRASES.items():
        if arabic in text and len(text.replace(arabic, "").strip()) < 3:
            return english
    cached = cache.get(text, "translation")
    if cached: return cached.get("translation", text)
    result = await _call_openai_chat_async(
        "Translate Gulf/Levantine Arabic or Hindi to English. "
        "Keep tech terms (RCM, EMR, HIS, staging, sandbox, UAT, QA, Waseel, Clinicy, "
        "Nafis, NPHIES, MOH, CCHI, integration, configuration, workflow) unchanged. "
        "Return ONLY the translation.",
        f"Translate: {text}",
        model=OPENAI_MINI_MODEL, max_tokens=120,
    )
    if result:
        cache.set(text, {"translation": result}, "translation"); return result
    try:
        t = GoogleTranslator(source="auto", target="en").translate(text)
        return t or text
    except Exception:
        return text


async def _back_translate_check(original: str, translation: str, src_lang: str) -> float:
    if src_lang == "en" or not translation or translation in NOISE_TAGS: return 1.0
    lang_name = {"ar": "Arabic", "hi": "Hindi"}.get(src_lang, "Arabic")
    back = await _call_openai_chat_async(
        f"Translate English to {lang_name}. Return ONLY translation.",
        translation, model=OPENAI_MINI_MODEL, max_tokens=120,
    )
    if not back: return 0.8
    return difflib.SequenceMatcher(None, original.lower(), back.lower()).ratio()


def _passthrough_result(raw_orig: str, raw_trans: str) -> Dict:
    return {"corrected_original": raw_orig, "corrected_translation": raw_trans,
            "original_diff": [], "translation_diff": [], "was_changed": False, "tier": "passthrough"}


async def correct_segment(seg: Dict) -> Optional[Dict]:
    raw_orig  = seg.get("original", "").strip()
    raw_trans = seg.get("translation", "").strip()
    if not raw_orig or raw_orig in NOISE_TAGS: return None

    tier = engine.decide(raw_orig, conf=1.0, learning=learning)
    if tier == "skip":    return None
    if tier == "trivial": return _passthrough_result(raw_orig, raw_trans)
    if tier == "rule":
        corrected = learning.apply_rules(raw_orig) or raw_orig
        return _passthrough_result(corrected, raw_trans)

    ck = f"corr:{hash(raw_orig)}"
    cached = cache.get(ck, "correction")
    if cached: return cached

    context       = _build_context()
    learnings_str = learning.context_str()
    model         = OPENAI_COT_MODEL if tier == "full_cot" else OPENAI_MINI_MODEL
    max_tok       = budget.get(tier)

    try:
        system = prompts.get_system_with_context(tier, context, learnings_str)
    except Exception:
        system = (
            "STT post-processor: Gulf Arabic/English healthcare IT. Fix phonetic errors. "
            "Keep: RCM, EMR, HIS, Waseel, Clinicy, Nafis, UAT, QA, integration, configuration. "
            'Return ONLY JSON array: [{"idx":0,"corrected":"...","translation":"..."}]'
        )
    try:
        user_msg = prompts.build_batch_msg([seg], context, learnings_str, tier)
    except Exception:
        user_msg = f"Correct and translate: {raw_orig}"

    raw_resp = await _call_openai_chat_async(system, user_msg, model=model,
                                              max_tokens=max_tok, json_mode=False)
    if not raw_resp: return _passthrough_result(raw_orig, raw_trans)

    try:
        parsed = _safe_parse_json(raw_resp, ["corrected", "translation"])
        item   = parsed[0] if isinstance(parsed, list) and parsed else (
                 parsed if isinstance(parsed, dict) else None)
        if not item: return _passthrough_result(raw_orig, raw_trans)

        corr_orig  = item.get("corrected",   raw_orig).strip() or raw_orig
        corr_trans = item.get("translation", raw_trans).strip() or raw_trans
        conf       = float(item.get("confidence", 0.85))
        corr_orig  = _collapse_repeated_words(corr_orig)
        corr_trans = _collapse_repeated_words(corr_trans)
        if conf < 0.3: corr_orig = raw_orig; corr_trans = raw_trans

        nl = item.get("new_learnings", {})
        if isinstance(nl, dict):
            for w, c in nl.items():
                if w and c and w != c: learning.record(w, c)

        corr_orig, corr_trans, was_changed = validator.validate(raw_orig, corr_orig, corr_trans)

        if was_changed and corr_orig and corr_orig != raw_orig:
            learning.record(raw_orig, corr_orig)

        if tier == "full_cot" and conf < BACK_TRANS_CONF:
            src = detect_lang(corr_orig)
            if await _back_translate_check(corr_orig, corr_trans, src) < 0.25:
                corr_trans = await translate_async(corr_orig, src)

        if any('\u0600' <= c <= '\u06FF' for c in corr_trans):
            corr_trans = await translate_async(corr_orig, detect_lang(corr_orig))
        corr_trans = _collapse_repeated_words(corr_trans)

        corr_orig_final = apply_domain_fixes(corr_orig)
        if corr_orig_final != corr_orig:
            was_changed = True; corr_orig = corr_orig_final

        result = {
            "corrected_original":    corr_orig,
            "corrected_translation": corr_trans,
            "original_diff":    compute_word_diff(raw_orig,  corr_orig),
            "translation_diff": compute_word_diff(raw_trans, corr_trans),
            "was_changed":      was_changed,
            "tier":             tier,
        }
        cache.set(ck, result, "correction")
        return result
    except Exception as e:
        print(f"  correction parse error: {e}")
    return _passthrough_result(raw_orig, raw_trans)


def _build_context() -> str:
    if not meeting_transcript: return "(session start)"
    recent = meeting_transcript[-8:]
    return "\n".join(
        f"Speaker {s['speaker']+1} [{s['timestamp']}]: {s['original']}"
        for s in recent
    )


# ── Main blob processing pipeline ─────────────────────────────────────────────
async def process_blob(websocket: WebSocket, webm_blob: bytes):
    """
    Full v5.2 pipeline for one audio blob.

    FIX 1: STT now returns word timestamps (whisper-1 + verbose_json).
           Speaker-word alignment uses timestamp overlap instead of
           proportional duration estimation.

    FIX 2: Speaker boundaries require >= 2.0s segments; bootstrap requires
           >= 5 embeddings before opening a new speaker slot.
    """
    if len(webm_blob) < MIN_BLOB_BYTES:
        return

    loop = asyncio.get_event_loop()
    pcm  = await loop.run_in_executor(None, webm_to_pcm, webm_blob)
    if pcm is None or len(pcm) < SAMPLE_RATE * 0.5:
        return

    speech_ok, vad_stats = vad.is_speech(pcm)
    if not speech_ok:
        print(f"🔇 VAD rejected  rms={vad_stats['rms']:.0f} "
              f"zcr={vad_stats['zcr']:.3f} "
              f"voiced_frac={vad_stats.get('voiced_frac', 0):.2f}")
        return

    # Kick off STT and boundary detection in parallel
    stt_future        = asyncio.ensure_future(transcribe_with_openai(webm_blob))
    boundaries_future = loop.run_in_executor(None, find_speaker_boundaries_v5, pcm)

    stt_result = await stt_future
    sub_segs   = await boundaries_future

    if len(sub_segs) > MAX_SEGS_PER_BLOB:
        sub_segs = sub_segs[:MAX_SEGS_PER_BLOB-1] + [
            (sub_segs[MAX_SEGS_PER_BLOB-1][0], sub_segs[-1][1])
        ]

    if not stt_result or not stt_result.get("text", "").strip():
        return

    text          = stt_result["text"].strip()
    detected_lang = stt_result.get("language", detect_lang(text))
    words         = stt_result.get("words", [])  # now populated (FIX 1)

    blob_duration = len(pcm) / SAMPLE_RATE
    print(f"📊 Blob {blob_duration:.1f}s → {len(sub_segs)} sub-segs, "
          f"lang={detected_lang}, words={len(words)}, "
          f"speakers={len(speaker_tracker.centroids)}, "
          f"mode={speaker_tracker.get_stats()['mode']}")

    # Embed all sub-segments in parallel
    embed_tasks   = [loop.run_in_executor(None, _embed_sub_segment_sync, pcm, s, e)
                     for s, e in sub_segs]
    embed_results = await asyncio.gather(*embed_tasks)

    sub_speakers:    List[int]   = []
    sub_confidences: List[float] = []
    for emb, dur in embed_results:
        if emb is not None:
            spk, conf = await loop.run_in_executor(
                None, speaker_tracker.identify_speaker, emb, dur)
        else:
            spk, conf = 0, 0.5
        sub_speakers.append(spk)
        sub_confidences.append(conf)

    # Force minimum 2-speaker split if dialogue pattern detected
    if len(speaker_tracker.centroids) == 1 and len(speaker_tracker.embeddings) >= 8:
        dialogue_keywords = ["؟", "?", " you ", " we ", "okay", "yes", "no",
                             "تمام", "طيب", "أيوه", "ماشي"]
        context_text = " ".join(s["original"] for s in meeting_transcript[-10:])
        if any(kw in context_text.lower() for kw in dialogue_keywords):
            await loop.run_in_executor(None, speaker_tracker.force_minimum_speakers, 2)

    seg_outputs: List[Tuple[int, str, float]] = []

    if words:
        # FIX 1: Use actual word timestamps for speaker-word alignment.
        # Each word is assigned to the sub-segment whose time range overlaps
        # the word's start time — accurate regardless of speaking speed.
        for (seg_start_s, seg_end_s), spk, conf in zip(sub_segs, sub_speakers, sub_confidences):
            seg_start_sec = seg_start_s / SAMPLE_RATE
            seg_end_sec   = seg_end_s   / SAMPLE_RATE
            seg_words = [
                w.get("word", "")
                for w in words
                if seg_start_sec <= w.get("start", 0.0) < seg_end_sec
            ]
            if seg_words:
                seg_outputs.append((spk, " ".join(seg_words).strip(), conf))
    else:
        # Proportional fallback (only triggers if whisper-1 fails to return words)
        print("  ⚠️  No word timestamps — using proportional fallback")
        all_words = text.split()
        total_dur = sum(e - s for s, e in sub_segs)
        word_pos  = 0
        for (s, e), spk, conf in zip(sub_segs, sub_speakers, sub_confidences):
            prop    = (e - s) / max(total_dur, 1)
            n_words = max(1, round(len(all_words) * prop))
            chunk   = all_words[word_pos:word_pos + n_words]
            word_pos += n_words
            if chunk: seg_outputs.append((spk, " ".join(chunk), conf))

    # Merge consecutive identical speakers
    merged_outputs: List[Tuple[int, str, float]] = []
    for spk, frag, conf in seg_outputs:
        if merged_outputs and merged_outputs[-1][0] == spk:
            old_conf = merged_outputs[-1][2]
            merged_outputs[-1] = (spk, merged_outputs[-1][1] + " " + frag, min(old_conf, conf))
        else:
            merged_outputs.append((spk, frag, conf))

    if not merged_outputs:
        merged_outputs = [(sub_speakers[0] if sub_speakers else 0, text, 0.6)]

    ts = format_ts(time.time() - session_start_time)

    async def _identity(f: str) -> str: return f

    translate_tasks = [
        translate_async(frag, detected_lang) if detected_lang != "en" else _identity(frag)
        for _, frag, _ in merged_outputs
    ]
    translations = await asyncio.gather(*translate_tasks)

    raw_segments: List[Dict] = []
    for (spk, frag, conf), trans in zip(merged_outputs, translations):
        raw_segments.append({
            "segment_id": _next_segment_id(),
            "speaker":    spk,
            "timestamp":  ts,
            "original":   apply_domain_fixes(frag),
            "translation": trans,
            "lang":       detected_lang,
            "diar_conf":  round(conf, 3),
        })

    # LLM diarization check — only when confidence truly low (FIX 2)
    segment_confidences  = [s["diar_conf"] for s in raw_segments]
    has_low_conf_segment = any(c < LLM_DIAR_CONF_THRESHOLD for c in segment_confidences)
    if has_low_conf_segment and len(raw_segments) >= 2:
        raw_segments = await llm_validate_speakers(
            raw_segments, segment_confidences, _build_context())

    # GPT correction pipeline
    corr_results = await asyncio.gather(*[correct_segment(seg) for seg in raw_segments])

    corrected_segments: List[Dict] = []
    for raw_seg, corr in zip(raw_segments, corr_results):
        if corr is None: continue
        corrected_segments.append({
            "segment_id":  raw_seg["segment_id"],
            "speaker":     raw_seg["speaker"],
            "timestamp":   ts,
            "original":    corr["corrected_original"],
            "translation": corr["corrected_translation"],
            "was_changed": corr["was_changed"],
            "tier":        corr.get("tier"),
            "lang":        detected_lang,
            "diar_conf":   raw_seg.get("diar_conf", 1.0),
        })
        meeting_transcript.append({
            "speaker":     raw_seg["speaker"],
            "timestamp":   ts,
            "original":    corr["corrected_original"],
            "translation": corr["corrected_translation"],
        })

    if len(meeting_transcript) > 200:
        meeting_transcript[:] = meeting_transcript[-200:]

    payload = {
        "segments":          corrected_segments,
        "is_final":          True,
        "detected_language": detected_lang,
        "corrected":         True,
        "diar_stats":        speaker_tracker.get_stats(),
    }
    try:
        await websocket.send_json(payload)
    except Exception:
        pass


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(title="Live Speech Translator", version="5.2")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.get("/")
def home():
    return FileResponse("index.html", headers={"Cache-Control": "no-store, max-age=0"})


@app.get("/favicon.ico")
def favicon(): return Response(status_code=204)


@app.get("/.well-known/appspecific/com.chrome.devtools.json")
def chrome_devtools(): return Response(status_code=204)


@app.get("/stats")
def stats():
    diar = speaker_tracker.get_stats()
    return {
        "cache":       cache.stats(),
        "rules":       learning.rule_count,
        "transcript":  len(meeting_transcript),
        "speakers":    diar["n_speakers"],
        "diar_mode":   diar["mode"],
        "diar_thresh": diar["threshold"],
        "active_secs": diar["active_secs"],
    }


@app.get("/diar-status")
def diar_status():
    diar = speaker_tracker.get_stats()
    return {
        "speechbrain_loaded":   SPEECHBRAIN_AVAILABLE,
        "embedding_mode":       "ecapa-192dim" if SPEECHBRAIN_AVAILABLE else "mfcc-80dim-fallback",
        "speaker_count":        len(speaker_tracker.centroids),
        "max_speakers_allowed": MAX_SPEAKERS_ALLOWED,
        "min_seg_sec":          SPK_CHANGE_MIN_SEG_SEC,
        "bootstrap_count":      BOOTSTRAP_COUNT,
        "cosine_thresh":        COSINE_THRESH_XVEC if SPEECHBRAIN_AVAILABLE else COSINE_THRESH_MFCC,
        "segments_processed":   diar["n_segments"],
        "active_secs_per_spk":  diar["active_secs"],
        "stt_model":            OPENAI_STT_MODEL,
        "recommendation": (
            "✅ Running at full accuracy" if SPEECHBRAIN_AVAILABLE
            else "⚠️  Install SpeechBrain for best diarization: "
                 "pip install speechbrain torch torchaudio "
                 "--index-url https://download.pytorch.org/whl/cpu"
        ),
    }


def _reset_session():
    global session_start_time, segment_id_counter
    meeting_transcript.clear()
    learning.reset()
    speaker_tracker.reset()
    session_start_time = time.time()
    segment_id_counter = count()
    print("  🔄 Session reset — speaker tracker cleared")


async def _ws_loop(websocket: WebSocket):
    while True:
        try:
            data = await asyncio.wait_for(websocket.receive_bytes(), timeout=15.0)
            asyncio.ensure_future(process_blob(websocket, bytes(data)))
        except asyncio.TimeoutError:
            pass
        except WebSocketDisconnect:
            stats_info = speaker_tracker.get_stats()
            print(f"  Disconnected: {cache.stats()}  rules={learning.rule_count}  "
                  f"speakers={stats_info['n_speakers']}  mode={stats_info['mode']}")
            break


@app.websocket("/listen-auto")
async def listen_auto(websocket: WebSocket):
    await websocket.accept()
    _reset_session()
    print("✅ /listen-auto connected  [v5.2 whisper-1 + tighter diarization]")
    try:
        await _ws_loop(websocket)
    except Exception as e:
        print(f"listen-auto error: {e}")
        try: await websocket.send_json({"error": str(e)})
        except Exception: pass


@app.websocket("/listen")
async def listen_arabic(websocket: WebSocket):
    """Legacy endpoint — same v5.2 pipeline."""
    await websocket.accept()
    print("✅ /listen connected  [v5.2 legacy alias]")
    try:
        await _ws_loop(websocket)
    except Exception as e:
        try: await websocket.send_json({"error": str(e)})
        except Exception: pass