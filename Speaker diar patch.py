"""
speaker_diar_patch.py  —  Drop-in fix for 9-speaker explosion
══════════════════════════════════════════════════════════════════
HOW TO APPLY
────────────
Add ONE import line at the very top of your main.py (right after the other imports):

    from speaker_diar_patch import FixedSpeakerTracker, NUM_SPEAKERS

Then replace the ONE line that creates speaker_tracker near line 615:

    # OLD:
    speaker_tracker = AdvancedSpeakerTracker()

    # NEW:
    speaker_tracker = FixedSpeakerTracker(num_speakers=NUM_SPEAKERS)

Also change these 3 constants near lines 148-158 in main.py:

    MAX_SEGS_PER_BLOB      = 2      # was 4  — max 2 speaker turns per blob
    SPK_CHANGE_MIN_SEG_SEC = 6.0    # was 3.0 — very conservative splitting
    SPK_CHANGE_HOP_MS      = 200    # was 150 — larger hop = fewer boundaries

That's it. Run uvicorn as normal. You will always get exactly NUM_SPEAKERS=2.

WHY THE OLD CODE BROKE
────────────────────────
Your ECAPA log shows cosine similarities of -0.074 to 0.475 even for the SAME
speaker talking continuously. Root causes:
  1. WebM/Opus codec crushes voice characteristics — ECAPA was trained on PCM
  2. Short 1-3s sub-segments don't give ECAPA enough frames
  3. Arabic mixed with English confuses the English-trained ECAPA model

With threshold=0.45-0.50, almost every sub-segment looked "too different" from
the last → new speaker created → 9 speakers in 40 seconds.

HOW THE FIX WORKS
──────────────────
  1. NUM_SPEAKERS=2 is FIXED — the tracker will NEVER create speaker 3, 4, etc.
  2. For the first 2 sub-segments we create one centroid each (bootstrap).
  3. After that, every embedding is always assigned to the NEAREST of the 2
     centroids — no threshold check, no new-speaker creation.
  4. Every 8 segments we run KMeans(k=2) on ALL accumulated embeddings to
     recalibrate centroids (fixes drift from EMA updates).
  5. Blob splitting is capped at MAX_SEGS_PER_BLOB=2 — one speaker turn max.
══════════════════════════════════════════════════════════════════
"""

import threading
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

# ── Import what we need from your existing code ───────────────────────────────
# These are already in main.py — we reuse them
try:
    from shared_components import SAMPLE_RATE, extract_mean_pitch
except ImportError:
    SAMPLE_RATE = 16_000
    def extract_mean_pitch(pcm, sr):
        return float(np.mean(np.abs(np.diff(np.sign(pcm)))) / 2.0)

# ── ONLY CHANGE THIS NUMBER to match your actual speaker count ────────────────
NUM_SPEAKERS: int = 2


class FixedSpeakerTracker:
    """
    Speaker tracker with a FIXED speaker count (default: 2).

    Core difference from AdvancedSpeakerTracker:
      • Never creates more than num_speakers centroids — ever.
      • After bootstrap (first num_speakers unique segments), all embeddings
        are assigned by nearest-centroid (cosine similarity), no threshold.
      • Periodic KMeans(k=num_speakers) reclustering every 8 segments to
        prevent centroid drift from EMA updates.
      • get_stats() is API-compatible with AdvancedSpeakerTracker so the
        rest of main.py works unchanged.
    """

    def __init__(self, num_speakers: int = 2):
        self.num_speakers = num_speakers
        self.lock         = threading.Lock()
        # Centroid state
        self.centroids:   List[np.ndarray] = []
        self.histories:   List[deque]      = []
        self.active_secs: List[float]      = []
        self.confidences: List[float]      = []
        # Full embedding history (for reclustering)
        self.embeddings:  List[np.ndarray] = []
        self.labels:      List[int]        = []
        self.timestamps:  List[float]      = []
        # Misc
        self.seg_count = 0
        self.turn_memory: deque = deque(maxlen=20)
        # Mode flags (set by main.py when SpeechBrain loads)
        self.use_xvec  = False
        self.threshold = 0.0   # not used for assignment, kept for get_stats()

    # ── Reuse embedding extractors from AdvancedSpeakerTracker ───────────────
    # We delegate to the same methods — just copy the two extract_* methods
    # from main.py here so this file is self-contained.

    def extract_xvector(self, pcm_float: np.ndarray) -> Optional[np.ndarray]:
        """ECAPA-TDNN via SpeechBrain (if available)."""
        try:
            # Import lazily — SpeechBrain may not be loaded yet
            import torch
            from main import sb_classifier, SPEECHBRAIN_AVAILABLE
            if not SPEECHBRAIN_AVAILABLE or sb_classifier is None:
                return None
            wav = pcm_float / (np.max(np.abs(pcm_float)) + 1e-9)
            wav_t = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                emb = sb_classifier.encode_batch(wav_t).squeeze().cpu().numpy()
            norm = np.linalg.norm(emb)
            return (emb / norm) if norm > 1e-9 else None
        except Exception:
            return None

    def extract_mfcc_embedding(
        self, pcm: np.ndarray, sr: int = SAMPLE_RATE
    ) -> Optional[np.ndarray]:
        """80-dim enhanced MFCC (same as AdvancedSpeakerTracker.extract_mfcc_enhanced)."""
        if len(pcm) < int(sr * 0.4):
            return None

        sig = np.append(pcm[0], pcm[1:] - 0.97 * pcm[:-1])
        sig /= (np.max(np.abs(sig)) + 1e-9)

        frame_len  = int(0.025 * sr)
        frame_step = int(0.010 * sr)
        n_frames   = (len(sig) - frame_len) // frame_step
        if n_frames < 3:
            return None

        idx    = np.arange(frame_len) + np.arange(n_frames)[:, None] * frame_step
        frames = sig[idx] * np.hamming(frame_len)
        ene    = np.sum(frames ** 2, axis=1)

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

        voiced = ene >= np.percentile(ene, 30)
        if voiced.sum() < 3:
            voiced = np.ones(n_frames, dtype=bool)
        mv = mfcc[voiced]

        def delta(f):
            d = np.zeros_like(f)
            for t in range(len(f)):
                d[t] = (f[min(t+2, len(f)-1)] - f[max(t-2, 0)]) / 4.0
            return d

        d1, d2 = delta(mv), delta(delta(mv))
        emb = np.concatenate([np.mean(mv,axis=0), np.mean(d1,axis=0),
                               np.mean(d2,axis=0), np.var(mv,axis=0)])
        pitch = extract_mean_pitch(pcm, sr)
        emb   = np.append(emb, pitch * 5.0)
        emb   = (emb - np.mean(emb)) / (np.std(emb) + 1e-9)
        return emb

    def extract_embedding(self, pcm: np.ndarray) -> Optional[np.ndarray]:
        """Try ECAPA first, fall back to MFCC."""
        # Check if SpeechBrain loaded since last reset
        try:
            from main import SPEECHBRAIN_AVAILABLE
            if SPEECHBRAIN_AVAILABLE and not self.use_xvec:
                self.use_xvec = True
                print("  🎙️  FixedSpeakerTracker: switched to ECAPA x-vec mode")
        except ImportError:
            pass

        emb = None
        if self.use_xvec:
            emb = self.extract_xvector(pcm)
        if emb is None:
            emb = self.extract_mfcc_embedding(pcm)
        if emb is None:
            return None

        norm = np.linalg.norm(emb)
        return (emb / norm) if norm > 1e-9 else None

    # ── Core: identify_speaker ────────────────────────────────────────────────
    def identify_speaker(
        self, embedding: np.ndarray, seg_duration_sec: float = 1.0
    ) -> Tuple[int, float]:
        """
        Assign embedding to one of exactly self.num_speakers speakers.
        Returns (speaker_id, confidence).

        RULES:
          Phase 1 (bootstrap): first num_speakers segments get unique IDs.
          Phase 2 (normal):    always assign to nearest centroid — no new speakers.
          Every 8 segs:        KMeans(k=num_speakers) recalibrates all centroids.
        """
        with self.lock:
            norm_val = np.linalg.norm(embedding)
            if norm_val < 1e-9:
                return 0, 0.5
            emb_n = embedding / norm_val

            # ── PHASE 1: Bootstrap ────────────────────────────────────────────
            if len(self.centroids) < self.num_speakers:
                spk = self._add_speaker(embedding, seg_duration_sec)
                self._record(embedding, spk)
                print(f"  🟢 Bootstrap Speaker {spk+1}/{self.num_speakers}")
                return spk, 1.0

            # ── PHASE 2: Always assign to nearest ────────────────────────────
            sims = []
            for c in self.centroids:
                cn = np.linalg.norm(c)
                if cn < 1e-9:
                    sims.append(-1.0)
                else:
                    sims.append(float(np.dot(emb_n, c / cn)))

            best_idx = int(np.argmax(sims))
            best_sim = sims[best_idx]
            spk      = best_idx

            # EMA centroid update (slow — prevents drift)
            alpha = 0.05
            self.centroids[spk] = ((1 - alpha) * self.centroids[spk]
                                   + alpha * embedding)
            self.histories[spk].append(embedding.copy())
            self.active_secs[spk] += seg_duration_sec

            # Confidence: map similarity to [0,1] range roughly
            confidence = max(0.0, min(1.0, (best_sim + 1.0) / 2.0))

            self._record(embedding, spk)
            self.turn_memory.append((spk, time.time()))

            # Periodic KMeans reclustering
            if self.seg_count % 8 == 0 and self.seg_count > 0:
                self._recluster_kmeans()

            return spk, float(confidence)

    def _add_speaker(self, embedding: np.ndarray, dur: float) -> int:
        """Add a new speaker centroid. Only called during bootstrap."""
        spk = len(self.centroids)
        self.centroids.append(embedding.copy())
        self.histories.append(deque(maxlen=30))
        self.histories[-1].append(embedding.copy())
        self.confidences.append(1.0)
        self.active_secs.append(dur)
        return spk

    def _record(self, embedding: np.ndarray, spk: int):
        """Record embedding in history for reclustering."""
        self.embeddings.append(embedding.copy())
        self.labels.append(spk)
        self.timestamps.append(time.time())
        self.seg_count += 1

    def _recluster_kmeans(self):
        """
        Re-run KMeans(k=num_speakers) on recent embeddings.
        This corrects centroid drift and re-identifies who-is-who after
        a long gap where EMA may have shifted a centroid wrongly.
        """
        n = len(self.embeddings)
        if n < self.num_speakers * 4:
            return   # not enough data yet
        try:
            recent = min(n, 200)
            X = normalize(np.array(self.embeddings[-recent:]))
            km = KMeans(
                n_clusters   = self.num_speakers,
                n_init       = 10,
                max_iter     = 300,
                random_state = 42,
                tol          = 1e-4,
            )
            km.fit(X)

            old_n = len(self.centroids)
            # Map KMeans centers back to original speaker IDs
            # (preserve order by matching each KMeans centre to nearest old centroid)
            new_centers = km.cluster_centers_
            used = set()
            mapping = {}   # kmeans_label → speaker_id
            for ki in range(self.num_speakers):
                best_s, best_v = 0, -2.0
                for si in range(len(self.centroids)):
                    if si in used:
                        continue
                    c  = self.centroids[si]
                    cn = np.linalg.norm(c)
                    if cn < 1e-9:
                        continue
                    v = float(np.dot(new_centers[ki],
                                     c / cn))
                    if v > best_v:
                        best_v, best_s = v, si
                mapping[ki] = best_s
                used.add(best_s)

            # Update centroids in-place (don't reorder — keep speaker IDs stable)
            for ki, si in mapping.items():
                self.centroids[si] = new_centers[ki].copy()

            changed = any(
                np.linalg.norm(self.centroids[mapping[ki]] - new_centers[ki]) > 0.05
                for ki in range(self.num_speakers)
            )
            if changed:
                print(f"  🔄 KMeans({self.num_speakers}) reclustered {recent} segs "
                      f"→ centroids updated")

        except Exception as e:
            print(f"  ⚠️  KMeans recluster error: {e}")

    # ── reset / force_minimum_speakers (API compatibility) ────────────────────
    def reset(self):
        with self.lock:
            self.centroids.clear()
            self.histories.clear()
            self.active_secs.clear()
            self.confidences.clear()
            self.embeddings.clear()
            self.labels.clear()
            self.timestamps.clear()
            self.seg_count  = 0
            self.turn_memory.clear()
            # Re-detect SpeechBrain availability
            try:
                from main import SPEECHBRAIN_AVAILABLE, COSINE_THRESH_XVEC, COSINE_THRESH_MFCC
                self.use_xvec  = SPEECHBRAIN_AVAILABLE
                self.threshold = COSINE_THRESH_XVEC if SPEECHBRAIN_AVAILABLE else COSINE_THRESH_MFCC
            except ImportError:
                self.use_xvec  = False
                self.threshold = 0.0
            mode = "ECAPA x-vec" if self.use_xvec else "MFCC"
            print(f"  🔄 FixedSpeakerTracker reset — {mode} mode, "
                  f"num_speakers={self.num_speakers} (FIXED)")

    def force_minimum_speakers(self, min_spk: int = 2):
        """API-compatible stub — not needed since count is always fixed."""
        pass   # No-op: we always have exactly num_speakers

    def get_stats(self) -> Dict:
        """API-compatible with AdvancedSpeakerTracker.get_stats()."""
        mode = "ecapa-xvec" if self.use_xvec else "mfcc-enhanced"
        return {
            "n_speakers":  len(self.centroids),
            "n_segments":  self.seg_count,
            "mode":        mode,
            "threshold":   round(self.threshold, 3),
            "active_secs": [round(s, 1) for s in self.active_secs],
        }