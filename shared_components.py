"""
shared_components.py — Shared Optimization Layer  (v3.2 — live-meeting fixes)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHANGES vs v3.1
────────────────
1. OPENAI_STT_MODEL  "gpt-4o-transcribe" → "whisper-1"
   gpt-4o-transcribe only accepts response_format="json"|"text".
   whisper-1 accepts response_format="verbose_json" with
   timestamp_granularities[]=word, giving per-word {start, end} objects.
   Without word timestamps every blob falls back to a crude proportional
   word-split that distributes words evenly by duration — systematically
   wrong whenever speakers talk at different speeds or pause mid-sentence.
   Cost: IDENTICAL ($0.006 / audio-minute for both models).

2. PipelineDecisionEngine.TRIVIAL_SET expanded (+18 common meeting phrases)
   Single-word and very-short English-only segments now skip GPT entirely.
   Estimated 25-35 % fewer correction API calls per meeting.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations
import re, json, time, difflib
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# ── API constants ─────────────────────────────────────────────────────────────
# whisper-1 is the only OpenAI transcription model that returns verbose_json
# with word-level timestamps via the REST multipart API.
OPENAI_STT_MODEL  = "whisper-1"       # ← was "gpt-4o-transcribe"
OPENAI_COT_MODEL  = "gpt-4.1"
OPENAI_MINI_MODEL = "gpt-4.1-mini"
OPENAI_STT_URL    = "https://api.openai.com/v1/audio/transcriptions"
OPENAI_CHAT_URL   = "https://api.openai.com/v1/chat/completions"

SAMPLE_RATE = 16_000

# ── STT prompt ────────────────────────────────────────────────────────────────
STT_PROMPT_HINT = (
    # ── Whisper prompt: vocabulary primer ONLY (≤ 120 tokens) ─────────────────
    # Whisper treats this as "prior audio context", NOT as instructions.
    # Long instruction text (e.g. "do NOT hallucinate") leaks into transcripts.
    # Solution: list domain terms only — Whisper learns vocabulary, not text.
    "Nafis, Waseel, Clinicy, RCM, EMR, EHR, HIS, FHIR, HL7, ERP, DICOM, "
    "NPHIES, MOH, CCHI, ICD-10, CPT, UAT, QA, API, "
    "radiology, laboratory, pharmacy, resubmission, eligibility, "
    "pre-authorization, revenue cycle, sandbox, staging, production, "
    "integration, mapping, configuration, module, workflow, "
    "approval, rejection, queue, escalation, notification, "
    "reconciliation, validation, synchronisation, master data, "
    "insurance claim, patient record, adjudication, clearinghouse."
)

GULF_PHRASES: Dict[str, str] = {
    "يا ربي": "Oh God", "يارب": "Oh God", "والله": "By God", "يلا": "Let's go",
    "طيب": "okay", "تمام": "perfect", "ماشي": "okay", "أيوه": "yes",
    "لأ": "no", "زين": "good", "التكامل": "integration", "الربط": "integration",
    "الخريطة": "mapping", "الإعدادات": "configuration", "الوحدة": "module",
    "سير العمل": "workflow", "الموافقة": "approval", "الرفض": "rejection",
    "قائمة الانتظار": "queue", "التصعيد": "escalation", "الإشعار": "notification",
    "التسوية": "reconciliation", "التحقق": "validation", "المزامنة": "synchronisation",
    "البيانات الرئيسية": "master data", "سجل المريض": "patient record",
    "الموافقة المسبقة": "pre-authorization", "حالة المطالبة": "claim status",
    "النظام": "the system", "البيئة": "environment", "الإنتاج": "production",
}

NOISE_TAGS: frozenset = frozenset({
    "[audio dropout]", "[unclear]", "[noise/silence]", "[hallucination]", "...",
})

# ── Whisper silence hallucinations ────────────────────────────────────────────
# Whisper (trained on YouTube) generates these phrases for silent/quiet audio.
# Any STT result that matches (case-insensitive, stripped) is discarded.
WHISPER_HALLUCINATIONS: frozenset = frozenset({
    # English filler
    "thank you", "thank you very much", "thanks", "thank you so much",
    "thank you for watching", "thanks for watching",
    "please subscribe", "don't forget to subscribe",
    "subscribe to the channel", "don't forget to subscribe to the channel",
    "like and subscribe", "hit the subscribe button",
    "see you next time", "see you in the next video",
    "bye", "goodbye", "bye bye",
    "you", ".", "..", "...", "okay", "ok",
    # Arabic equivalents Whisper hallucinates
    "شكراً", "شكرا", "شكراً جزيلاً", "شكرا جزيلا",
    "لا تنسى الاشتراك", "اشترك في القناة",
    "لا تنسى الاشتراك في القناة",
})

_HALL_PARTIAL: tuple = (
    # Partial-match strings — if the entire output IS only one of these (no
    # real content alongside it) it is a hallucination.
    "thank you for watching", "don't forget to subscribe",
    "لا تنسى الاشتراك", "subscribe to the channel",
    "translated by",          # "Translated by Nancy Qanqar" etc.
)


def is_whisper_hallucination(text: str) -> bool:
    """Return True if *text* is a known Whisper silence hallucination."""
    t = text.strip().rstrip(".!،.").strip().lower()
    if not t:
        return True
    # Exact match against blocklist
    if t in WHISPER_HALLUCINATIONS or t.rstrip(".") in WHISPER_HALLUCINATIONS:
        return True
    # Very short output that contains only a partial hallucination trigger
    if len(t.split()) <= 6:
        for partial in _HALL_PARTIAL:
            if partial.lower() in t:
                return True
    return False

COMMON_FIXES: Dict[str, str] = {
    "radios": "radiology", "radiose": "radiology",
    "efficient": "environment", "administrate": "process",
    "insurance cycle": "revenue cycle",
    "respond": "resubmission",
    "h3": "HIS", "ncm": "RCM", "pcm": "RCM",
    "wallue": "Waseel", "clinisey": "Clinicy", "clinizy": "Clinicy",
    "decloy": "deploy", "stabox": "sandbox", "sandbok": "sandbox",
    "6sting": "staging", "mirment": "environment", "invirment": "environment",
    "anshurns": "insurance",
    "nifis": "Nafis", "nefis": "Nafis", "wazeel": "Waseel",
    "integrashion": "integration", "integrasion": "integration",
    "konfig": "configuration", "approoval": "approval", "appruval": "approval",
    "rejecshion": "rejection", "notifikashion": "notification",
    "validashion": "validation", "rekonsilyashion": "reconciliation",
    "synk": "synchronisation", "sinkronizashion": "synchronisation",
    "mashter data": "master data", "pree-auth": "pre-authorization",
    "prior approov": "prior approval",
    "resubmit": "resubmission", "re-submission": "resubmission",
    "adjudikashion": "adjudication",
}

TECH_WORDS: List[str] = [
    "RCM", "ERP", "radiology", "laboratory", "pharmacy", "claim",
    "staging", "sandbox", "production", "environment", "deployment",
    "resubmission", "eligibility", "insurance", "authorization",
    "Waseel", "Clinicy", "Nafis", "HIS", "EMR", "EHR", "FHIR",
    "ICD-10", "CPT", "UAT", "QA", "API", "DICOM",
    "integration", "mapping", "configuration", "module", "workflow",
    "approval", "rejection", "queue", "escalation", "notification",
    "reconciliation", "validation", "synchronisation", "master data",
    "NPHIES", "MOH", "CCHI", "pre-authorization", "claim status",
    "adjudication", "clearinghouse", "ERA", "EDI",
]


def apply_domain_fixes(text: str) -> str:
    result = text
    for wrong, correct in COMMON_FIXES.items():
        result = re.sub(rf"\b{re.escape(wrong)}\b", correct, result, flags=re.IGNORECASE)
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [1]  SEMANTIC CACHE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SemanticCache:
    def __init__(self, max_size: int = 400, fuzzy_threshold: float = 0.93, ttl: int = 3600):
        self._store: Dict[str, OrderedDict] = {}
        self._ts:    Dict[str, Dict[str, float]] = {}
        self.max_size = max_size
        self.fuzz     = fuzzy_threshold
        self.ttl      = ttl
        self.hits = 0
        self.misses = 0

    def _norm(self, t: str) -> str:
        return re.sub(r'\s+', ' ', t.lower().strip())

    def _ensure(self, kind: str):
        if kind not in self._store:
            self._store[kind] = OrderedDict()
            self._ts[kind] = {}

    def get(self, text: str, kind: str = "correction") -> Optional[Dict]:
        self._ensure(kind)
        key = self._norm(text)
        store = self._store[kind]
        now = time.time()
        ts = self._ts[kind]
        if key in store and now - ts.get(key, 0) < self.ttl:
            store.move_to_end(key)
            self.hits += 1
            return store[key]
        for k in list(store.keys())[-40:]:
            if now - ts.get(k, 0) < self.ttl:
                if difflib.SequenceMatcher(None, key, k).ratio() >= self.fuzz:
                    store.move_to_end(k)
                    self.hits += 1
                    return store[k]
        self.misses += 1
        return None

    def set(self, text: str, val: Dict, kind: str = "correction"):
        self._ensure(kind)
        key = self._norm(text)
        store = self._store[kind]
        if len(store) >= self.max_size:
            store.pop(next(iter(store)))
        store[key] = val
        self._ts[kind][key] = time.time()

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def stats(self) -> str:
        total = self.hits + self.misses
        return f"cache {self.hits}/{total} hits ({100*self.hit_rate():.0f}%)"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [2]  TOKEN BUDGET MANAGER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TokenBudgetManager:
    BUDGETS: Dict[str, int] = {"trivial": 60, "simple": 150, "standard": 512, "full_cot": 1024}

    def get(self, tier: str) -> int:
        return self.BUDGETS.get(tier, 512)

    def compress_context(self, transcript: List[Dict], max_entries: int = 5) -> str:
        if not transcript:
            return "(session start)"
        tail = transcript[-max_entries:]
        older = transcript[:-max_entries] if len(transcript) > max_entries else []
        lines: List[str] = []
        if older:
            snips = [f"Spk{s['speaker']+1}:{' '.join(s['original'].split()[:5])}…" for s in older[-8:]]
            lines.append("[Earlier] " + " | ".join(snips))
        for s in tail:
            lines.append(f"  Speaker {s['speaker']+1} [{s.get('timestamp','--:--')}]: "
                         f"{s['original']} → {s.get('translation', '')}")
        return "\n".join(lines)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [3]  AUTO LEARNING LOOP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class AutoLearningLoop:
    PROMOTE_THRESHOLD = 1

    def __init__(self):
        self._corr: Dict[str, str] = {}
        self._freq: Dict[str, int] = {}
        self._rules: Dict[str, str] = {}

    def record(self, wrong: str, correct: str):
        if not wrong or not correct or wrong.strip() == correct.strip():
            return
        k = wrong.strip().lower()
        self._corr[k] = correct.strip()
        self._freq[k] = self._freq.get(k, 0) + 1
        if self._freq[k] >= self.PROMOTE_THRESHOLD and k not in self._rules:
            self._rules[k] = correct.strip()
            print(f"  📌 Rule promoted: '{wrong}' → '{correct}'")

    def apply_rules(self, text: str) -> Optional[str]:
        if not self._rules:
            return None
        low = text.lower()
        result = text
        changed = False
        for wrong, correct in self._rules.items():
            if wrong in low:
                result = re.sub(re.escape(wrong), correct, result, flags=re.IGNORECASE)
                changed = True
        return result if changed else None

    def context_str(self, limit: int = 10) -> str:
        items = list(self._corr.items())[-limit:]
        return "\n".join(f"  '{w}' = '{c}'" for w, c in items) or "(none)"

    def reset(self):
        self._corr.clear(); self._freq.clear(); self._rules.clear()

    @property
    def rule_count(self) -> int:
        return len(self._rules)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [4]  ENHANCED VAD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class EnhancedVAD:
    # energy_floor raised 120 → 350: WebM/Opus blobs with RMS < 350 are
    # near-silent after decoding — sending them to Whisper causes hallucinations
    # ("Thank you", "Thank you for watching", etc.)
    def __init__(self, energy_floor: float = 350.0, zcr_max: float = 0.88,
                 voiced_min_frac: float = 0.30):
        self.energy_floor     = energy_floor
        self.zcr_max          = zcr_max
        self.voiced_min_frac  = voiced_min_frac   # ≥30% of frames must be voiced

    def is_speech(self, audio: np.ndarray) -> Tuple[bool, Dict]:
        if len(audio) == 0:
            return False, {"rms": 0.0, "zcr": 0.0, "voiced_frac": 0.0}
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        if rms < self.energy_floor:
            return False, {"rms": rms, "zcr": 0.0, "voiced_frac": 0.0}
        signs  = np.sign(audio)
        crosses = int(np.sum(signs[:-1] != signs[1:]))
        zcr    = float(crosses / max(len(audio), 1))
        if zcr > self.zcr_max:
            return False, {"rms": rms, "zcr": zcr, "voiced_frac": 0.0}
        # Voiced-frames gate: split into 20ms frames, count those above
        # half the median energy — if fewer than voiced_min_frac are active,
        # this is mostly silence with a few spikes (music, paper rustle, etc.)
        frame_size  = max(int(16_000 * 0.020), 1)
        n_frames    = len(audio) // frame_size
        if n_frames >= 5:
            frames      = audio[:n_frames * frame_size].reshape(n_frames, frame_size)
            frame_rms   = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
            median_rms  = float(np.median(frame_rms))
            voiced_frac = float(np.mean(frame_rms >= median_rms * 0.5))
        else:
            voiced_frac = 1.0
        if voiced_frac < self.voiced_min_frac:
            return False, {"rms": rms, "zcr": zcr, "voiced_frac": voiced_frac}
        return True, {"rms": rms, "zcr": zcr, "voiced_frac": voiced_frac}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [5]  PIPELINE DECISION ENGINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class PipelineDecisionEngine:
    # Expanded: 18 extra common meeting back-channel phrases skip GPT entirely
    TRIVIAL_SET = {
        # English
        "yes", "no", "ok", "okay", "hello", "hi", "bye", "thanks", "good",
        "sure", "right", "hmm", "yeah", "yep", "great", "fine", "alright",
        "got it", "understood", "correct", "exactly", "absolutely", "agreed",
        "noted", "perfect", "clear", "sounds good", "let me check",
        "one moment", "just a moment", "hold on", "please go ahead",
        "go ahead", "i see", "okay thank you", "thank you", "thanks okay",
        "of course", "i agree", "definitely", "certainly", "indeed",
        # Arabic
        "طيب", "ماشي", "أيوه", "لأ", "تمام", "زين", "مرحبا",
        "نعم", "لا", "حسنا", "تفضل", "شكرا", "ممتاز", "صحيح",
        "بالضبط", "موافق", "مفهوم", "طبعا", "بالتأكيد",
    }

    TECH_TERMS = set(
        "RCM EMR EHR HIS FHIR HL7 UAT QA API staging sandbox "
        "production deployment environment Waseel Clinicy backend "
        "frontend database server sprint release build cycle IRP DICOM "
        "integration mapping configuration module workflow approval "
        "rejection queue escalation notification reconciliation "
        "validation synchronisation NPHIES MOH CCHI Nafis".split()
    )

    MIN_CONF         = 0.45
    SIMPLE_CONF      = 0.75
    MAX_SIMPLE_WORDS = 10

    def decide(self, text: str, conf: float = 1.0,
               learning: Optional[AutoLearningLoop] = None) -> str:
        s = text.strip()
        if not s or s in NOISE_TAGS:
            return "skip"
        if conf < self.MIN_CONF:
            return "trivial"
        words = s.split()

        # Full phrase in trivial set
        if s.lower() in self.TRIVIAL_SET:
            return "trivial"
        # Single word not a tech term → trivial
        if len(words) == 1 and s.upper() not in self.TECH_TERMS:
            return "trivial"
        # ≤3 pure-English words with no tech terms → trivial
        if (len(words) <= 3
                and not any('\u0600' <= c <= '\u06FF' for c in s)
                and not any(w.upper() in self.TECH_TERMS for w in words)):
            return "trivial"

        if learning and learning.apply_rules(s) is not None:
            return "rule"

        has_ar   = any('\u0600' <= c <= '\u06FF' for c in s)
        has_en   = any(c.isalpha() and c.isascii() for c in s)
        has_tech = any(w.upper() in self.TECH_TERMS for w in words)

        if (len(words) <= self.MAX_SIMPLE_WORDS
                and not (has_ar and has_en)
                and not has_tech
                and conf >= self.SIMPLE_CONF):
            return "simple"
        return "full_cot"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [6]  VALIDATION LAYER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class ValidationLayer:
    HALL_RE = re.compile(
        r'[\u0400-\u04FF]{4,}|[\u4E00-\u9FFF]{3,}|[\u0900-\u097F]{4,}'
        r'|[\u0C00-\u0C7F]{3,}|dziękuję|спасибо|merci|gracias'
    )
    DIFF_MIN = 0.12
    SIM_MIN  = 0.20

    def is_hallucination(self, t: str) -> bool:
        return bool(self.HALL_RE.search(t))

    def semantic_similarity(self, a: str, b: str) -> float:
        aw = set(a.lower().split()); bw = set(b.lower().split())
        if not aw and not bw: return 1.0
        if not aw or not bw: return 0.0
        return len(aw & bw) / len(aw | bw)

    def was_changed(self, orig: str, corr: str) -> bool:
        r = difflib.SequenceMatcher(None, orig.lower().split(), corr.lower().split()).ratio()
        return (1.0 - r) >= self.DIFF_MIN

    def domain_score(self, text: str) -> int:
        t = text.lower()
        return sum(1 for w in TECH_WORDS if w.lower() in t)

    def validate(self, raw: str, corr: str, trans: str) -> Tuple[str, str, bool]:
        if self.is_hallucination(corr): corr = raw
        if self.is_hallucination(trans): trans = "[noise/silence]"; corr = "..."
        if self.semantic_similarity(raw, corr) < self.SIM_MIN and corr != "...":
            corr = raw
        if self.domain_score(raw) > 0 and self.domain_score(corr) < self.domain_score(raw):
            print("  🛡️  Domain guard — reverting correction"); corr = raw
        return corr, trans, self.was_changed(raw, corr)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [7]  PROMPT TEMPLATE MANAGER  v3.2
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class PromptTemplateManager:
    VERSION = "3.2"

    SYSTEM_SIMPLE = (
        "STT post-processor v{version}: Gulf Arabic/English healthcare IT.\n"
        "DOMAIN GLOSSARY (never change): Nafis, Waseel, Clinicy, RCM, EMR, EHR, HIS, FHIR, ERP, API, "
        "radiology, laboratory, pharmacy, resubmission, revenue cycle, ICD-10, CPT, UAT, QA, "
        "sandbox, staging, production, integration, mapping, configuration, module, workflow, "
        "approval, rejection, queue, escalation, notification, reconciliation, validation, "
        "synchronisation, NPHIES, MOH, CCHI.\n"
        "Fix ONLY: phonetic mis-spellings, tech-term errors, word-dropout loops (3+ identical → once), "
        "and foreign-script hallucinations. PROTECT domain terms. Foreign script → '...'.\n"
        'Return ONLY JSON array: [{"idx":N,"corrected":"...","translation":"..."}]'
    )

    SYSTEM_FULL_COT = """\
Expert Arabic-English STT post-processor — Gulf/Levantine healthcare IT. v{version}
STRICT GROUNDING: never invent content. Apply checks a-e silently.
  a) Collapse 3+ identical consecutive words → once
  b) Foreign script (Cyrillic/CJK/Telugu) → "..." / "[noise/silence]"
  c) Phonetic near-miss → apply MASTER DICT
  d) Translation cross-check vs corrected text
  e) Semantic guard: reject if Jaccard < 0.20 vs raw

DOMAIN GLOSSARY — never change: Nafis, Waseel, Clinicy, RCM, EMR, EHR, HIS, FHIR, HL7, ERP, API,
  radiology, laboratory, pharmacy, resubmission, revenue cycle, ICD-10, CPT, UAT, QA, DICOM,
  integration, mapping, configuration, module, workflow, approval, rejection, queue, escalation,
  notification, reconciliation, validation, synchronisation, NPHIES, MOH, CCHI, master data.

MASTER DICT: mirment/invirment→environment | 6sting→staging | decloy→deploy | sandbok→sandbox
  NCM/PCM→RCM | H3→HIS | Wallue→Waseel | Clinisey→Clinicy | Anshurns→insurance
  respond→resubmission | radios→radiology | nifis→Nafis | integrashion→integration
  konfig→configuration | approoval→approval | rejecshion→rejection
  "imagine the atmosphere"→يا ربي | "noses are ready"→النظام جاهز

PROTECT RULE: if original has a domain term and correction removes it → revert.
Return ONLY valid JSON array — no markdown, no preamble, CoT silent.
[{"idx":0,"corrected":"...","translation":"...","new_learnings":{}}]"""

    def get_system(self, tier: str) -> str:
        if tier == "full_cot":
            return self.SYSTEM_FULL_COT.format(version=self.VERSION)
        return self.SYSTEM_SIMPLE.format(version=self.VERSION)

    def get_system_with_context(self, tier: str, context: str, learnings: str) -> str:
        base = self.get_system(tier)
        if not context or context == "(session start)":
            return base
        return base + (
            f"\n\n=== LIVE MEETING CONTEXT ===\n{context}\n"
            f"=== CORRECTIONS LEARNED ===\n{learnings}\n"
            "Use context to resolve ambiguous domain terms."
        )

    def build_batch_msg(self, segments: List[Dict], context: str,
                        learnings: str, tier: str, text_key: str = "original") -> str:
        lines = [
            f"[{i}] Speaker {s['speaker']+1} [{s.get('timestamp','--')}]: "
            f"{s.get(text_key, s.get('transcript', ''))}"
            for i, s in enumerate(segments)
        ]
        body = "\n".join(lines)
        if tier == "full_cot":
            return (f"=== MEETING CONTEXT ===\n{context}\n\n"
                    f"=== CORRECTIONS LEARNED ===\n{learnings}\n\n"
                    f"=== SEGMENTS TO CORRECT ===\n{body}\n\nReturn JSON array.")
        if context and context != "(session start)":
            return f"=== RECENT CONTEXT ===\n{context}\n\n=== SEGMENTS ===\n{body}\nReturn JSON array."
        return f"Segments:\n{body}\nReturn JSON array."


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [8]  UTILITIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def detect_lang(text: str) -> str:
    if not text: return "en"
    n = max(len(text), 1)
    if sum(1 for c in text if '\u0600' <= c <= '\u06FF') / n > 0.15: return "ar"
    if sum(1 for c in text if '\u0900' <= c <= '\u097F') / n > 0.15: return "hi"
    return "en"


def extract_mean_pitch(pcm: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    FRAME_LEN = int(sr * 0.025); HOP = int(sr * 0.010)
    MIN_LAG = int(sr / 300);     MAX_LAG = int(sr / 60)
    pitches: List[float] = []
    for i in range(0, len(pcm) - FRAME_LEN, HOP):
        frame = pcm[i:i + FRAME_LEN].astype(np.float64) - np.mean(pcm[i:i + FRAME_LEN])
        if np.max(np.abs(frame)) < 50: continue
        corr = np.correlate(frame, frame, mode='full')[len(frame)-1:]
        if MAX_LAG < len(corr):
            peak_idx = MIN_LAG + int(np.argmax(corr[MIN_LAG:MAX_LAG]))
            f0 = sr / peak_idx
            if 60.0 <= f0 <= 300.0: pitches.append(f0)
    if not pitches: return 0.5
    return float(np.clip((float(np.median(pitches)) - 60.0) / 240.0, 0.0, 1.0))


def compute_word_diff(raw: str, corrected: str) -> List[Dict]:
    raw_w = raw.split(); cor_w = corrected.split()
    tokens: List[Dict] = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, raw_w, cor_w).get_opcodes():
        if op == "equal":
            tokens.extend({"type": "same",    "text": w} for w in raw_w[i1:i2])
        elif op == "replace":
            tokens.extend({"type": "removed", "text": w} for w in raw_w[i1:i2])
            tokens.extend({"type": "added",   "text": w} for w in cor_w[j1:j2])
        elif op == "delete":
            tokens.extend({"type": "removed", "text": w} for w in raw_w[i1:i2])
        elif op == "insert":
            tokens.extend({"type": "added",   "text": w} for w in cor_w[j1:j2])
    return tokens


def format_ts(secs: float) -> str:
    s = max(0, int(secs)); return f"{s // 60:02d}:{s % 60:02d}"


def _safe_parse_json(raw: str, fallback_keys: Optional[List[str]] = None) -> Any:
    cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    for suffix in ['"}]', '"}}]', '"}', '"}}']:
        try:
            return json.loads(cleaned + suffix)
        except json.JSONDecodeError:
            pass
    result: Dict[str, str] = {}
    if fallback_keys:
        for key in fallback_keys:
            m = re.search(rf'"{re.escape(key)}"\s*:\s*"((?:[^"\\]|\\.)*)', cleaned)
            if m:
                result[key] = m.group(1).replace('\\"', '"').replace('\\n', '\n')
    if result:
        return result
    raise json.JSONDecodeError(f"Cannot parse: {raw[:120]}", raw, 0)


def build_shared_bundle() -> Dict:
    return {
        "cache": SemanticCache(), "budget": TokenBudgetManager(),
        "learning": AutoLearningLoop(), "vad": EnhancedVAD(),
        "engine": PipelineDecisionEngine(), "validator": ValidationLayer(),
        "prompts": PromptTemplateManager(),
    }