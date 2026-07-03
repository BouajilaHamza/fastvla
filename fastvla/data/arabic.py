"""
Arabic instruction grounding — Roadmap Thrust B / Phase 0.1
===========================================================

Foundations for LIBERO-AR: turning English LIBERO instructions into Arabic
(Modern Standard Arabic + dialect) and quantifying the tokenization penalty that
motivates Hypothesis H5 (the English->Arabic grounding gap).

Nothing here claims to be the *validated* LIBERO-AR corpus — the roadmap is
explicit that human validation is not optional. This module provides:

  * ``tokenizer_fertility``   — tokens/word, the H5 metric (Arabic fragments more
                                than English in English-centric tokenizers, which
                                both inflates sequence length and hurts grounding).
  * ``ArabicInstructionTranslator`` — pluggable EN->AR translation with three
                                backends: a validated JSON mapping (preferred), a
                                neural MT model (NLLB), or a lexicon-substitution
                                stub for smoke tests.
  * ``LIBERO_AR_LEXICON``     — a seed EN->AR (MSA) lexicon of the verbs, objects
                                and prepositions that dominate LIBERO instructions.

The output feeds ``LIBEROArabicDataset`` (see ``datasets.py``).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


# ── Tokenizer fertility (H5 metric) ──────────────────────────────────────────
def tokenizer_fertility(tokenizer, texts: Iterable[str]) -> Dict[str, float]:
    """Mean tokens-per-word for a set of texts under ``tokenizer``.

    Fertility > 1 means the tokenizer fragments words into multiple sub-tokens.
    Arabic typically fertilizes 2-4x higher than English in English-centric
    vocabularies; that penalty (a) lengthens the LLM sequence and (b) scatters a
    word's meaning across sub-tokens, which is a plausible driver of the
    grounding gap. Reported per language in Phase 0.1.

    Returns a dict with ``tokens``, ``words``, and ``fertility``.
    """
    total_tokens = 0
    total_words = 0
    for text in texts:
        if not text:
            continue
        words = [w for w in re.split(r"\s+", text.strip()) if w]
        if not words:
            continue
        # ``add_special_tokens=False`` to measure content, not BOS/EOS overhead.
        try:
            ids = tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            ids = tokenizer.encode(text)
        total_tokens += len(ids)
        total_words += len(words)
    fertility = (total_tokens / total_words) if total_words else 0.0
    return {
        "tokens": total_tokens,
        "words": total_words,
        "fertility": round(fertility, 4),
    }


# ── Seed lexicon (MSA) — smoke-test fallback, NOT a validated corpus ─────────
# Covers the high-frequency LIBERO instruction vocabulary. Human validation and
# full-sentence naturalness are Phase 0.1 deliverables; this exists so the
# pipeline is runnable end-to-end before the validated corpus lands.
LIBERO_AR_LEXICON: Dict[str, str] = {
    # verbs
    "pick": "التقط", "pick up": "التقط", "place": "ضع", "put": "ضع",
    "push": "ادفع", "pull": "اسحب", "open": "افتح", "close": "أغلق",
    "turn": "أدر", "move": "حرّك", "grasp": "أمسك", "lift": "ارفع",
    "stack": "كدّس", "insert": "أدخل", "press": "اضغط",
    # objects
    "bowl": "الوعاء", "plate": "الطبق", "cup": "الكوب", "mug": "القدح",
    "bottle": "الزجاجة", "box": "الصندوق", "drawer": "الدرج", "cabinet": "الخزانة",
    "block": "المكعب", "button": "الزر", "stove": "الموقد", "pot": "القدر",
    "ramekin": "الوعاء الصغير", "book": "الكتاب", "tray": "الصينية",
    # prepositions / connectors
    "the": "", "on": "على", "in": "في", "into": "في", "to": "إلى",
    "between": "بين", "and": "و", "of": "من", "onto": "على",
    "left": "اليسار", "right": "اليمين", "top": "الأعلى", "bottom": "الأسفل",
}


# ── Translator ───────────────────────────────────────────────────────────────
class ArabicInstructionTranslator:
    """Translate English LIBERO instructions to Arabic.

    Backends (chosen by ``backend``):
      * ``"dict"``    — exact-match lookup in a validated ``{en: ar}`` JSON map.
                        The production path once LIBERO-AR is human-validated.
      * ``"nllb"``    — neural MT via ``facebook/nllb-200-distilled-600M``.
                        Good for bootstrapping the corpus for human review.
      * ``"lexicon"`` — word-by-word substitution using ``LIBERO_AR_LEXICON``.
                        A deterministic, dependency-free stub for smoke tests.

    ``register`` selects ``"msa"`` (default) or ``"dialect"``. Dialect requires a
    dialect mapping JSON — without one it falls back to MSA and warns, so results
    never silently pretend to be dialectal.
    """

    def __init__(
        self,
        backend: str = "lexicon",
        register: str = "msa",
        mapping_path: Optional[str] = None,
        nllb_model: str = "facebook/nllb-200-distilled-600M",
        hf_token: Optional[str] = None,
    ):
        self.backend = backend
        self.register = register
        self.nllb_model = nllb_model
        self.hf_token = hf_token
        self._mapping: Dict[str, str] = {}
        self._mt = None  # lazy NLLB pipeline
        self._cache: Dict[str, str] = {}

        if mapping_path:
            self._mapping = self._load_mapping(mapping_path)
        if register == "dialect" and not self._mapping:
            logger.warning(
                "register='dialect' needs a dialect mapping JSON; none provided. "
                "Falling back to MSA — do not report these as dialectal results."
            )
            self.register = "msa"

    @staticmethod
    def _load_mapping(path: str) -> Dict[str, str]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Arabic mapping file not found: {path}")
        with p.open(encoding="utf-8") as f:
            raw = json.load(f)
        # Normalize keys for robust exact-match lookup.
        return {ArabicInstructionTranslator._norm(k): v for k, v in raw.items()}

    @staticmethod
    def _norm(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def translate(self, text: str) -> str:
        if not text:
            return text
        if text in self._cache:
            return self._cache[text]

        if self.backend == "dict":
            out = self._mapping.get(self._norm(text))
            if out is None:
                logger.debug(f"No validated Arabic mapping for: {text!r}")
                out = text  # leave untranslated rather than fabricate
        elif self.backend == "nllb":
            out = self._translate_nllb(text)
        elif self.backend == "lexicon":
            out = self._translate_lexicon(text)
        else:
            raise ValueError(f"Unknown translation backend: {self.backend!r}")

        self._cache[text] = out
        return out

    def translate_many(self, texts: List[str]) -> List[str]:
        return [self.translate(t) for t in texts]

    def _translate_lexicon(self, text: str) -> str:
        # Handle two-word verbs first ("pick up"), then single tokens.
        norm = self._norm(text)
        for phrase in ("pick up",):
            norm = norm.replace(phrase, LIBERO_AR_LEXICON.get(phrase, phrase))
        words = re.split(r"(\s+)", norm)  # keep separators
        out = []
        for w in words:
            key = re.sub(r"[^\w]", "", w)
            if key in LIBERO_AR_LEXICON:
                out.append(LIBERO_AR_LEXICON[key])
            else:
                out.append(w)
        return re.sub(r"\s+", " ", "".join(out)).strip()

    def _translate_nllb(self, text: str) -> str:
        if self._mt is None:
            try:
                from transformers import pipeline

                self._mt = pipeline(
                    "translation",
                    model=self.nllb_model,
                    src_lang="eng_Latn",
                    tgt_lang="arb_Arab",
                    token=self.hf_token,
                )
            except Exception as e:  # pragma: no cover - needs network/model
                logger.error(f"NLLB translation unavailable ({e}); returning source.")
                return text
        try:  # pragma: no cover - needs model weights
            return self._mt(text, max_length=128)[0]["translation_text"]
        except Exception as e:
            logger.error(f"NLLB translation failed for {text!r}: {e}")
            return text

    def build_corpus(self, texts: Iterable[str], out_path: str) -> Dict[str, str]:
        """Translate a set of instructions and write a ``{en: ar}`` JSON map.

        This is how you bootstrap LIBERO-AR: run with ``backend='nllb'`` over the
        unique LIBERO instructions, then hand the JSON to a human validator, then
        load it back with ``backend='dict'`` for the actual experiments.
        """
        uniq = sorted({t for t in texts if t})
        mapping = {t: self.translate(t) for t in uniq}
        Path(out_path).write_text(
            json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(f"Wrote {len(mapping)} EN->AR pairs to {out_path} for validation.")
        return mapping
