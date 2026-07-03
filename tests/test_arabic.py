"""Tests for Arabic instruction foundations (Roadmap Thrust B / Phase 0.1)."""

import json
import torch
import pytest

from fastvla.data.arabic import (
    ArabicInstructionTranslator,
    tokenizer_fertility,
    LIBERO_AR_LEXICON,
)


class TestLexiconTranslator:
    def test_translates_known_words(self):
        tr = ArabicInstructionTranslator(backend="lexicon")
        out = tr.translate("pick up the bowl")
        # "pick up" and "bowl" are in the seed lexicon -> Arabic script present.
        assert any("؀" <= ch <= "ۿ" for ch in out), out
        assert out != "pick up the bowl"

    def test_caches_repeated_calls(self):
        tr = ArabicInstructionTranslator(backend="lexicon")
        first = tr.translate("push the block")
        assert tr.translate("push the block") == first
        assert "push the block" in tr._cache

    def test_empty_passthrough(self):
        tr = ArabicInstructionTranslator(backend="lexicon")
        assert tr.translate("") == ""


class TestDictTranslator:
    def test_uses_validated_mapping(self, tmp_path):
        mapping = {"pick up the red block": "التقط المكعب الأحمر"}
        p = tmp_path / "map.json"
        p.write_text(json.dumps(mapping, ensure_ascii=False), encoding="utf-8")
        tr = ArabicInstructionTranslator(backend="dict", mapping_path=str(p))
        assert tr.translate("Pick up the red block") == "التقط المكعب الأحمر"

    def test_unmapped_returns_source(self, tmp_path):
        p = tmp_path / "map.json"
        p.write_text("{}", encoding="utf-8")
        tr = ArabicInstructionTranslator(backend="dict", mapping_path=str(p))
        # Never fabricate: unknown instructions stay in English.
        assert tr.translate("unknown instruction") == "unknown instruction"

    def test_dialect_without_mapping_falls_back_to_msa(self):
        tr = ArabicInstructionTranslator(backend="lexicon", register="dialect")
        assert tr.register == "msa"


class TestCorpusBuild:
    def test_build_corpus_writes_json(self, tmp_path):
        tr = ArabicInstructionTranslator(backend="lexicon")
        out = tmp_path / "corpus.json"
        mapping = tr.build_corpus(
            ["pick up the bowl", "push the block", "pick up the bowl"],
            str(out),
        )
        assert out.exists()
        assert len(mapping) == 2  # deduplicated
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert set(loaded) == {"pick up the bowl", "push the block"}


class TestTokenizerFertility:
    def test_fertility_metric(self):
        class FakeTok:
            # Fragments every word into 2 tokens.
            def encode(self, text, add_special_tokens=False):
                return list(range(2 * len(text.split())))

        stats = tokenizer_fertility(FakeTok(), ["one two three", "four five"])
        assert stats["words"] == 5
        assert stats["tokens"] == 10
        assert stats["fertility"] == pytest.approx(2.0)

    def test_ignores_empty(self):
        class FakeTok:
            def encode(self, text, add_special_tokens=False):
                return [0]

        stats = tokenizer_fertility(FakeTok(), ["", "  "])
        assert stats["fertility"] == 0.0


def test_lexicon_covers_core_vocab():
    for word in ["pick", "place", "push", "bowl", "plate", "drawer"]:
        assert word in LIBERO_AR_LEXICON
