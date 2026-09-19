"""A spaCy NLP engine that never reaches the network.

Presidio needs an ``NlpEngine`` to tokenize, but we do not want its NER: tier 1
is regex plus validators, tier 2 is our own ONNX recognizer, tier 3 is GLiNER2.
The stock ``SpacyNlpEngine.load()`` calls ``spacy.cli.download()`` for any model
that is not already installed, which in an air-gapped government deployment is
not a slow path -- it is a crash on startup, at the worst possible moment.

So this engine builds ``spacy.blank(lang)`` pipelines. Those ship inside the
spaCy wheel: a tokenizer and nothing else. No weights, no download, no GPU, a
few milliseconds to construct.

One consequence worth stating plainly, because it shapes the design elsewhere:
a blank pipeline has no lemmatizer, so ``token.lemma_`` is empty, so Presidio's
``LemmaContextAwareEnhancer`` cannot boost anything. That is fine, because it
was never going to work for us anyway -- our context terms are Arabic
("الرقم القومي"), and an English lemmatizer has nothing useful to say about
them. Context boosting is done inside our own recognizers instead, against
GAZETTEER-normalized text. See ``detect/context.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import spacy
from presidio_analyzer.nlp_engine import SpacyNlpEngine

__all__ = ["DEFAULT_LANGUAGES", "BlankSpacyNlpEngine"]

DEFAULT_LANGUAGES: Final[tuple[str, ...]] = ("en", "ar")


class BlankSpacyNlpEngine(SpacyNlpEngine):
    """Tokenizer-only spaCy engine, constructed offline."""

    engine_name = "blank_spacy"

    def __init__(self, languages: Sequence[str] = DEFAULT_LANGUAGES) -> None:
        if not languages:
            raise ValueError("at least one language is required")
        self._languages: Final = tuple(languages)
        super().__init__(
            models=[{"lang_code": lang, "model_name": f"blank:{lang}"} for lang in self._languages]
        )

    def load(self) -> None:
        """Build blank pipelines. Deliberately does not call the base implementation."""
        # The base class annotates `nlp` as None until loaded; this is the load.
        self.nlp = {lang: spacy.blank(lang) for lang in self._languages}  # type: ignore[assignment]

    def get_supported_entities(self) -> list[str]:
        """No NER in a blank pipeline. Every entity comes from a recognizer."""
        return []
