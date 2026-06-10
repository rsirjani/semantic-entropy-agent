"""R1.1 — SDLG importance scoring with text-level vocabulary unification.

Aichberger 2025 (App. D) relies on the generator and NLI model sharing a
vocabulary; Qwen3 BPE and DeBERTa do not align. The fix bridges at the TEXT
level: each DeBERTa-proposed substitute is scored as p_LLM(v_j | prefix) under
the generator's own tokenization — (1) surface-text match against the
generator's top-k next-token strings, (2) exact echo-scored prompt logprobs for
misses. These tests prove, with a mocked HTTP layer (no vLLM), that I_ij is
per-substitute — the old hash-id matching could never match a DeBERTa
replacement_id and silently degraded to a position-level constant.
"""

import math

import pytest

from src.diversity.sdlg import SDLGGenerator


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #

def test_normalize_token_text_strips_word_markers():
    assert SDLGGenerator._normalize_token_text("Ġlexer") == "lexer"
    assert SDLGGenerator._normalize_token_text("▁Fix", keep_case=True) == "Fix"
    assert SDLGGenerator._normalize_token_text("▁Fix") == "fix"
    assert SDLGGenerator._normalize_token_text(" parser ") == "parser"
    assert SDLGGenerator._normalize_token_text("") == ""


def test_match_substitute_probability_matches_surface_text():
    top_logprobs = {" lexer": math.log(0.25), " parser": math.log(0.5)}
    # DeBERTa-style token matches the generator's OpenAI-style " tok" string.
    p = SDLGGenerator.match_substitute_probability(top_logprobs, "Ġlexer")
    assert p == pytest.approx(0.25)
    # Case-insensitive surface match.
    p = SDLGGenerator.match_substitute_probability(top_logprobs, "Parser")
    assert p == pytest.approx(0.5)


def test_match_substitute_probability_misses_return_none():
    top_logprobs = {" lexer": math.log(0.25)}
    assert SDLGGenerator.match_substitute_probability(top_logprobs, "tokenizer") is None
    assert SDLGGenerator.match_substitute_probability({}, "lexer") is None
    assert SDLGGenerator.match_substitute_probability(top_logprobs, "") is None


# --------------------------------------------------------------------------- #
# Mocked HTTP layer
# --------------------------------------------------------------------------- #

class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


def _fake_post_factory(top_logprobs, echo_offsets=None, echo_logprobs=None):
    """requests.post stand-in serving both the top-k and the echo call."""

    def fake_post(url, json=None, timeout=None):
        if json.get("echo"):
            return _FakeResponse({
                "choices": [{"logprobs": {
                    "text_offset": echo_offsets or [],
                    "token_logprobs": echo_logprobs or [],
                }}]
            })
        return _FakeResponse({
            "choices": [{"logprobs": {"top_logprobs": [top_logprobs]}}]
        })

    return fake_post


def test_echo_score_sums_continuation_token_logprobs(monkeypatch):
    import requests

    prefix = "fix the "  # len 8 — tokens at offsets >= 8 belong to the continuation
    monkeypatch.setattr(requests, "post", _fake_post_factory(
        {}, echo_offsets=[0, 4, 8, 13], echo_logprobs=[None, -0.1, -0.2, -0.3],
    ))
    p = SDLGGenerator._echo_score_continuation("http://x", "m", prefix, "lexer bug")
    assert p == pytest.approx(math.exp(-0.5))


def test_echo_score_returns_none_on_failure(monkeypatch):
    import requests

    def boom(url, json=None, timeout=None):
        raise ConnectionError("no server")

    monkeypatch.setattr(requests, "post", boom)
    assert SDLGGenerator._echo_score_continuation("http://x", "m", "p", "c") is None


def test_importance_scores_are_per_substitute(monkeypatch):
    """The regression the old hash-matching failed: two substitutes at the SAME
    position must get DIFFERENT probabilities (top-k hit vs echo-scored miss)."""
    import requests

    top_logprobs = {" lexer": math.log(0.25), " the": math.log(0.4)}
    monkeypatch.setattr(requests, "post", _fake_post_factory(
        top_logprobs,
        # echo call: prefix "fix the" (len 7) + " tokenizer" → one token at off 7
        echo_offsets=[0, 7], echo_logprobs=[None, -1.0],
    ))

    gen = SDLGGenerator(nli_model=object(), n_candidates=3)
    candidates = [
        {"position": 3, "token": "Ġparser", "replacement": "Ġlexer",
         "replacement_id": 7},
        {"position": 3, "token": "Ġparser", "replacement": "Ġtokenizer",
         "replacement_id": 9},
    ]
    scores = gen._get_importance_scores(
        "fix the parser bug now", candidates, "openai/qwen", {}, [],
    )

    assert scores[(3, 7)] == pytest.approx(0.25)          # surface match in top-k
    assert scores[(3, 9)] == pytest.approx(math.exp(-1.0))  # exact echo fallback
    assert scores[(3, 7)] != scores[(3, 9)]


def test_importance_scores_zero_when_generator_unreachable(monkeypatch):
    """Scoring failures mean negligible generator mass (0.0), never a borrowed
    position-level probability."""
    import requests

    def boom(url, json=None, timeout=None):
        raise ConnectionError("no server")

    monkeypatch.setattr(requests, "post", boom)
    gen = SDLGGenerator(nli_model=object(), n_candidates=3)
    scores = gen._get_importance_scores(
        "fix the parser bug",
        [{"position": 2, "token": "Ġparser", "replacement": "Ġlexer",
          "replacement_id": 5}],
        "openai/qwen", {}, [],
    )
    assert scores == {(2, 5): 0.0}
