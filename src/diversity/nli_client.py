"""NLI client — talks to the DeBERTa NLI server over HTTP.

Drop-in replacement for NLIModel that keeps DeBERTa in a separate process.
This avoids memory issues when running alongside vLLM + Docker containers.
"""

import logging
from typing import Any

import requests
import torch

logger = logging.getLogger(__name__)


class NLIClient:
    """HTTP client for the DeBERTa NLI server. Same interface as NLIModel."""

    def __init__(self, server_url: str = "http://localhost:8100"):
        self.server_url = server_url.rstrip("/")
        # Verify connection
        try:
            r = requests.get(f"{self.server_url}/health", timeout=5)
            r.raise_for_status()
            logger.info(f"Connected to NLI server at {self.server_url}")
        except Exception as e:
            raise ConnectionError(f"Cannot connect to NLI server at {self.server_url}: {e}")

    def classify(self, premise: str, hypothesis: str) -> dict[str, float]:
        """Classify NLI relationship between premise and hypothesis."""
        r = requests.post(
            f"{self.server_url}/classify",
            json={"premise": premise, "hypothesis": hypothesis},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def entails(self, premise: str, hypothesis: str, threshold: float = 0.5) -> bool:
        result = self.classify(premise, hypothesis)
        return result["entailment"] > threshold

    def bidirectional_entailment(
        self, text_a: str, text_b: str, threshold: float = 0.5
    ) -> bool:
        return self.entails(text_a, text_b, threshold) and self.entails(
            text_b, text_a, threshold
        )

    def classify_batch(
        self, pairs: list[tuple[str, str]]
    ) -> list[dict[str, float]]:
        if not pairs:
            return []
        r = requests.post(
            f"{self.server_url}/classify_batch",
            json={"pairs": pairs},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["results"]

    def compute_sdlg_scores(self, text: str) -> dict[str, Any]:
        """Get SDLG token attribution scores from the server."""
        r = requests.post(
            f"{self.server_url}/sdlg_scores",
            json={"text": text},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        # Convert lists back to tensors
        return {
            "tokens": data.get("tokens", []),
            "token_ids": data.get("token_ids", []),
            "attributions": torch.tensor(data["attributions"]) if data.get("attributions") else torch.tensor([]),
            "gradients": torch.tensor(data["gradients"]) if data.get("gradients") else torch.tensor([]),
            "embeddings": torch.tensor(data["embeddings"]) if data.get("embeddings") else torch.tensor([]),
            "word_starts": data.get("word_starts", []),
        }

    def get_embedding_matrix(self) -> torch.Tensor:
        """Not available via HTTP — SDLG substitution scoring runs server-side."""
        raise NotImplementedError(
            "get_embedding_matrix() not available via NLI client. "
            "SDLG substitution scoring should run on the server."
        )
