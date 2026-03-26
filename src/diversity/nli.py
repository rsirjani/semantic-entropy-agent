"""DeBERTa-large-MNLI wrapper for NLI classification and gradient attribution."""

import logging
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

# DeBERTa-large-MNLI label mapping: 0=contradiction, 1=neutral, 2=entailment
LABEL_NAMES = ["contradiction", "neutral", "entailment"]


class NLIModel:
    """DeBERTa-large-MNLI wrapper for entailment classification and gradient attribution.

    Used for:
    1. Bidirectional entailment checking (semantic clustering)
    2. Gradient-based token attribution (SDLG, Phase 2b)
    """

    def __init__(
        self,
        model_path: str = "D:/models/deberta-large-mnli",
        device: str = "cpu",
    ):
        logger.info(f"Loading NLI model from {model_path} on {device}")
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        logger.info("NLI model loaded")

    def classify(self, premise: str, hypothesis: str) -> dict[str, float]:
        """Classify the NLI relationship between premise and hypothesis.

        Returns dict with probabilities for entailment, neutral, contradiction.
        """
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = F.softmax(logits[0], dim=-1).cpu().tolist()
        return {name: prob for name, prob in zip(LABEL_NAMES, probs)}

    def entails(self, premise: str, hypothesis: str, threshold: float = 0.5) -> bool:
        """Check if premise entails hypothesis (entailment probability > threshold)."""
        result = self.classify(premise, hypothesis)
        return result["entailment"] > threshold

    def bidirectional_entailment(
        self, text_a: str, text_b: str, threshold: float = 0.5
    ) -> bool:
        """Check if A entails B AND B entails A.

        Used for semantic clustering: two texts are in the same semantic class
        if and only if they bidirectionally entail each other.
        """
        return self.entails(text_a, text_b, threshold) and self.entails(
            text_b, text_a, threshold
        )

    def classify_batch(
        self, pairs: list[tuple[str, str]]
    ) -> list[dict[str, float]]:
        """Classify multiple premise-hypothesis pairs efficiently.

        Batches all pairs into a single forward pass for speed.
        """
        if not pairs:
            return []

        premises, hypotheses = zip(*pairs)
        inputs = self.tokenizer(
            list(premises),
            list(hypotheses),
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = F.softmax(logits, dim=-1).cpu().tolist()
        return [
            {name: prob for name, prob in zip(LABEL_NAMES, row_probs)}
            for row_probs in probs
        ]

    def compute_sdlg_scores(self, text: str) -> dict[str, Any]:
        """Compute SDLG token scores for a given text (Algorithm 2 from Aichberger et al. 2025).

        Feeds the text as both premise and hypothesis (self-entailment),
        computes loss toward "contradiction", and backprops to get gradients.

        Returns dict with:
            tokens: list of token strings
            token_ids: list of token IDs
            attributions: tensor [T] — A_i = ||z_i ⊙ ∇z_i L||_2
            gradients: tensor [T, D] — ∇z_i L for each token
            embeddings: tensor [T, D] — z_i for each token
            word_starts: list of int — indices of word-initial tokens
        """
        # Per paper: feed text as BOTH premise and hypothesis (self-entailment)
        inputs = self.tokenizer(
            text, text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        input_ids = inputs["input_ids"][0]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.tolist())

        # Find hypothesis range (after the separator tokens between premise & hypothesis)
        # DeBERTa uses [CLS] premise [SEP][SEP] hypothesis [SEP]
        sep_positions = [
            i for i, t in enumerate(tokens) if t == self.tokenizer.sep_token
        ]
        if len(sep_positions) < 2:
            return {"tokens": [], "attributions": torch.tensor([])}
        # Hypothesis starts after the consecutive SEPs, ends before the last SEP
        # Find the first non-SEP token after the first SEP
        hyp_start = sep_positions[0] + 1
        while hyp_start < len(tokens) and tokens[hyp_start] == self.tokenizer.sep_token:
            hyp_start += 1
        hyp_end = sep_positions[-1]  # Last SEP is the end boundary

        # Forward pass with gradient tracking on embeddings
        self.model.eval()
        embedding_layer = self.model.deberta.embeddings.word_embeddings
        embeddings = embedding_layer(inputs["input_ids"])
        embeddings.retain_grad()
        embeddings.requires_grad_(True)

        outputs = self.model.deberta(
            inputs_embeds=embeddings,
            attention_mask=inputs.get("attention_mask"),
            token_type_ids=inputs.get("token_type_ids"),
        )
        # ContextPooler expects full hidden states [batch, seq_len, D]
        encoder_output = outputs[0]
        pooled = self.model.pooler(encoder_output)
        pooled = self.model.dropout(pooled)
        logits = self.model.classifier(pooled)

        # Loss targeting contradiction (class 0)
        target = torch.tensor([0], device=self.device)
        loss = F.cross_entropy(logits, target)
        loss.backward()

        grad = embeddings.grad[0]  # [seq_len, D]
        emb = embeddings.detach()[0]  # [seq_len, D]

        # Extract hypothesis portion only
        hyp_tokens = tokens[hyp_start:hyp_end]
        hyp_ids = input_ids[hyp_start:hyp_end].tolist()
        hyp_grad = grad[hyp_start:hyp_end].detach()  # [T, D]
        hyp_emb = emb[hyp_start:hyp_end]  # [T, D]

        # Attribution scores: A_i = ||z_i ⊙ ∇z_i L||_2
        attributions = (hyp_emb * hyp_grad).norm(dim=-1)  # [T]

        # Identify word-initial tokens (per paper: only substitute at word boundaries)
        word_starts = []
        for i, tok in enumerate(hyp_tokens):
            # DeBERTa uses Ġ prefix for word-initial tokens (GPT-style BPE)
            # Also treat the first token as word-initial
            if i == 0 or tok.startswith("Ġ") or tok.startswith("▁"):
                word_starts.append(i)

        return {
            "tokens": hyp_tokens,
            "token_ids": hyp_ids,
            "attributions": attributions,
            "gradients": hyp_grad,
            "embeddings": hyp_emb,
            "word_starts": word_starts,
        }

    def get_embedding_matrix(self) -> torch.Tensor:
        """Return the full word embedding matrix [vocab_size, D]."""
        return self.model.deberta.embeddings.word_embeddings.weight.detach()
