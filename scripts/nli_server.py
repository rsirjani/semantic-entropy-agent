"""DeBERTa NLI Server — separate process to avoid memory issues in main pipeline.

Run with:
    python scripts/nli_server.py [--port 8100] [--model-path D:/models/deberta-large-mnli]

Endpoints:
    POST /classify          — NLI classification (premise, hypothesis)
    POST /classify_batch    — Batch NLI classification
    POST /sdlg_scores       — SDLG token attribution scores
    GET  /health            — Health check
"""

import argparse
import logging
import sys
import os

# Add project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from src.diversity.nli import NLIModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="DeBERTa NLI Server")
nli_model: NLIModel | None = None


class ClassifyRequest(BaseModel):
    premise: str
    hypothesis: str


class ClassifyBatchRequest(BaseModel):
    pairs: list[tuple[str, str]]


class SDLGRequest(BaseModel):
    text: str


@app.on_event("startup")
def load_model():
    global nli_model
    model_path = os.environ.get("NLI_MODEL_PATH", "D:/models/deberta-large-mnli")
    device = os.environ.get("NLI_DEVICE", "cpu")
    logger.info(f"Loading NLI model from {model_path} on {device}")
    nli_model = NLIModel(model_path=model_path, device=device)
    logger.info("NLI model loaded and ready")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": nli_model is not None}


@app.post("/classify")
def classify(req: ClassifyRequest):
    result = nli_model.classify(req.premise, req.hypothesis)
    return result


@app.post("/classify_batch")
def classify_batch(req: ClassifyBatchRequest):
    results = nli_model.classify_batch(req.pairs)
    return {"results": results}


@app.post("/sdlg_scores")
def sdlg_scores(req: SDLGRequest):
    scores = nli_model.compute_sdlg_scores(req.text)
    # Convert tensors to lists for JSON serialization
    return {
        "tokens": scores.get("tokens", []),
        "token_ids": scores.get("token_ids", []),
        "attributions": scores["attributions"].tolist() if torch.is_tensor(scores.get("attributions")) else [],
        "gradients": scores["gradients"].tolist() if torch.is_tensor(scores.get("gradients")) else [],
        "embeddings": scores["embeddings"].tolist() if torch.is_tensor(scores.get("embeddings")) else [],
        "word_starts": scores.get("word_starts", []),
    }


class SDLGRankRequest(BaseModel):
    text: str
    top_k: int = 20


@app.post("/sdlg_rank")
def sdlg_rank(req: SDLGRankRequest):
    """Full SDLG ranking: attribution + substitution scores, server-side.

    Returns ranked (position, replacement) candidates with scores.
    No embedding matrix transfer needed — everything computed here.
    """
    scores = nli_model.compute_sdlg_scores(req.text)
    if not scores["tokens"]:
        return {"candidates": []}

    tokens = scores["tokens"]
    token_ids = scores["token_ids"]
    attributions = scores["attributions"]  # [T]
    gradients = scores["gradients"]  # [T, D]
    embeddings = scores["embeddings"]  # [T, D]
    word_starts = scores["word_starts"]

    if not word_starts:
        return {"candidates": []}

    emb_matrix = nli_model.get_embedding_matrix()  # [V, D]

    # Normalize attributions
    attr_max = attributions.max()
    attr_norm = attributions / attr_max if attr_max > 0 else attributions

    candidates = []
    for i in word_starts:
        z_i = embeddings[i]
        grad_i = gradients[i]
        a_i = attr_norm[i].item()

        grad_norm = grad_i.norm()
        if grad_norm < 1e-8:
            continue

        # Substitution scores for all vocab tokens
        diff = z_i.unsqueeze(0) - emb_matrix
        diff_norms = diff.norm(dim=-1).clamp(min=1e-8)
        s_ij = (diff @ grad_i) / (diff_norms * grad_norm)
        s_ij[token_ids[i]] = -1.0  # Exclude self

        # Top-K by substitution score
        top_k = min(req.top_k, s_ij.shape[0])
        top_scores, top_indices = s_ij.topk(top_k)

        for rank in range(top_k):
            s_score = top_scores[rank].item()
            if s_score <= 0:
                break
            j = top_indices[rank].item()
            replacement = nli_model.tokenizer.convert_ids_to_tokens([j])[0]

            # Skip trivial substitutions
            orig_clean = tokens[i].replace("Ġ", "").replace("▁", "").lower()
            alt_clean = replacement.replace("Ġ", "").replace("▁", "").lower()
            if orig_clean == alt_clean:
                continue

            candidates.append({
                "position": i,
                "token": tokens[i],
                "token_id": token_ids[i],
                "replacement_id": j,
                "replacement": replacement,
                "attribution": round(a_i, 4),
                "substitution": round(s_score, 4),
            })

    # Sort by attribution + substitution combined
    candidates.sort(key=lambda c: c["attribution"] + c["substitution"], reverse=True)

    # Deduplicate: best replacement per position
    seen = set()
    deduped = []
    for c in candidates:
        if c["position"] not in seen:
            seen.add(c["position"])
            deduped.append(c)

    return {"candidates": deduped[:50]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeBERTa NLI Server")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--model-path", default="D:/models/deberta-large-mnli")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    os.environ["NLI_MODEL_PATH"] = args.model_path
    os.environ["NLI_DEVICE"] = args.device

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
