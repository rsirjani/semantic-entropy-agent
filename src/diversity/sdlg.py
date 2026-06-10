"""SDLG: Semantically Diverse Language Generation (Aichberger et al. 2025).

Generates diverse yet likely alternative responses by:
1. Computing gradient-based attribution through DeBERTa NLI to find
   which tokens most impact the semantic meaning
2. Identifying substitute tokens that shift semantics (substitution score)
   while remaining likely under the LLM (importance score)
3. Replacing the highest-scored token and letting the LLM complete from there

Applied to the REASONING (THOUGHT) portion of agent responses. Per the SDLG
paper, the NLI model's gradients are meaningful on natural language — not on
bash commands where token semantics are arbitrary. Diversifying the reasoning
(e.g., "remove the duplicate check" → "compose cycles before validation")
forces the LLM to regenerate both the reasoning chain and code action,
producing genuinely different fix approaches rather than syntactic variants
of the same command.
"""

import logging
import re
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import litellm

from src.diversity.nli import NLIModel

logger = logging.getLogger(__name__)


def extract_code_block(response: str) -> tuple[str, str, str]:
    """Split an agent response into (preamble, code, postamble).

    The code block is delimited by ```mswea_bash_command ... ```.
    Returns (text_before_code_block, code_inside_block, text_after_code_block).
    If no code block is found, returns (response, "", "").
    """
    pattern = r"(```mswea_bash_command\n)(.*?)(```)"
    match = re.search(pattern, response, re.DOTALL)
    if not match:
        return response, "", ""
    start, end = match.span()
    preamble = response[:start] + match.group(1)  # everything up to and including ```mswea_bash_command\n
    code = match.group(2)  # the actual command(s)
    postamble = match.group(3) + response[end:]   # closing ``` and anything after
    return preamble, code.strip(), postamble


def extract_thought_text(response: str) -> tuple[str, str]:
    """Split an agent response into (thought_text, rest_of_response).

    The thought is everything before the code block (```mswea_bash_command).
    Returns (thought_text, everything_from_code_block_onward).
    If no code block is found, returns (response, "").
    """
    pattern = r"```mswea_bash_command\n"
    match = re.search(pattern, response)
    if not match:
        return response, ""
    return response[:match.start()].rstrip(), response[match.start():]


@dataclass
class SubstitutionCandidate:
    """A candidate (position, substitute) pair for SDLG."""
    position: int           # Token index in the THOUGHT portion
    original_token: str
    substitute_token: str
    substitute_id: int
    attribution: float      # A_i
    substitution: float     # S_ij
    importance: float       # I_ij
    combined_score: float   # Average of normalized scores


class SDLGGenerator:
    """Generate diverse candidates via SDLG token substitution.

    Algorithm (from Aichberger et al. 2025, Algorithms 1 & 2):
    1. Get greedy response y¹
    2. Compute token scores via DeBERTa gradient attribution
    3. Rank all (position, substitute) pairs by combined score
    4. For each alternative: substitute the n-th ranked token,
       let the LLM complete from the substitution point
    """

    def __init__(
        self,
        nli_model: NLIModel,
        n_candidates: int = 5,
        top_k_substitutes: int = 20,
        importance_threshold: float = 0.001,
        diversify_code: bool = False,
    ):
        self.nli = nli_model
        self.n_candidates = n_candidates
        self.top_k = top_k_substitutes
        self.importance_threshold = importance_threshold
        # Faithful SDLG (Aichberger 2025) substitutes tokens in the REASONING
        # only — the rubric (R1.1) and CLAUDE.md both require "substitutions
        # apply to reasoning, NOT action tokens", because NLI gradients are
        # meaningful on natural language but near-random on bash tokens. Code-
        # level substitution is therefore an explicit, OFF-by-default extension;
        # enable it only if you intend to claim/ablate it separately.
        self.diversify_code = diversify_code

        # Cache the NLI embedding matrix (used for substitution scores)
        self._emb_matrix = None

    @property
    def emb_matrix(self) -> torch.Tensor:
        if self._emb_matrix is None:
            try:
                self._emb_matrix = self.nli.get_embedding_matrix()
            except (NotImplementedError, AttributeError):
                # NLI client doesn't have embedding matrix — use server-side ranking
                self._emb_matrix = None
        return self._emb_matrix

    def generate(
        self,
        model_name: str,
        model_kwargs: dict,
        messages: list[dict],
        greedy_response: str,
    ) -> list[str]:
        """Generate N diverse candidates using SDLG on BOTH thought and code.

        Two axes of diversity:
        1. THOUGHT substitution — NLI gradients on natural language reasoning
           produce fundamentally different approaches (e.g., "remove the check"
           → "compose cycles before validation"). LLM regenerates everything
           from the substitution point: rest of reasoning + code block.
        2. CODE substitution — substitution within the bash command produces
           different implementations of the same approach.

        Half the alternatives come from thought-level SDLG, half from code-level.

        Args:
            model_name: litellm model name.
            model_kwargs: Model kwargs (api_base, api_key, etc.).
            messages: Conversation history BEFORE the greedy response.
            greedy_response: The greedy (temp=0) response.

        Returns:
            List of N response strings. First is always the greedy response.
        """
        candidates = [greedy_response]

        # Reasoning-only by default (R1.1 / CLAUDE.md: substitutions apply to
        # reasoning, NOT action tokens). All of the budget goes to thought-level
        # SDLG unless code-level substitution is explicitly enabled.
        if self.diversify_code:
            # Split: thought-level alternatives get the majority (more impactful),
            # code-level alternatives fill the rest
            n_thought = max(1, (self.n_candidates - 1 + 1) // 2)  # ceil half
            n_code = (self.n_candidates - 1) - n_thought
        else:
            n_thought = self.n_candidates - 1
            n_code = 0

        # --- THOUGHT-level SDLG ---
        thought_text, code_and_rest = extract_thought_text(greedy_response)
        if thought_text and len(thought_text.split()) >= 5:
            logger.info(f"SDLG THOUGHT: targeting reasoning ({len(thought_text)} chars): {thought_text[:100]}")
            try:
                thought_ranked = self._rank_substitutions(thought_text, model_name, model_kwargs, messages)
            except Exception as e:
                logger.warning(f"SDLG THOUGHT scoring failed: {e}")
                thought_ranked = []

            thought_alts = self._generate_alternatives_from_ranked(
                thought_ranked, n_thought, thought_text, greedy_response,
                model_name, model_kwargs, messages, target="thought",
            )
            candidates.extend(thought_alts)
            logger.info(f"SDLG THOUGHT: generated {len(thought_alts)} alternatives")
        else:
            logger.warning("SDLG: thought text too short for attribution, skipping thought-level")
            # Only redirect budget to code if code-level SDLG is enabled;
            # otherwise stay reasoning-only (no alternatives -> no fork).
            if self.diversify_code:
                n_code = self.n_candidates - 1  # All budget goes to code

        # --- CODE-level SDLG ---
        preamble, code_text, postamble = extract_code_block(greedy_response)
        if code_text and len(code_text) >= 5 and n_code > 0:
            logger.info(f"SDLG CODE: targeting action ({len(code_text)} chars): {code_text[:100]}")
            try:
                code_ranked = self._rank_substitutions(code_text, model_name, model_kwargs, messages)
            except Exception as e:
                logger.warning(f"SDLG CODE scoring failed: {e}")
                code_ranked = []

            code_alts = self._generate_alternatives_from_ranked(
                code_ranked, n_code, code_text, greedy_response,
                model_name, model_kwargs, messages, target="code",
                preamble=preamble, postamble=postamble,
            )
            candidates.extend(code_alts)
            logger.info(f"SDLG CODE: generated {len(code_alts)} alternatives")

        # R3.1 arm attributability: if SDLG produced no alternatives (too-short
        # reasoning, scoring failure), the arm must NOT silently switch to a
        # different diversity mechanism. The previous temperature-sampling
        # fallback (a) contaminated the sdlg-vs-strategy ablation — the
        # orchestrator records every fork as `sdlg_fork`, so temperature-sampled
        # branches were attributed to SDLG with no trace in the artifacts — and
        # (b) hardcoded T=0.7, breaking the R2.4 temperature match in the
        # T=0.2/1.0 sweep cells. No alternatives -> no fork: the instance stays
        # a single greedy trajectory and the realized-N reporting flags it.
        if len(candidates) <= 1:
            logger.warning(
                "SDLG: no alternatives from either target — no fork "
                "(deliberately no temperature fallback; the arm stays pure SDLG)"
            )
        return candidates

    def _generate_alternatives_from_ranked(
        self,
        ranked: list[SubstitutionCandidate],
        n_alts: int,
        target_text: str,
        greedy_response: str,
        model_name: str,
        model_kwargs: dict,
        messages: list[dict],
        target: str = "thought",
        preamble: str = "",
        postamble: str = "",
    ) -> list[str]:
        """Generate alternatives from ranked substitution candidates.

        Args:
            target: "thought" — substitute in reasoning, regenerate everything after.
                    "code" — substitute in code block, keep reasoning fixed.
        """
        alternatives = []
        if not ranked:
            return alternatives

        used_substitutions = set()  # (position, substitute_id) pairs
        for n in range(n_alts):
            # Find next unused (position, substitute) pair — allow multiple
            # substitutions at the same position since different replacements
            # at a high-attribution position produce genuinely different completions
            attempt = n
            while attempt < len(ranked):
                key = (ranked[attempt].position, ranked[attempt].substitute_id)
                if key not in used_substitutions:
                    break
                attempt += 1
            if attempt >= len(ranked):
                break
            sub = ranked[attempt]
            used_substitutions.add((sub.position, sub.substitute_id))

            try:
                if target == "thought":
                    alt = self._generate_thought_alternative(
                        sub, target_text, greedy_response,
                        model_name, model_kwargs, messages,
                    )
                else:
                    alt = self._generate_code_alternative(
                        sub, target_text, preamble, postamble,
                        greedy_response, model_name, model_kwargs, messages,
                    )

                if alt != greedy_response:
                    alternatives.append(alt)
                    logger.info(
                        f"SDLG {target} alt {n+1}: '{sub.original_token}' → "
                        f"'{sub.substitute_token}' at pos {sub.position} "
                        f"(A={sub.attribution:.3f}, S={sub.substitution:.3f}, I={sub.importance:.3f})"
                    )
                else:
                    logger.debug(f"SDLG {target} alt {n+1}: identical to greedy, skipping")
            except Exception as e:
                logger.warning(f"SDLG {target} completion failed for alt {n+1}: {e}")

        return alternatives

    def _rank_substitutions(
        self,
        target_text: str,
        model_name: str,
        model_kwargs: dict,
        messages: list[dict],
    ) -> list[SubstitutionCandidate]:
        """Algorithm 2: Token Score Ranking.

        Uses the NLI server's /sdlg_rank endpoint for attribution + substitution
        scoring (runs server-side to avoid transferring the embedding matrix).
        Then combines with LLM importance scores locally.
        """
        import requests

        # Step 1: Get server-side ranking (attribution + substitution)
        try:
            server_url = getattr(self.nli, 'server_url', None)
            if server_url:
                # Using NLI client — call server endpoint
                r = requests.post(
                    f"{server_url}/sdlg_rank",
                    json={"text": target_text, "top_k": self.top_k},
                    timeout=30,
                )
                r.raise_for_status()
                server_candidates = r.json()["candidates"]
            else:
                # Using local NLI model — fall back to local computation
                scores = self.nli.compute_sdlg_scores(target_text)
                if not scores["tokens"]:
                    return []
                # Simplified local ranking (attribution only, no substitution)
                tokens = scores["tokens"]
                attributions = scores["attributions"]
                word_starts = scores["word_starts"]
                server_candidates = []
                for i in word_starts:
                    server_candidates.append({
                        "position": i,
                        "token": tokens[i],
                        "token_id": scores["token_ids"][i],
                        "replacement_id": 0,
                        "replacement": "",
                        "attribution": attributions[i].item(),
                        "substitution": 0.0,
                    })
        except Exception as e:
            logger.warning(f"SDLG ranking failed: {e}")
            return []

        if not server_candidates:
            return []

        # Step 2: Importance scores from the GENERATOR, per candidate.
        # Vocabulary unification is TEXT-LEVEL (see _get_importance_scores):
        # each NLI-vocabulary substitute is scored as p_LLM(v_j | prefix) under
        # the generator's own tokenization, so I_ij is specific to the
        # substitute, never a position-level stand-in.
        importance_scores = self._get_importance_scores(
            target_text, server_candidates[:self.top_k], model_name,
            model_kwargs, messages,
        )

        # Step 3: Combine server scores with importance scores
        all_candidates = []
        for c in server_candidates:
            pos = c["position"]
            A_i = c["attribution"]
            S_ij = c["substitution"]
            I_ij = importance_scores.get((pos, c.get("replacement_id", 0)), 0.0)

            # Documented deviation from Aichberger Alg. 2: scores are combined
            # by arithmetic mean rather than product. The mean keeps a candidate
            # rankable on A+S when the generator assigns it negligible mass
            # (I_ij ~ 0), where a product would zero the whole score.
            combined = (A_i + S_ij + I_ij) / 3.0

            all_candidates.append(SubstitutionCandidate(
                position=pos,
                original_token=c["token"],
                substitute_token=c.get("replacement", ""),
                substitute_id=c.get("replacement_id", 0),
                attribution=A_i,
                substitution=S_ij,
                importance=I_ij,
                combined_score=combined,
            ))

        all_candidates.sort(key=lambda c: c.combined_score, reverse=True)
        return all_candidates

    @staticmethod
    def _normalize_token_text(token: str, keep_case: bool = False) -> str:
        """Surface form of a tokenizer token (strips Ġ/▁ word-start markers)."""
        text = token.replace("Ġ", " ").replace("▁", " ").strip()
        return text if keep_case else text.lower()

    @staticmethod
    def match_substitute_probability(top_logprobs: dict, sub_text: str) -> float | None:
        """Generator probability of `sub_text` among its own top-k next tokens.

        Matching is on normalized surface text, so a substitute proposed in the
        NLI model's vocabulary is scored against the GENERATOR's own token
        strings — no embedding/vocabulary alignment between the two tokenizers
        is needed (cf. Aichberger 2025 App. D, which relied on the generator
        and NLI model sharing a vocabulary). Returns None when the substitute
        is not among the top-k (caller falls back to exact echo scoring).
        """
        import math

        target = SDLGGenerator._normalize_token_text(sub_text)
        if not target or not top_logprobs:
            return None
        for token_text, logprob in top_logprobs.items():
            if SDLGGenerator._normalize_token_text(token_text) == target:
                return math.exp(logprob)
        return None

    @staticmethod
    def _echo_score_continuation(
        base_url: str, model: str, prefix: str, continuation: str
    ) -> float | None:
        """Exact p(continuation | prefix) under the generator via prompt logprobs.

        Sends prefix+continuation with echo=true, max_tokens=0 and sums the
        logprobs of the tokens whose text offsets fall inside the continuation,
        so the substitute is scored under the generator's own tokenization even
        when it spans multiple generator tokens. Returns None if the endpoint
        does not support echo/prompt logprobs or the response is malformed.
        """
        import math
        import requests

        try:
            resp = requests.post(
                f"{base_url}/v1/completions",
                json={
                    "model": model,
                    "prompt": prefix + continuation,
                    "max_tokens": 0,
                    "echo": True,
                    "logprobs": 0,
                    "temperature": 0,
                },
                timeout=10,
            )
            lp = resp.json()["choices"][0]["logprobs"]
            offsets = lp.get("text_offset") or []
            token_logprobs = lp.get("token_logprobs") or []
            total, n = 0.0, 0
            for off, tlp in zip(offsets, token_logprobs):
                if off >= len(prefix) and tlp is not None:
                    total += tlp
                    n += 1
            return math.exp(total) if n else None
        except Exception:
            return None

    def _get_importance_scores(
        self,
        target_text: str,
        candidates: list[dict],
        model_name: str,
        model_kwargs: dict,
        messages: list[dict],
    ) -> dict[tuple[int, int], float]:
        """I_ij per (position, substitute): the GENERATOR's probability of the
        specific substitute, I_ij = p_LLM(v_j | y_<i).

        Vocabulary unification: Aichberger 2025 (App. D) relies on the
        generator and the NLI model sharing a vocabulary; Qwen3's ~151k BPE and
        DeBERTa's vocabulary do not align, so we bridge at the TEXT level:

          1. ONE top-k logprobs query at the text prefix preceding the original
             token; each candidate's substitute surface form is matched against
             the generator's own top-k token strings
             (match_substitute_probability);
          2. substitutes outside the top-k get an exact echo-scored query of
             prefix + substitute (_echo_score_continuation), reading the
             substitute's prompt logprobs under the generator's tokenization.

        Scoring failures yield 0.0 — negligible generator mass — which
        correctly down-ranks substitutes the generator would not produce
        (instead of the previous behaviour of borrowing the position's top
        alternative probability regardless of the substitute).

        The prefix is located by the original token's surface form in
        `target_text` (first occurrence), the same convention the generation
        splice uses (_generate_thought_alternative); the conversation context
        is not re-encoded — I_ij conditions on the generated text prefix only.

        Returns {(position, replacement_id): probability}.
        """
        import requests

        api_base = model_kwargs.get("api_base", "http://localhost:8000/v1")
        # Suffix-safe server-root derivation. rstrip("/v1") strips a CHARACTER
        # SET, not a suffix — it eats trailing "1"s of the PORT too
        # (":8001/v1" -> ":800"), silently sending every importance query to a
        # dead port and zeroing all I_ij. The checked-in config uses port 8001
        # (host port 8000 is owned by the PDF-reader relay), so this is the
        # live configuration, not an edge case.
        base_url = api_base.rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[: -len("/v1")]
        model = model_name.replace("openai/", "")

        # Group candidates by position; locate each position's text prefix via
        # the original token's surface form.
        by_position: dict[int, list[dict]] = {}
        prefixes: dict[int, str] = {}
        for c in candidates:
            pos = c["position"]
            if pos not in prefixes:
                orig = self._normalize_token_text(c.get("token", ""), keep_case=True)
                if not orig:
                    continue
                idx = target_text.find(orig)
                if idx < 0:
                    idx = target_text.lower().find(orig.lower())
                    if idx < 0:
                        continue
                prefixes[pos] = target_text[:idx].rstrip() or " "
            by_position.setdefault(pos, []).append(c)

        result: dict[tuple[int, int], float] = {}
        for pos, cands in by_position.items():
            prefix = prefixes[pos]
            top_logprobs: dict = {}
            try:
                resp = requests.post(
                    f"{base_url}/v1/completions",
                    json={
                        "model": model,
                        "prompt": prefix,
                        "max_tokens": 1,
                        "logprobs": self.top_k,
                        "temperature": 0,
                    },
                    timeout=10,
                )
                top_logprobs = (
                    resp.json()["choices"][0]["logprobs"]["top_logprobs"][0] or {}
                )
            except Exception as e:
                logger.debug(f"SDLG importance: logprobs failed at pos {pos}: {e}")

            for c in cands:
                sub_text = self._normalize_token_text(
                    c.get("replacement", ""), keep_case=True
                )
                key = (pos, c.get("replacement_id", 0))
                prob = self.match_substitute_probability(top_logprobs, sub_text)
                if prob is None and sub_text:
                    prob = self._echo_score_continuation(
                        base_url, model, prefix, " " + sub_text
                    )
                result[key] = float(prob) if prob is not None else 0.0

        return result

    def _generate_thought_alternative(
        self,
        sub: SubstitutionCandidate,
        thought_text: str,
        greedy_response: str,
        model_name: str,
        model_kwargs: dict,
        messages: list[dict],
    ) -> str:
        """Generate an alternative by substituting a token in the THOUGHT.

        Per SDLG Algorithm 1, applied to natural language reasoning:
        1. Substitute a high-attribution token in the thought text
        2. Truncate after the substitution point
        3. Let the LLM regenerate BOTH the rest of reasoning AND the code block

        This produces fundamentally different approaches because the NLI model's
        gradients identify tokens that most affect semantic meaning in natural
        language (unlike code tokens where gradients are near-random).
        """
        orig_text = sub.original_token.replace("Ġ", " ").replace("▁", " ").strip()
        sub_text = sub.substitute_token.replace("Ġ", " ").replace("▁", " ").strip()

        if not orig_text or not sub_text:
            return greedy_response

        # Find the original token in the THOUGHT TEXT and build a prefix
        if orig_text in thought_text:
            idx = thought_text.index(orig_text)
            thought_prefix = thought_text[:idx] + sub_text
        elif orig_text.lower() in thought_text.lower():
            idx = thought_text.lower().index(orig_text.lower())
            thought_prefix = thought_text[:idx] + sub_text
        else:
            return greedy_response

        # Build assistant prefix: just the substituted thought prefix
        # LLM will regenerate the rest of reasoning + code block from here
        assistant_prefix = thought_prefix

        api_messages = [{k: v for k, v in msg.items() if k != "extra"} for msg in messages]
        api_messages.append({"role": "assistant", "content": assistant_prefix})

        try:
            kwargs = {k: v for k, v in model_kwargs.items() if k != "temperature"}
            response = litellm.completion(
                model=model_name,
                messages=api_messages,
                temperature=0.0,
                max_tokens=1500,  # More tokens — regenerating thought + code
                **kwargs,
            )
            completion = response.choices[0].message.content or ""
            full_response = assistant_prefix + completion
            return full_response
        except Exception as e:
            logger.warning(f"SDLG thought completion failed: {e}")
            return greedy_response

    def _generate_code_alternative(
        self,
        sub: SubstitutionCandidate,
        code_text: str,
        preamble: str,
        postamble: str,
        greedy_response: str,
        model_name: str,
        model_kwargs: dict,
        messages: list[dict],
    ) -> str:
        """Generate an alternative by substituting a token in the CODE BLOCK.

        Keeps reasoning fixed, diversifies the implementation:
        1. Keep the reasoning preamble fixed (same THOUGHT)
        2. Substitute a token in the code block
        3. Truncate code after the substitution point
        4. Let the LLM complete the code from there
        5. Reassemble: preamble + new_code + postamble
        """
        orig_text = sub.original_token.replace("Ġ", " ").replace("▁", " ").strip()
        sub_text = sub.substitute_token.replace("Ġ", " ").replace("▁", " ").strip()

        if not orig_text or not sub_text:
            return greedy_response

        # Find the original token in the CODE TEXT and build a prefix
        if orig_text in code_text:
            idx = code_text.index(orig_text)
            code_prefix = code_text[:idx] + sub_text
        elif orig_text.lower() in code_text.lower():
            idx = code_text.lower().index(orig_text.lower())
            code_prefix = code_text[:idx] + sub_text
        else:
            return greedy_response

        # Build the assistant prefix: reasoning + code block opener + substituted code prefix
        # The preamble already includes ```mswea_bash_command\n
        assistant_prefix = preamble + code_prefix

        # Ask LLM to complete from this prefix (it will finish the code block)
        api_messages = [{k: v for k, v in msg.items() if k != "extra"} for msg in messages]
        api_messages.append({"role": "assistant", "content": assistant_prefix})

        try:
            kwargs = {k: v for k, v in model_kwargs.items() if k != "temperature"}
            response = litellm.completion(
                model=model_name,
                messages=api_messages,
                temperature=0.0,
                max_tokens=800,
                **kwargs,
            )
            completion = response.choices[0].message.content or ""
            full_response = assistant_prefix + completion
            return full_response
        except Exception as e:
            logger.warning(f"SDLG code completion failed: {e}")
            return greedy_response
