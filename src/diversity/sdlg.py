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
    ):
        self.nli = nli_model
        self.n_candidates = n_candidates
        self.top_k = top_k_substitutes
        self.importance_threshold = importance_threshold

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

        # Split: thought-level alternatives get the majority (more impactful),
        # code-level alternatives fill the rest
        n_thought = max(1, (self.n_candidates - 1 + 1) // 2)  # ceil half
        n_code = (self.n_candidates - 1) - n_thought

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

        # If we got nothing, fall back to temperature
        if len(candidates) <= 1:
            logger.warning("SDLG: no alternatives from either target, falling back to temperature")
            return self._fallback_temperature(model_name, model_kwargs, messages, greedy_response)

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

        # Step 2: Get importance scores from LLM for the top positions
        positions = [c["position"] for c in server_candidates[:self.top_k]]
        tokens_list = [c["token"] for c in server_candidates[:self.top_k]]

        importance_scores = self._get_importance_scores(
            target_text, positions, tokens_list, model_name, model_kwargs, messages
        )

        # Step 3: Combine server scores with importance scores
        all_candidates = []
        for c in server_candidates:
            pos = c["position"]
            A_i = c["attribution"]
            S_ij = c["substitution"]

            # Get importance for this position's replacement
            I_ij = 0.0
            if pos in importance_scores:
                # Find matching replacement or use top importance
                for nli_id, prob in importance_scores[pos]:
                    if nli_id == c.get("replacement_id", -1):
                        I_ij = prob
                        break
                if I_ij == 0.0 and importance_scores[pos]:
                    I_ij = importance_scores[pos][0][1]  # Use top importance

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

    def _get_importance_scores(
        self,
        code_text: str,
        word_starts: list[int],
        tokens: list[str],
        model_name: str,
        model_kwargs: dict,
        messages: list[dict],
    ) -> dict[int, list[tuple[int, float]]]:
        """Get LLM token probabilities at word-start positions in the target text.

        I_ij = p(v_j | y_<i, x, w) — the probability the LLM assigns
        to alternative token v_j given the context up to position i.

        Uses vLLM's completions endpoint with logprobs to get real LLM
        probabilities at each position. Works for both thought and code text.

        Returns {position: [(nli_token_id, probability), ...]} for top-k alternatives.
        """
        import math
        import requests

        result = {}

        nli_tokens = tokens  # DeBERTa tokens
        text_pieces = []
        for t in nli_tokens:
            text_pieces.append(t.replace("Ġ", " ").replace("▁", " "))

        api_base = model_kwargs.get("api_base", "http://localhost:8000/v1")
        base_url = api_base.rstrip("/v1").rstrip("/")

        # Use the code text as prompt context for importance scoring
        for pos in word_starts:
            prefix = "".join(text_pieces[:pos]).strip()
            if not prefix:
                prefix = " "

            try:
                resp = requests.post(
                    f"{base_url}/v1/completions",
                    json={
                        "model": model_name.replace("openai/", ""),
                        "prompt": prefix,
                        "max_tokens": 1,
                        "logprobs": self.top_k,
                        "temperature": 0,
                    },
                    timeout=10,
                )
                data = resp.json()
                top_logprobs = data["choices"][0]["logprobs"]["top_logprobs"][0]

                alternatives = []
                for token_text, logprob in top_logprobs.items():
                    prob = math.exp(logprob)
                    clean = token_text.strip()
                    orig_clean = nli_tokens[pos].replace("Ġ", "").replace("▁", "").strip()
                    if clean.lower() != orig_clean.lower() and clean:
                        alternatives.append((hash(clean) % 100000, prob))

                result[pos] = alternatives[:self.top_k]
            except Exception as e:
                logger.debug(f"Logprobs failed for pos {pos}: {e}")
                continue

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

    def _fallback_temperature(
        self,
        model_name: str,
        model_kwargs: dict,
        messages: list[dict],
        greedy_response: str,
    ) -> list[str]:
        """Fallback to temperature sampling when SDLG can't compute scores."""
        from src.diversity.temperature_sampler import TemperatureSampler
        sampler = TemperatureSampler(
            n_candidates=self.n_candidates,
            temperature=0.7,
        )
        return sampler.generate(model_name, model_kwargs, messages, greedy_response)
