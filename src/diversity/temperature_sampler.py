"""Temperature-based diverse generation (Phase 2a).

Simple diversity mechanism: generate N candidates by sampling at different temperatures.
The first candidate is the greedy response (already obtained), the rest are sampled
at temperature > 0.
"""

import logging

import litellm

logger = logging.getLogger(__name__)


class TemperatureSampler:
    """Generate diverse candidates via temperature sampling.

    Uses the same litellm model interface as DefaultAgent to ensure
    response format compatibility.
    """

    def __init__(
        self,
        n_candidates: int = 5,
        temperature: float = 0.7,
    ):
        self.n_candidates = n_candidates
        self.temperature = temperature

    def generate(
        self,
        model_name: str,
        model_kwargs: dict,
        messages: list[dict],
        greedy_response: str,
    ) -> list[str]:
        """Generate N diverse candidate responses.

        Args:
            model_name: litellm model name (e.g., "openai/qwen3-coder").
            model_kwargs: Model kwargs dict (api_base, api_key, etc.).
            messages: Conversation history (same format as DefaultAgent.messages).
            greedy_response: The greedy (temp=0) response already obtained.

        Returns:
            List of N response content strings. First is always the greedy response.
        """
        candidates = [greedy_response]

        # Prepare messages for API (strip 'extra' field like LitellmModel does)
        api_messages = [{k: v for k, v in msg.items() if k != "extra"} for msg in messages]

        n_remaining = self.n_candidates - 1
        if n_remaining <= 0:
            return candidates

        # Try batch sampling with n parameter first (single API call)
        try:
            candidates.extend(
                self._batch_sample(model_name, model_kwargs, api_messages, n_remaining)
            )
            return candidates
        except Exception as e:
            logger.warning(f"Batch sampling failed ({e}), falling back to sequential")

        # Fallback: sequential sampling
        for i in range(n_remaining):
            try:
                response = litellm.completion(
                    model=model_name,
                    messages=api_messages,
                    temperature=self.temperature,
                    **{k: v for k, v in model_kwargs.items() if k != "temperature"},
                )
                content = response.choices[0].message.content or ""
                candidates.append(content)
            except Exception as e:
                logger.warning(f"Sample {i + 1} failed: {e}")
                # Duplicate greedy as fallback
                candidates.append(greedy_response)

        return candidates

    def _batch_sample(
        self,
        model_name: str,
        model_kwargs: dict,
        api_messages: list[dict],
        n: int,
    ) -> list[str]:
        """Try to generate n samples in a single API call using the n parameter."""
        kwargs = {k: v for k, v in model_kwargs.items() if k != "temperature"}
        response = litellm.completion(
            model=model_name,
            messages=api_messages,
            temperature=self.temperature,
            n=n,
            **kwargs,
        )
        return [
            choice.message.content or ""
            for choice in response.choices
        ]
