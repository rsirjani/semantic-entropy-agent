"""Single source of truth for branching/clustering config defaults.

These defaults are only used when a key is ABSENT from the loaded config — the
shipped ``configs/branching.yaml`` sets all of them explicitly. They exist so
that the orchestrators, the clusterer, and the eval drivers cannot silently
disagree on a fallback value (previously ``entailment_threshold`` defaulted to
0.3 in one place and 0.5 in two others). Read every fallback from here.
"""

BRANCHING_DEFAULTS: dict = {
    # Diversity arm — the single mutually-exclusive switch.
    "diversity_method": "strategy_proposal",  # {"strategy_proposal","sdlg","none"}
    "n_strategies": 5,
    # Shared sampling temperature for the proposer AND the vanilla resample
    # baseline, so the arms differ only in the branching mechanism.
    "sample_temperature": 1.0,

    # Clustering / branching gate.
    "clustering_strategy": "greedy",          # {"greedy","connected","kernel"}
    "kernel_t": 1.0,                           # heat-kernel diffusion time (>0)
    "entailment_threshold": 0.5,               # bidirectional-entailment cutoff
    "entropy_threshold": 0.0,                  # tau — branch iff entropy > tau

    # Trajectory / step budgets.
    "max_trajectories": 30,
    "max_search_steps": 240,
    "min_search_steps": 8,
    "patch_read_budget": 5,
    "relevance_threshold": 0.5,
    "relevance_use_nli": False,
    "low_relevance_streak": 3,

    # SDLG arm.
    "sdlg_n_alternatives": 5,
    "sdlg_top_k": 20,
    # Faithful SDLG substitutes only the REASONING (R1.1 / Aichberger 2025).
    # Code-token substitution is an explicit, ablatable opt-in, OFF by default.
    "sdlg_diversify_code": False,
}

# Allowed values for validated string switches.
VALID_DIVERSITY_METHODS = ("strategy_proposal", "sdlg", "none")


def cfg(branching_config: dict, key: str):
    """Read ``key`` from a branching config, falling back to BRANCHING_DEFAULTS."""
    return branching_config.get(key, BRANCHING_DEFAULTS[key])
