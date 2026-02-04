"""Simple configuration for DatBench evaluation."""

# DatBench subsets
DATBENCH_SUBSETS = [
    "chart",
    "counting",
    "document",
    "general",
    "grounding",
    "math",
    "scene",
    "spatial",
    "table",
]

# Default parameters for each pruner
DEFAULT_PARAMS = {
    "baseline": {},
    "fastv": {
        "target_layers": [1],
        "filtering_ratio": 0.5,
    },
    "droplet": {
        "target_layers": [8, 16, 24, 27],
        "filtering_ratio": 0.2,
    },
    "visiondrop": {
        "vision_target_layers": [],
        "llm_target_layers": [8, 16, 24, 27],
        "filtering_ratio": 0.2,
    },
    "feather": {
        "target_layers": [8, 16],
        "uniform_target_layers": [8],
        "filtering_ratio": 0.75,
        "stride": 3,
    },
    "uniform": {
        "target_layers": [3],
        "stride": 2,
    },
    "token_embeddings": {
        "filtering_ratio": 0.5,
    },
}


def get_pruner(name, **override_params):
    """
    Create a pruner by name. Returns None for baseline.

    Args:
        name: One of 'baseline', 'fastv', 'droplet', 'visiondrop', 'feather', 'uniform', 'token_embeddings'
        **override_params: Override default parameters
    """
    if name not in DEFAULT_PARAMS:
        raise ValueError(
            f"Unknown pruner '{name}'. Available: {list(DEFAULT_PARAMS.keys())}"
        )

    if name == "baseline":
        return None

    # Merge defaults with overrides
    params = {**DEFAULT_PARAMS[name], **override_params}

    if name == "fastv":
        from pruners.fastv import FastVPruner

        return FastVPruner(**params)
    elif name == "droplet":
        from pruners.droplet import Droplet

        return Droplet(**params)
    elif name == "visiondrop":
        from pruners.visiondrop import VisionDropPruner

        return VisionDropPruner(**params)
    elif name == "feather":
        from pruners.feather import FeatherPruner

        return FeatherPruner(**params)
    elif name == "uniform":
        from pruners.uniform import UniformPruner

        return UniformPruner(**params)
    elif name == "token_embeddings":
        from pruners.token_embeddings import TokenEmbeddingsPruner

        return TokenEmbeddingsPruner(**params)
