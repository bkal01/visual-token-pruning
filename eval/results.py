from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class SampleResult:
    sample_id: int
    subset: str
    pruner_name: str
    success: bool
    question: str | None = None
    ground_truth: str | None = None
    prediction: str | None = None
    score: float | None = None
    ttft_ms: float | None = None
    decode_latency_ms: float | None = None
    num_output_tokens: int | None = None
    initial_visual_tokens: int | None = None
    final_visual_tokens: int | None = None
    pruning_ratio: float | None = None
    pruning_steps: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SampleResult":
        return cls(**data)
