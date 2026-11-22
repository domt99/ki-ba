from dataclasses import dataclass


@dataclass
class Config:
    data_set_type: str
    topology: list[int]
    activations: list[str]
    calculation_type: str
    optimizer_type: str
    epoch_count: int
    batch_size: int
    seed: int
    # step size for GradientDescent (learning_rate) and Adam (alpha); Rprop doesn't use it
    learning_rate: float | None = None
