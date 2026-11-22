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
    # initial step size for Rprop; only used by rprop, None keeps the optimizer default
    delta_null: float | None = None
    # template file the values came from, None if they are set directly in .env
    template: str | None = None
