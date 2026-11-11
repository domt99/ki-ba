from dataclasses import dataclass

@dataclass
class Config:
    dataSetType: str
    topology: list[int]
    activations: list[str]
    calculationType: str
    optimizerType: str
    epochCount: int
    batchSize: int
    seed: int
    learningRate: float | None = None
    alpha: float | None = None