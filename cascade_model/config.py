from dataclasses import dataclass


@dataclass
class PipelineConfig:
    observation_seconds: int = 6 * 3600
    slice_seconds: int = 1800
    test_ratio: float = 0.3
    learning_rate: float = 0.01
    epochs: int = 5000
    l2_penalty: float = 0.0005
    random_seed: int = 42
    use_log_target: bool = True
    knn_neighbors: int = 9
