from dataclasses import dataclass


@dataclass
class PipelineConfig:
    observation_seconds: int = 6 * 3600
    slice_seconds: int = 1800
    test_ratio: float = 0.3
    learning_rate: float = 0.0001  # 降低学习率以适应更大的模型
    epochs: int = 800  # 增加训练轮数
    l2_penalty: float = 0.0005  # 增加 L2 正则化
    random_seed: int = 42
    use_log_target: bool = True
    knn_neighbors: int = 9
    patience: int = 80  # 增加早停耐心值
    hidden_dim: int = 256  # 保持隐藏层维度
    gru_layers: int = 2  # 保持 GRU 层数
    dropout: float = 0.3  # 增加 Dropout 概率