from dataclasses import dataclass


@dataclass
class PipelineConfig:
    observation_seconds: int = 6 * 3600
    slice_seconds: int = 1800
    test_ratio: float = 0.3
    learning_rate: float = 0.001  # 正常学习率
    # ============================================================
    # 训练参数优化（方案B：加速训练）
    # ============================================================
    epochs: int = 300           # 优化：800→300，减少62%训练时间
    l2_penalty: float = 0.0005  # L2 正则化
    random_seed: int = 42
    use_log_target: bool = True
    knn_neighbors: int = 9
    patience: int = 30          # 优化：80→30，更快早停
    hidden_dim: int = 128       # 优化：256→128，减少计算量
    gru_layers: int = 1         # 优化：2→1，减少计算量
    dropout: float = 0.2        # 正常Dropout概率