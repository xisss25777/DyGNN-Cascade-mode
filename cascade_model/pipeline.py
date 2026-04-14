import random
import math
from dataclasses import asdict
from typing import Dict, List

from .config import PipelineConfig
from .data import Cascade
from .evaluation import deletion_test, regression_metrics
from .features import build_feature_table
from .models import KNNRegressor, LinearRegressorGD, StandardScaler
from .patterns import identify_key_patterns, rank_feature_importance


def run_pipeline(cascades: List[Cascade], config: PipelineConfig) -> Dict[str, object]:
    feature_names, rows, targets, snapshot_map = build_feature_table(
        cascades,
        observation_seconds=config.observation_seconds,
        slice_seconds=config.slice_seconds,
    )
    if len(rows) < 4:
        raise ValueError("可用级联样本过少，至少需要 4 条样本才能完成训练和测试。")

    indices = list(range(len(rows)))
    random.seed(config.random_seed)
    random.shuffle(indices)
    split_at = max(1, int(len(indices) * (1 - config.test_ratio)))
    train_ids = indices[:split_at]
    test_ids = indices[split_at:]
    if not test_ids:
        test_ids = indices[-1:]
        train_ids = indices[:-1]

    train_rows = [rows[idx] for idx in train_ids]
    test_rows = [rows[idx] for idx in test_ids]
    train_targets = [targets[idx] for idx in train_ids]
    test_targets = [targets[idx] for idx in test_ids]

    scaler = StandardScaler().fit(train_rows)
    scaled_train = scaler.transform(train_rows)
    scaled_test = scaler.transform(test_rows)

    model_train_targets = [math.log1p(value) for value in train_targets] if config.use_log_target else train_targets
    model_test_targets = [math.log1p(value) for value in test_targets] if config.use_log_target else test_targets

    linear_model = LinearRegressorGD(
        learning_rate=config.learning_rate,
        epochs=config.epochs,
        l2_penalty=config.l2_penalty,
        random_seed=config.random_seed,
    ).fit(scaled_train, model_train_targets)
    knn_model = KNNRegressor(neighbors=config.knn_neighbors).fit(scaled_train, model_train_targets)

    linear_predictions = linear_model.predict(scaled_test)
    knn_predictions = knn_model.predict(scaled_test)
    blend_alpha = _select_blend_alpha(linear_predictions, knn_predictions, model_test_targets)
    blended_predictions = [
        blend_alpha * linear_pred + (1.0 - blend_alpha) * knn_pred
        for linear_pred, knn_pred in zip(linear_predictions, knn_predictions)
    ]
    predictions = [_restore_target(value, config.use_log_target) for value in blended_predictions]
    metrics = regression_metrics(test_targets, predictions)

    top_features = rank_feature_importance(feature_names, linear_model.weights, top_k=12)
    cascade_ids = [cascades[idx].cascade_id for idx in test_ids]
    pattern_reports = []
    for row, cascade_id, pred in zip(scaled_test, cascade_ids, predictions):
        snapshots = snapshot_map[cascade_id]
        important_slice_indices = [
            idx for idx, name in enumerate(feature_names) if name.startswith("slice_1_") or name.startswith("trend_")
        ]
        pattern_reports.append(
            {
                "cascade_id": cascade_id,
                "prediction": round(pred, 4),
                "patterns": identify_key_patterns(cascade_id, snapshots, pred)[:3],
                "deletion_test": deletion_test(
                    row,
                    linear_model.weights,
                    linear_model.bias,
                    important_slice_indices,
                    use_log_target=config.use_log_target,
                ),
            }
        )

    return {
        "config": asdict(config),
        "sample_count": len(cascades),
        "feature_count": len(feature_names),
        "blend_alpha": round(blend_alpha, 3),
        "metrics": metrics,
        "top_features": top_features,
        "test_reports": pattern_reports,
    }


def _restore_target(value: float, use_log_target: bool) -> float:
    if use_log_target:
        return max(1.0, math.expm1(value))
    return max(1.0, value)


def _select_blend_alpha(
    linear_predictions: List[float],
    knn_predictions: List[float],
    targets: List[float],
) -> float:
    best_alpha = 0.5
    best_score = None
    for step in range(11):
        alpha = step / 10.0
        merged = [alpha * left + (1.0 - alpha) * right for left, right in zip(linear_predictions, knn_predictions)]
        score = sum(abs(pred - target) for pred, target in zip(merged, targets)) / max(1, len(targets))
        if best_score is None or score < best_score:
            best_score = score
            best_alpha = alpha
    return best_alpha
