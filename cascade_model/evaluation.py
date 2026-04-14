import math
from typing import Dict, List, Sequence


def regression_metrics(targets: Sequence[float], predictions: Sequence[float]) -> Dict[str, float]:
    count = len(targets)
    mae = sum(abs(pred - true) for pred, true in zip(predictions, targets)) / count
    rmse = math.sqrt(sum((pred - true) ** 2 for pred, true in zip(predictions, targets)) / count)
    mape = (
        sum(abs((pred - true) / true) for pred, true in zip(predictions, targets) if true != 0) / count
        if count
        else 0.0
    )
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "mape": round(mape, 4)}


def deletion_test(
    row: List[float],
    weights: List[float],
    bias: float,
    slice_feature_indices: List[int],
    use_log_target: bool = False,
) -> Dict[str, float]:
    base_raw = sum(value * weight for value, weight in zip(row, weights)) + bias
    ablated_row = list(row)
    for idx in slice_feature_indices:
        ablated_row[idx] = 0.0
    ablated_raw = sum(value * weight for value, weight in zip(ablated_row, weights)) + bias
    base_prediction = _restore(base_raw, use_log_target)
    ablated_prediction = _restore(ablated_raw, use_log_target)
    delta = base_prediction - ablated_prediction
    drop_ratio = delta / max(1.0, base_prediction)
    return {
        "base_prediction": round(base_prediction, 4),
        "ablated_prediction": round(ablated_prediction, 4),
        "drop_ratio": round(drop_ratio, 4),
        "delta": round(delta, 4),
        "effect_direction": "decrease" if delta >= 0 else "increase",
    }


def _restore(value: float, use_log_target: bool) -> float:
    if use_log_target:
        return max(1.0, math.expm1(value))
    return max(1.0, value)
