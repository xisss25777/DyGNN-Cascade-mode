"""
增强版评估模块
补充 MSE / MAPE / 误差分布 / 相关性 / 偏差分析
对应建模思路：第四层：模型性能与解释结果验证
"""

import math
import numpy as np
from typing import Dict, List, Sequence, Tuple


# ────────────────────────────────────────────────────
# 核心指标
# ────────────────────────────────────────────────────

def compute_all_metrics(
    targets: Sequence[float],
    predictions: Sequence[float],
) -> Dict[str, float]:
    """
    计算全套回归指标（在 log 尺度和原始尺度）

    Returns:
        dict，包含 mae / mse / rmse / mape / msle / r2 / pearson_r / bias / accuracy
    """
    n = len(targets)
    y_true = np.array(targets, dtype=np.float64)
    y_pred = np.array(predictions, dtype=np.float64)

    # ── 对数尺度指标（与训练目标一致）
    log_true = np.log1p(y_true)
    log_pred = np.log1p(np.maximum(y_pred, 0))

    mae_log  = float(np.mean(np.abs(log_pred - log_true)))
    mse_log  = float(np.mean((log_pred - log_true) ** 2))
    rmse_log = math.sqrt(mse_log)

    # ── 原始尺度指标
    mae   = float(np.mean(np.abs(y_pred - y_true)))
    mse   = float(np.mean((y_pred - y_true) ** 2))
    rmse  = math.sqrt(mse)

    # MAPE：避免零除
    mask  = y_true > 0
    mape  = float(np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask]))) if mask.any() else 0.0

    # MSLE（Mean Squared Log Error）
    msle  = float(np.mean((np.log1p(np.maximum(y_pred, 0)) - np.log1p(y_true)) ** 2))

    # R²
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Pearson 相关系数
    if n > 1:
        pearson_r = float(np.corrcoef(y_true, y_pred)[0, 1])
        if math.isnan(pearson_r):
            pearson_r = 0.0
    else:
        pearson_r = 0.0

    # 系统性偏差（预测均值 - 真实均值）
    bias = float(np.mean(y_pred) - np.mean(y_true))
    bias_log = float(np.mean(log_pred) - np.mean(log_true))

    # 准确率：误差 < 0.5（log 尺度）
    accuracy = float(np.mean(np.abs(log_pred - log_true) < 0.5))

    # 准确率：误差 < 1.0（log 尺度，宽松版）
    accuracy_1 = float(np.mean(np.abs(log_pred - log_true) < 1.0))

    return {
        # 对数尺度（与训练目标对齐）
        "mae_log":    round(mae_log,  4),
        "mse_log":    round(mse_log,  4),
        "rmse_log":   round(rmse_log, 4),
        "bias_log":   round(bias_log, 4),
        # 原始尺度
        "mae":        round(mae,  2),
        "mse":        round(mse,  2),
        "rmse":       round(rmse, 2),
        "mape":       round(mape, 4),
        "msle":       round(msle, 4),
        "r2":         round(r2,   4),
        "pearson_r":  round(pearson_r, 4),
        "bias":       round(bias, 2),
        # 分类准确率（误差阈值）
        "acc_0.5":    round(accuracy,   4),
        "acc_1.0":    round(accuracy_1, 4),
        "n":          n,
    }


# ────────────────────────────────────────────────────
# 误差分布分析
# ────────────────────────────────────────────────────

def error_distribution(
    targets: Sequence[float],
    predictions: Sequence[float],
) -> Dict[str, object]:
    """
    分析误差的详细分布，用于诊断系统性偏差
    """
    y_true = np.array(targets, dtype=np.float64)
    y_pred = np.array(predictions, dtype=np.float64)
    log_true = np.log1p(y_true)
    log_pred = np.log1p(np.maximum(y_pred, 0))

    errors     = log_pred - log_true          # 有符号误差（正=高估，负=低估）
    abs_errors = np.abs(errors)

    # 分位数
    percentiles = [10, 25, 50, 75, 90, 95]
    pct_vals    = {f"p{p}": round(float(np.percentile(abs_errors, p)), 4)
                   for p in percentiles}

    # 按真实规模分组误差
    small_mask  = y_true < 10
    medium_mask = (y_true >= 10) & (y_true < 100)
    large_mask  = y_true >= 100

    def group_stats(mask: np.ndarray) -> dict:
        if not mask.any():
            return {"count": 0, "mae_log": 0.0, "bias_log": 0.0}
        return {
            "count":    int(mask.sum()),
            "mae_log":  round(float(np.mean(abs_errors[mask])), 4),
            "bias_log": round(float(np.mean(errors[mask])),     4),
        }

    return {
        "error_mean":   round(float(np.mean(errors)),     4),   # 系统偏差
        "error_std":    round(float(np.std(errors)),      4),
        "error_median": round(float(np.median(errors)),   4),
        "abs_error_mean":   round(float(np.mean(abs_errors)),   4),
        "abs_error_max":    round(float(np.max(abs_errors)),    4),
        "percentiles":  pct_vals,
        "overestimate_ratio":  round(float((errors > 0).mean()), 4),   # 高估比例
        "underestimate_ratio": round(float((errors < 0).mean()), 4),   # 低估比例
        "by_group": {
            "small_lt10":   group_stats(small_mask),
            "medium_10_100":group_stats(medium_mask),
            "large_ge100":  group_stats(large_mask),
        },
    }


# ────────────────────────────────────────────────────
# 多数据集对比表
# ────────────────────────────────────────────────────

def format_comparison_table(
    results: Dict[str, Dict],
    metric_keys: List[str] = None,
) -> str:
    """
    将多数据集结果格式化为 Markdown 表格（用于论文）
    """
    if metric_keys is None:
        metric_keys = ["mae_log", "rmse_log", "pearson_r", "acc_0.5", "r2", "bias_log"]

    header = "| 数据集 | " + " | ".join(metric_keys) + " |"
    sep    = "|--------|" + "|".join(["--------"] * len(metric_keys)) + "|"
    rows   = [header, sep]

    for dataset, res in results.items():
        metrics = res.get("metrics", res)
        vals = [str(metrics.get(k, "-")) for k in metric_keys]
        rows.append(f"| {dataset} | " + " | ".join(vals) + " |")

    return "\n".join(rows)


# ────────────────────────────────────────────────────
# 偏差校正建议
# ────────────────────────────────────────────────────

def diagnose_bias(metrics: Dict, dist: Dict) -> List[str]:
    """
    根据评估结果给出偏差诊断和改进建议
    """
    suggestions = []

    bias_log = metrics.get("bias_log", 0.0)
    pearson  = metrics.get("pearson_r", 0.0)
    mae_log  = metrics.get("mae_log", 0.0)
    over_r   = dist.get("overestimate_ratio", 0.5)

    if abs(bias_log) > 0.3:
        direction = "高估" if bias_log > 0 else "低估"
        suggestions.append(
            f"⚠️ 系统性{direction}偏差: bias_log={bias_log:.3f}。"
            "建议在损失函数中加入偏差正则项，或对预测结果进行后校正。"
        )

    if pearson < 0.3:
        suggestions.append(
            f"⚠️ 相关性极低 (r={pearson:.3f})，说明模型几乎无判别力。"
            "核心问题：特征与目标之间的信息不足，或训练样本太少导致过拟合。"
            "建议：(1) 增加数据增强 (2) 使用交叉验证 (3) 检查特征泄露。"
        )

    if over_r > 0.7:
        suggestions.append(
            f"⚠️ 过度高估比例: {over_r*100:.0f}%。"
            "可能原因：特征重缩放导致输入值偏大，或 Softplus 激活函数的正值偏置。"
            "建议：(1) 检查 rescale_features 的目标均值设置 (2) 添加输出偏置校正层。"
        )

    if mae_log > 0.8:
        suggestions.append(
            f"⚠️ MAE_log={mae_log:.3f} 过大。"
            "建议：(1) 增加训练轮数 (2) 调低学习率至 1e-4 以下 "
            "(3) 使用数据增强扩展样本量至 100+ 条。"
        )

    if not suggestions:
        suggestions.append("✅ 模型偏差在可接受范围内，继续优化细节。")

    return suggestions
