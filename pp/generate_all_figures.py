"""
全套可视化图表生成器
运行: python generate_all_figures.py [--json path/to/report.json] [--dataset wikipedia]

生成图表列表:
  1. 级联规模分布直方图（log 尺度）
  2. 预测值 vs 真实值散点图
  3. 误差分布箱线图（按规模分组）
  4. 时间注意力权重热力图
  5. 关键传播模式柱状图
  6. 多数据集指标对比雷达图
  7. 训练损失曲线（若有记录）
  8. 特征重要性条形图
"""

import json
import math
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# Windows 控制台编码兼容
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# ── 中文字体配置
plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

OUTPUT_DIR = Path("outputs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 颜色方案
COLORS = {
    "primary":  "#2196F3",
    "secondary":"#FF5722",
    "success":  "#4CAF50",
    "warning":  "#FF9800",
    "purple":   "#9C27B0",
    "teal":     "#009688",
    "grid":     "#EEEEEE",
}

DATASET_COLORS = {
    "wikipedia": "#2196F3",
    "reddit":    "#FF5722",
    "enron":     "#4CAF50",
    "mooc":      "#FF9800",
}


# ═══════════════════════════════════════════════════
# 1. 级联规模分布图
# ═══════════════════════════════════════════════════

def plot_cascade_size_distribution(cascades_by_dataset: dict, save_path: str = None):
    """
    绘制各数据集的级联规模分布（log 尺度直方图 + KDE）
    """
    n_datasets = len(cascades_by_dataset)
    fig, axes = plt.subplots(1, n_datasets, figsize=(5 * n_datasets, 4), sharey=False)
    if n_datasets == 1:
        axes = [axes]

    for ax, (name, sizes) in zip(axes, cascades_by_dataset.items()):
        log_sizes = np.log1p(sizes)
        color = DATASET_COLORS.get(name, COLORS["primary"])

        ax.hist(log_sizes, bins=20, color=color, alpha=0.75, edgecolor="white", linewidth=0.5)

        # 统计线
        ax.axvline(np.mean(log_sizes),   color="red",    ls="--", lw=1.5, label=f"均值={np.mean(log_sizes):.2f}")
        ax.axvline(np.median(log_sizes), color="orange", ls=":",  lw=1.5, label=f"中位={np.median(log_sizes):.2f}")

        ax.set_title(f"{name.capitalize()}\n(n={len(sizes)})", fontsize=12, fontweight="bold")
        ax.set_xlabel("log(1 + 级联规模)", fontsize=10)
        ax.set_ylabel("频次", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#FAFAFA")

    fig.suptitle("信息级联规模分布 (对数尺度)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    path = save_path or str(OUTPUT_DIR / "fig1_cascade_size_distribution.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"✅ 图1 级联规模分布 → {path}")
    return path


# ═══════════════════════════════════════════════════
# 2. 预测 vs 真实 散点图
# ═══════════════════════════════════════════════════

def plot_pred_vs_true(
    targets, predictions, dataset_name: str = "",
    metrics: dict = None, save_path: str = None
):
    """
    预测值 vs 真实值散点图（log 尺度）+ 对角线
    """
    log_true = np.log1p(np.array(targets, dtype=float))
    log_pred = np.log1p(np.maximum(np.array(predictions, dtype=float), 0))

    fig, ax = plt.subplots(figsize=(6, 6))

    # 散点
    scatter = ax.scatter(
        log_true, log_pred,
        c=np.abs(log_pred - log_true),     # 颜色 = 误差大小
        cmap="RdYlGn_r",
        s=80, alpha=0.8, edgecolors="white", linewidths=0.5
    )
    plt.colorbar(scatter, ax=ax, label="|误差| (log尺度)")

    # 对角线（完美预测）
    lim = [min(log_true.min(), log_pred.min()) - 0.2,
           max(log_true.max(), log_pred.max()) + 0.2]
    ax.plot(lim, lim, "k--", lw=1.5, alpha=0.6, label="完美预测线 y=x")

    # ±0.5 误差带
    ax.fill_between(lim, [x - 0.5 for x in lim], [x + 0.5 for x in lim],
                    alpha=0.1, color="green", label="±0.5 容差带")

    # 指标文字
    if metrics:
        info = (f"MAE_log={metrics.get('mae_log', '?'):.3f}\n"
                f"r={metrics.get('pearson_r', '?'):.3f}\n"
                f"R²={metrics.get('r2', '?'):.3f}\n"
                f"acc@0.5={metrics.get('acc_0.5', '?'):.2%}")
        ax.text(0.05, 0.95, info, transform=ax.transAxes,
                fontsize=9, va="top", ha="left",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("真实值 log(1+N)", fontsize=11)
    ax.set_ylabel("预测值 log(1+N̂)", fontsize=11)
    ax.set_title(f"预测精度散点图 — {dataset_name}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#FAFAFA")

    path = save_path or str(OUTPUT_DIR / f"fig2_pred_vs_true_{dataset_name}.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"✅ 图2 预测散点图 → {path}")
    return path


# ═══════════════════════════════════════════════════
# 3. 误差分布图（按规模分组）
# ═══════════════════════════════════════════════════

def plot_error_distribution(
    targets, predictions, dataset_name: str = "", save_path: str = None
):
    """
    绘制误差分布直方图 + 按规模分组的箱线图
    """
    y_true = np.array(targets, dtype=float)
    y_pred = np.maximum(np.array(predictions, dtype=float), 0)
    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)
    errors = log_pred - log_true     # 有符号误差

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ── 左图：误差直方图
    ax1.hist(errors, bins=30, color=COLORS["primary"], alpha=0.75,
             edgecolor="white", linewidth=0.5)
    ax1.axvline(0,  color="black", ls="-",  lw=1.5, label="零偏差")
    ax1.axvline(np.mean(errors), color="red", ls="--", lw=2,
                label=f"均值偏差={np.mean(errors):.3f}")
    ax1.axvline(np.median(errors), color="orange", ls=":", lw=2,
                label=f"中位偏差={np.median(errors):.3f}")
    ax1.set_xlabel("预测误差 (log尺度)", fontsize=11)
    ax1.set_ylabel("频次", fontsize=11)
    ax1.set_title("误差分布直方图", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor("#FAFAFA")

    # ── 右图：按规模分组箱线图
    groups = []
    group_labels = []
    group_sizes  = []
    bins = [(0, 10, "小规模\n(<10)"),
            (10, 50, "中规模\n(10-50)"),
            (50, 200, "大规模\n(50-200)"),
            (200, 1e9, "超大规模\n(≥200)")]

    for lo, hi, label in bins:
        mask = (y_true >= lo) & (y_true < hi)
        if mask.sum() >= 2:
            groups.append(errors[mask])
            group_labels.append(label)
            group_sizes.append(int(mask.sum()))

    if groups:
        bp = ax2.boxplot(groups, patch_artist=True, notch=False,
                         medianprops=dict(color="red", lw=2))
        colors_bp = [COLORS["primary"], COLORS["secondary"],
                     COLORS["success"], COLORS["warning"]]
        for patch, color in zip(bp["boxes"], colors_bp):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax2.axhline(0, color="black", ls="--", lw=1, alpha=0.5)
        ax2.set_xticklabels(
            [f"{label}\n(n={n})" for label, n in zip(group_labels, group_sizes)],
            fontsize=9
        )
        ax2.set_ylabel("预测误差 (log尺度)", fontsize=11)
        ax2.set_title("各规模组误差分布", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="y")
        ax2.set_facecolor("#FAFAFA")
    else:
        ax2.text(0.5, 0.5, "样本不足", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=14)

    fig.suptitle(f"预测误差分析 — {dataset_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    path = save_path or str(OUTPUT_DIR / f"fig3_error_distribution_{dataset_name}.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"✅ 图3 误差分布图 → {path}")
    return path


# ═══════════════════════════════════════════════════
# 4. 时间注意力权重热力图
# ═══════════════════════════════════════════════════

def plot_attention_heatmap(
    pattern_reports: list, dataset_name: str = "",
    max_samples: int = 15, save_path: str = None
):
    """
    绘制各样本的时间注意力权重热力图（行=级联, 列=时间片）
    """
    # 提取注意力权重
    attention_matrix = []
    cascade_labels   = []
    for report in pattern_reports[:max_samples]:
        slices = report.get("top_attention_slices", [])
        if not slices:
            continue
        # 重建完整权重序列
        max_slice = max(s["slice"] for s in slices)
        weights = [0.0] * max_slice
        for s in slices:
            weights[s["slice"] - 1] = s["weight"]
        attention_matrix.append(weights)
        cascade_labels.append(f"C{report['cascade_id']}")

    if not attention_matrix:
        print("⚠️  无注意力数据，跳过热力图")
        return None

    # 补齐长度
    max_len = max(len(w) for w in attention_matrix)
    attention_matrix = [w + [0.0] * (max_len - len(w)) for w in attention_matrix]
    mat = np.array(attention_matrix)

    fig, ax = plt.subplots(figsize=(min(max_len * 0.8 + 2, 14), max(4, len(cascade_labels) * 0.5 + 1.5)))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=mat.max())
    plt.colorbar(im, ax=ax, label="注意力权重")

    ax.set_xticks(range(max_len))
    ax.set_xticklabels([f"T{i+1}" for i in range(max_len)], fontsize=9)
    ax.set_yticks(range(len(cascade_labels)))
    ax.set_yticklabels(cascade_labels, fontsize=9)
    ax.set_xlabel("时间片", fontsize=11)
    ax.set_ylabel("级联样本", fontsize=11)
    ax.set_title(f"时间注意力权重热力图 — {dataset_name}", fontsize=13, fontweight="bold")

    # 在格子中显示数值
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j] > 0.01:
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if mat[i, j] < 0.6 * mat.max() else "white")

    plt.tight_layout()
    path = save_path or str(OUTPUT_DIR / f"fig4_attention_heatmap_{dataset_name}.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"✅ 图4 注意力热力图 → {path}")
    return path


# ═══════════════════════════════════════════════════
# 5. 关键传播模式分析图
# ═══════════════════════════════════════════════════

def plot_key_patterns(
    pattern_reports: list, dataset_name: str = "", save_path: str = None
):
    """
    统计各传播模式的出现频率，绘制水平条形图
    """
    from collections import Counter

    pattern_counter = Counter()
    for report in pattern_reports:
        for p in report.get("patterns", []):
            pattern_name = p.get("pattern", "unknown")
            pattern_counter[pattern_name] += 1

    if not pattern_counter:
        print("⚠️  无模式数据，跳过")
        return None

    patterns_sorted = sorted(pattern_counter.items(), key=lambda x: x[1], reverse=True)
    names  = [p[0].replace("_", "\n") for p in patterns_sorted[:10]]
    counts = [p[1] for p in patterns_sorted[:10]]
    total  = sum(pattern_counter.values())

    fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.7 + 1.5)))
    bars = ax.barh(range(len(names)), counts,
                   color=[COLORS["primary"]] * len(counts),
                   alpha=0.8, edgecolor="white")

    # 颜色渐变
    norm = plt.Normalize(min(counts), max(counts))
    cmap = plt.cm.Blues
    for bar, count in zip(bars, counts):
        bar.set_facecolor(cmap(norm(count) * 0.6 + 0.3))

    # 百分比标注
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(count + 0.2, i, f"{count} ({count/total*100:.1f}%)",
                va="center", fontsize=9)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("出现次数", fontsize=11)
    ax.set_title(f"关键传播模式频率分布 — {dataset_name}", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    ax.set_facecolor("#FAFAFA")
    ax.invert_yaxis()   # 最高频在上

    plt.tight_layout()
    path = save_path or str(OUTPUT_DIR / f"fig5_key_patterns_{dataset_name}.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"✅ 图5 关键传播模式 → {path}")
    return path


# ═══════════════════════════════════════════════════
# 6. 多数据集指标对比雷达图
# ═══════════════════════════════════════════════════

def plot_multi_dataset_radar(
    results_by_dataset: dict, save_path: str = None
):
    """
    绘制多数据集在各指标上的雷达图（蜘蛛图）
    """
    # 指标轴（越大越好需要取反）
    axes_config = [
        ("acc@0.5",  "准确率@0.5",   True,  "acc_0.5"),
        ("acc@1.0",  "准确率@1.0",   True,  "acc_1.0"),
        ("1-MAE",    "1-MAE_log",   True,  None),       # 需要计算
        ("r",        "Pearson r",   True,  "pearson_r"),
        ("R²",       "R²",          True,  "r2"),
        ("低偏差",   "低偏差",       True,  None),       # 1-|bias_log|
    ]

    datasets = list(results_by_dataset.keys())
    n_axes   = len(axes_config)
    angles   = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles  += angles[:1]   # 闭合

    fig, ax = plt.subplots(figsize=(7, 7),
                           subplot_kw=dict(polar=True))

    for dataset_name in datasets:
        m = results_by_dataset[dataset_name].get("metrics", {})
        vals = []
        for _, _, higher_is_better, key in axes_config:
            if key == "acc_0.5":
                v = float(m.get("acc_0.5", 0.5))
            elif key == "acc_1.0":
                v = float(m.get("acc_1.0", 0.6))
            elif key == "pearson_r":
                v = max(0.0, float(m.get("pearson_r", 0.0)))
            elif key == "r2":
                v = max(0.0, float(m.get("r2", 0.0)))
            elif key is None and _ == "1-MAE":
                mae = float(m.get("mae_log", 1.0))
                v   = max(0.0, 1.0 - mae)
            elif key is None and _ == "低偏差":
                bias = abs(float(m.get("bias_log", 0.0)))
                v    = max(0.0, 1.0 - min(bias, 1.0))
            else:
                v = 0.5
            vals.append(float(np.clip(v, 0, 1)))

        vals += vals[:1]
        color = DATASET_COLORS.get(dataset_name, "#666666")
        ax.plot(angles, vals, "o-", color=color, lw=2, label=dataset_name)
        ax.fill(angles, vals, color=color, alpha=0.1)

    ax.set_thetagrids(np.degrees(angles[:-1]),
                      [c[1] for c in axes_config], fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("多数据集性能对比雷达图", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)

    path = save_path or str(OUTPUT_DIR / "fig6_multi_dataset_radar.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"✅ 图6 多数据集雷达图 → {path}")
    return path


# ═══════════════════════════════════════════════════
# 7. 特征重要性条形图
# ═══════════════════════════════════════════════════

def plot_feature_importance(
    top_features: list, dataset_name: str = "", save_path: str = None
):
    """
    绘制特征重要性横向条形图（Top-N）
    """
    if not top_features:
        print("⚠️  无特征重要性数据，跳过")
        return None

    names  = [f["feature"] for f in top_features[:15]]
    scores = [f["importance"] for f in top_features[:15]]

    fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.55 + 1.5)))

    norm   = plt.Normalize(min(scores), max(scores))
    colors = [plt.cm.viridis(norm(s)) for s in scores]

    bars = ax.barh(range(len(names)), scores, color=colors,
                   alpha=0.85, edgecolor="white")

    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(score + max(scores) * 0.01, i, f"{score:.4f}",
                va="center", fontsize=8.5)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("重要性得分", fontsize=11)
    ax.set_title(f"Top 特征重要性 — {dataset_name}", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    ax.set_facecolor("#FAFAFA")
    ax.invert_yaxis()

    plt.tight_layout()
    path = save_path or str(OUTPUT_DIR / f"fig7_feature_importance_{dataset_name}.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"✅ 图7 特征重要性 → {path}")
    return path


# ═══════════════════════════════════════════════════
# 8. 综合仪表板（论文结果总览）
# ═══════════════════════════════════════════════════

def plot_dashboard(report: dict, dataset_name: str = "", save_path: str = None):
    """
    一页综合仪表板：散点图 + 误差分布 + 注意力图 + 指标摘要
    """
    targets     = report.get("raw_targets", [])
    predictions = report.get("raw_predictions", [])
    metrics     = report.get("metrics", {})
    pattern_reps = report.get("test_reports", [])

    if not targets or not predictions:
        print("⚠️  无原始预测数据，跳过仪表板")
        return None

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    log_true = np.log1p(np.array(targets, dtype=float))
    log_pred = np.log1p(np.maximum(np.array(predictions, dtype=float), 0))
    errors   = log_pred - log_true

    # ── 左上：散点图
    ax1 = fig.add_subplot(gs[0, 0])
    sc  = ax1.scatter(log_true, log_pred, c=np.abs(errors),
                      cmap="RdYlGn_r", s=60, alpha=0.75, edgecolors="white", lw=0.5)
    lim = [min(log_true.min(), log_pred.min()) - 0.1,
           max(log_true.max(), log_pred.max()) + 0.1]
    ax1.plot(lim, lim, "k--", lw=1.5, alpha=0.5)
    ax1.fill_between(lim, [x-0.5 for x in lim], [x+0.5 for x in lim],
                     alpha=0.08, color="green")
    ax1.set_xlabel("真实值 (log)", fontsize=9)
    ax1.set_ylabel("预测值 (log)", fontsize=9)
    ax1.set_title("预测散点图", fontsize=11, fontweight="bold")
    ax1.grid(True, alpha=0.25)

    # ── 中上：误差直方图
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(errors, bins=20, color=COLORS["primary"], alpha=0.75, edgecolor="white")
    ax2.axvline(0, color="black", ls="-", lw=1.5)
    ax2.axvline(np.mean(errors), color="red", ls="--", lw=2,
                label=f"偏差={np.mean(errors):+.3f}")
    ax2.set_xlabel("预测误差", fontsize=9)
    ax2.set_ylabel("频次", fontsize=9)
    ax2.set_title("误差分布", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.25)

    # ── 右上：指标摘要文字
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis("off")
    metric_text = (
        f"📊 性能指标摘要\n"
        f"{'─'*28}\n"
        f"MAE (log):    {metrics.get('mae_log', '?'):.4f}\n"
        f"RMSE (log):   {metrics.get('rmse_log', '?'):.4f}\n"
        f"Pearson r:    {metrics.get('pearson_r', '?'):.4f}\n"
        f"R²:           {metrics.get('r2', '?'):.4f}\n"
        f"acc@0.5:      {metrics.get('acc_0.5', '?'):.2%}\n"
        f"acc@1.0:      {metrics.get('acc_1.0', '?'):.2%}\n"
        f"MAPE:         {metrics.get('mape', '?'):.4f}\n"
        f"MSLE:         {metrics.get('msle', '?'):.4f}\n"
        f"偏差 (log):   {metrics.get('bias_log', '?'):+.4f}\n"
        f"样本量 (test):{metrics.get('n', '?')}\n"
    )
    ax3.text(0.05, 0.95, metric_text, transform=ax3.transAxes,
             fontsize=10, va="top", ha="left", family="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#EEF4FF", alpha=0.9))

    # ── 左下：规模分组误差箱线图
    ax4 = fig.add_subplot(gs[1, 0])
    y_true = np.array(targets, dtype=float)
    bins_cfg = [(0, 10, "<10"), (10, 50, "10-50"), (50, 200, "50-200"), (200, 1e9, "≥200")]
    groups, glabels = [], []
    for lo, hi, lbl in bins_cfg:
        mask = (y_true >= lo) & (y_true < hi)
        if mask.sum() >= 2:
            groups.append(errors[mask])
            glabels.append(lbl)
    if groups:
        ax4.boxplot(groups, patch_artist=True,
                    medianprops=dict(color="red", lw=2),
                    boxprops=dict(facecolor=COLORS["primary"], alpha=0.5))
        ax4.axhline(0, color="black", ls="--", lw=1, alpha=0.5)
        ax4.set_xticklabels(glabels, fontsize=9)
        ax4.set_ylabel("误差 (log)", fontsize=9)
        ax4.set_title("分组误差", fontsize=11, fontweight="bold")
        ax4.grid(True, alpha=0.25, axis="y")

    # ── 中下：注意力均值柱状图
    ax5 = fig.add_subplot(gs[1, 1])
    all_weights = {}
    for rep in pattern_reps:
        for s in rep.get("top_attention_slices", []):
            idx = s["slice"]
            all_weights.setdefault(idx, []).append(s["weight"])
    if all_weights:
        sorted_slices = sorted(all_weights.keys())
        means  = [np.mean(all_weights[i]) for i in sorted_slices]
        stds   = [np.std(all_weights[i]) if len(all_weights[i]) > 1 else 0 for i in sorted_slices]
        ax5.bar(sorted_slices, means, yerr=stds,
                color=COLORS["teal"], alpha=0.75, capsize=3, edgecolor="white")
        ax5.set_xlabel("时间片", fontsize=9)
        ax5.set_ylabel("平均注意力权重", fontsize=9)
        ax5.set_title("时间注意力分布", fontsize=11, fontweight="bold")
        ax5.grid(True, alpha=0.25, axis="y")

    # ── 右下：传播模式频率
    ax6 = fig.add_subplot(gs[1, 2])
    from collections import Counter
    pat_counter = Counter()
    for rep in pattern_reps:
        for p in rep.get("patterns", []):
            pat_counter[p.get("pattern", "?")] += 1
    if pat_counter:
        top_pats = pat_counter.most_common(6)
        pnames = [p[0][:15] for p in top_pats]
        pcounts = [p[1] for p in top_pats]
        ax6.barh(range(len(pnames)), pcounts,
                 color=COLORS["purple"], alpha=0.75, edgecolor="white")
        ax6.set_yticks(range(len(pnames)))
        ax6.set_yticklabels(pnames, fontsize=8)
        ax6.set_xlabel("频次", fontsize=9)
        ax6.set_title("传播模式", fontsize=11, fontweight="bold")
        ax6.invert_yaxis()
        ax6.grid(True, alpha=0.25, axis="x")

    fig.suptitle(f"DyGNN 级联预测结果仪表板 — {dataset_name}",
                 fontsize=14, fontweight="bold")

    path = save_path or str(OUTPUT_DIR / f"fig0_dashboard_{dataset_name}.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"✅ 综合仪表板 → {path}")
    return path


# ═══════════════════════════════════════════════════
# 主入口：从 JSON 报告生成所有图表
# ═══════════════════════════════════════════════════

def generate_all_from_report(report_path: str, dataset_name: str = "wikipedia"):
    """
    读取 JSON 报告，生成全套图表
    """
    print(f"\n{'='*60}")
    print(f"  图表生成器 — 数据集: {dataset_name}")
    print(f"  报告文件: {report_path}")
    print(f"{'='*60}\n")

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    targets     = report.get("raw_targets", [])
    predictions = report.get("raw_predictions", [])
    metrics     = report.get("metrics", {})
    pat_reports = report.get("test_reports", [])
    top_features= report.get("top_features", [])

    generated = []

    if targets and predictions:
        generated.append(plot_pred_vs_true(targets, predictions, dataset_name, metrics))
        generated.append(plot_error_distribution(targets, predictions, dataset_name))
        generated.append(plot_dashboard(report, dataset_name))

    if pat_reports:
        generated.append(plot_attention_heatmap(pat_reports, dataset_name))
        generated.append(plot_key_patterns(pat_reports, dataset_name))

    if top_features:
        generated.append(plot_feature_importance(top_features, dataset_name))

    print(f"\n📁 所有图表已保存至: {OUTPUT_DIR.resolve()}")
    return [p for p in generated if p]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DyGNN 图表生成器")
    parser.add_argument("--json",    default="outputs/wikipedia_report_final.json")
    parser.add_argument("--dataset", default="wikipedia")
    parser.add_argument("--outdir",  default="outputs/figures")
    args = parser.parse_args()

    OUTPUT_DIR = Path(args.outdir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generate_all_from_report(args.json, args.dataset)
