"""
多数据集批量实验入口
运行: python run_all_experiments.py

功能:
  - 在 Wikipedia / Reddit / Enron / MOOC 四个数据集上运行完整 pipeline
  - 生成每个数据集的 JSON 报告和可视化图表
  - 输出多数据集对比汇总表

用法:
  python run_all_experiments.py                    # 跑全部数据集
  python run_all_experiments.py --dataset wikipedia # 只跑一个
  python run_all_experiments.py --quick             # 快速模式(少epoch)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from cascade_model.dataset_profiles import load_dataset_and_config
from cascade_model.dgnn import run_dgnn_pipeline
from cascade_model.enhanced_evaluation import (
    compute_all_metrics, error_distribution, diagnose_bias, format_comparison_table
)

# 导入图表生成器（相对路径）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_all_figures import (
    generate_all_from_report, plot_multi_dataset_radar, OUTPUT_DIR
)


OUTPUT_BASE = Path("outputs")
OUTPUT_BASE.mkdir(exist_ok=True)


def run_single_dataset(dataset_name: str, quick: bool = False) -> dict:
    """
    运行单个数据集的完整实验 pipeline
    """
    print(f"\n{'━'*65}")
    print(f"  🚀 数据集: {dataset_name.upper()}")
    print(f"{'━'*65}")

    t_start = time.time()

    # ── 加载数据
    dataset_name_out, cascades, config = load_dataset_and_config(None, dataset_name)

    if quick:
        config.epochs    = 50
        config.patience  = 20
        print(f"  [快速模式] epochs={config.epochs}")

    print(f"  级联总数: {len(cascades)}, 配置 epochs={config.epochs}")

    # ── 运行 DGNN Pipeline
    report = run_dgnn_pipeline(cascades, config)
    report["dataset"] = dataset_name

    # ── 保存 JSON
    out_json = OUTPUT_BASE / f"{dataset_name}_report.json"
    out_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ── 生成摘要 Markdown
    out_md = out_json.with_suffix(".md")
    out_md.write_text(
        _build_markdown_summary(report, dataset_name), encoding="utf-8"
    )

    elapsed = time.time() - t_start
    print(f"\n  ✅ 完成，耗时 {elapsed:.1f}s")
    print(f"     JSON  → {out_json}")
    print(f"     报告  → {out_md}")

    # ── 生成图表
    generate_all_from_report(str(out_json), dataset_name)

    return report


def run_all_experiments(datasets=None, quick=False):
    """
    批量运行所有数据集实验
    """
    if datasets is None:
        datasets = ["wikipedia", "reddit"]   # 默认跑 2 个，Enron/MOOC 需要准备CSV

    all_results = {}
    for dataset_name in datasets:
        try:
            report = run_single_dataset(dataset_name, quick=quick)
            all_results[dataset_name] = report
        except Exception as e:
            print(f"\n  ❌ {dataset_name} 失败: {e}")
            import traceback
            traceback.print_exc()

    # ── 多数据集对比
    if len(all_results) > 1:
        print(f"\n\n{'═'*65}")
        print("  📊 多数据集性能对比")
        print('═'*65)

        table = format_comparison_table(
            all_results,
            metric_keys=["mae_log", "rmse_log", "pearson_r", "acc_0.5", "r2", "bias_log", "n"]
        )
        print(table)

        # 保存对比表
        compare_path = OUTPUT_BASE / "comparison_table.md"
        compare_path.write_text(f"# 多数据集性能对比\n\n{table}\n", encoding="utf-8")
        print(f"\n对比表 → {compare_path}")

        # 雷达图
        try:
            plot_multi_dataset_radar(all_results)
        except Exception as e:
            print(f"  ⚠️ 雷达图生成失败: {e}")

    return all_results


def _build_markdown_summary(report: dict, dataset_name: str) -> str:
    m   = report.get("metrics", {})
    dist = report.get("error_distribution", {})
    sug  = report.get("bias_suggestions", [])

    lines = [
        f"# {dataset_name.capitalize()} 数据集实验报告",
        "",
        "## 一、实验配置",
        f"- 样本数: {report.get('sample_count', '?')}",
        f"- 特征维度: {report.get('feature_count', '?')}",
        f"- 模型类型: {report.get('config', {}).get('model_type', 'dgnn_gru_attention')}",
        f"- 训练轮数: {report.get('config', {}).get('epochs', '?')}",
        f"- 学习率: {report.get('config', {}).get('learning_rate', '?')}",
        "",
        "## 二、核心评估指标",
        "",
        "### 对数尺度（与训练目标一致）",
        f"| 指标 | 数值 |",
        f"|------|------|",
        f"| MAE_log | {m.get('mae_log', '?')} |",
        f"| RMSE_log | {m.get('rmse_log', '?')} |",
        f"| 系统偏差 (log) | {m.get('bias_log', '?')} |",
        "",
        "### 原始尺度",
        f"| 指标 | 数值 |",
        f"|------|------|",
        f"| Pearson r | {m.get('pearson_r', '?')} |",
        f"| R² | {m.get('r2', '?')} |",
        f"| MAPE | {m.get('mape', '?')} |",
        f"| MSLE | {m.get('msle', '?')} |",
        f"| acc@0.5 | {m.get('acc_0.5', '?')} |",
        f"| acc@1.0 | {m.get('acc_1.0', '?')} |",
        "",
        "## 三、误差分布分析",
        f"- 高估比例: {dist.get('overestimate_ratio', '?')}",
        f"- 低估比例: {dist.get('underestimate_ratio', '?')}",
        f"- 误差中位数: {dist.get('error_median', '?')}",
    ]

    # 分组误差
    by_group = dist.get("by_group", {})
    if by_group:
        lines += ["", "### 分规模组误差"]
        lines += ["| 规模组 | 样本数 | MAE_log | 偏差 |",
                  "|--------|--------|---------|------|"]
        for g_name, g_stats in by_group.items():
            lines.append(
                f"| {g_name} | {g_stats.get('count', 0)} | "
                f"{g_stats.get('mae_log', '?')} | {g_stats.get('bias_log', '?')} |"
            )

    lines += ["", "## 四、偏差诊断与改进建议", ""]
    for s in sug:
        lines.append(f"- {s}")

    lines += ["", "## 五、Top 关键传播特征", ""]
    for f in report.get("top_features", [])[:8]:
        lines.append(f"- {f['feature']}: {f['importance']}")

    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DyGNN 多数据集实验")
    parser.add_argument("--dataset", choices=["wikipedia", "reddit", "enron", "mooc", "all"],
                        default="all", help="要运行的数据集")
    parser.add_argument("--quick", action="store_true", help="快速模式（少轮次）")
    args = parser.parse_args()

    if args.dataset == "all":
        datasets_to_run = ["wikipedia", "reddit"]
    else:
        datasets_to_run = [args.dataset]

    run_all_experiments(datasets=datasets_to_run, quick=args.quick)
