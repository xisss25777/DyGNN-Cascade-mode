import argparse
import json
import sys
import os
import time
from pathlib import Path

# 添加父目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cascade_model.dataset_profiles import get_baseline_config, get_dgnn_config, load_dataset
from cascade_model import run_dgnn_pipeline
from cascade_model.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="比较 DGNN 与动态图特征基线模型")
    parser.add_argument("--dataset", choices=["wikipedia", "reddit", "cascade"], required=True)
    parser.add_argument("--input", help="可选，显式指定输入文件路径")
    parser.add_argument("--output", help="比较结果输出路径", default=None)
    args = parser.parse_args()

    dataset_name, cascades = load_dataset(args.dataset, args.input)
    baseline_config = get_baseline_config(dataset_name)
    dgnn_config = get_dgnn_config(dataset_name)
    output_path = Path(args.output or f"outputs/{dataset_name}_model_compare.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    baseline_start = time.perf_counter()
    baseline_report = run_pipeline(cascades, baseline_config)
    baseline_time = time.perf_counter() - baseline_start

    dgnn_start = time.perf_counter()
    dgnn_report = run_dgnn_pipeline(cascades, dgnn_config)
    dgnn_time = time.perf_counter() - dgnn_start

    comparison = {
        "dataset": dataset_name,
        "baseline": {
            "model": "dynamic_feature_baseline",
            "config": baseline_report["config"],
            "runtime_seconds": round(baseline_time, 3),
            "metrics": baseline_report["metrics"],
            "top_features": baseline_report["top_features"][:8],
        },
        "dgnn": {
            "model": "dgnn_gru_attention",
            "config": dgnn_report["config"],
            "runtime_seconds": round(dgnn_time, 3),
            "metrics": dgnn_report["metrics"],
            "top_features": dgnn_report["top_features"][:8],
            "top_channels": dgnn_report.get("top_channels", []),
        },
    }
    comparison["improvement"] = compute_improvement(comparison["baseline"]["metrics"], comparison["dgnn"]["metrics"])

    output_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = output_path.with_suffix(".md")
    md_path.write_text(build_markdown(comparison, output_path), encoding="utf-8")

    print(f"数据集: {dataset_name}")
    print("模型对比结果:")
    print(
        f"  - Baseline: MAE={comparison['baseline']['metrics']['mae']}, "
        f"RMSE={comparison['baseline']['metrics']['rmse']}, "
        f"MAPE={comparison['baseline']['metrics']['mape']}, "
        f"time={comparison['baseline']['runtime_seconds']}s"
    )
    print(
        f"  - DGNN: MAE={comparison['dgnn']['metrics']['mae']}, "
        f"RMSE={comparison['dgnn']['metrics']['rmse']}, "
        f"MAPE={comparison['dgnn']['metrics']['mape']}, "
        f"time={comparison['dgnn']['runtime_seconds']}s"
    )
    print(
        f"  - MAPE 改变量: {comparison['improvement']['mape_change']} "
        f"({comparison['improvement']['better_model_by_mape']})"
    )
    print(f"JSON 结果已输出到: {output_path}")
    print(f"摘要结果已输出到: {md_path}")


def compute_improvement(baseline: dict, dgnn: dict) -> dict:
    def delta(key: str) -> float:
        return round(dgnn[key] - baseline[key], 4)

    return {
        "mae_change": delta("mae"),
        "rmse_change": delta("rmse"),
        "mape_change": delta("mape"),
        "better_model_by_mape": "dgnn" if dgnn["mape"] < baseline["mape"] else "baseline",
        "better_model_by_mae": "dgnn" if dgnn["mae"] < baseline["mae"] else "baseline",
        "better_model_by_rmse": "dgnn" if dgnn["rmse"] < baseline["rmse"] else "baseline",
    }


def build_markdown(comparison: dict, output_path: Path) -> str:
    baseline = comparison["baseline"]
    dgnn = comparison["dgnn"]
    improvement = comparison["improvement"]
    lines = [
        f"# {comparison['dataset']} 数据集模型评价与对比",
        "",
        "## 对比对象",
        "- 动态图特征基线模型：早期动态图特征 + 线性/KNN 融合回归",
        "- DGNN 模型：GraphConv + GRU + Attention",
        "",
        "## 评价指标",
        f"- Baseline: MAE={baseline['metrics']['mae']}, RMSE={baseline['metrics']['rmse']}, MAPE={baseline['metrics']['mape']}, Runtime={baseline['runtime_seconds']}s",
        f"- DGNN: MAE={dgnn['metrics']['mae']}, RMSE={dgnn['metrics']['rmse']}, MAPE={dgnn['metrics']['mape']}, Runtime={dgnn['runtime_seconds']}s",
        "",
        "## 结果比较",
        f"- MAE 改变量（DGNN - Baseline）：{improvement['mae_change']}",
        f"- RMSE 改变量（DGNN - Baseline）：{improvement['rmse_change']}",
        f"- MAPE 改变量（DGNN - Baseline）：{improvement['mape_change']}",
        f"- 按 MAPE 更优的模型：{improvement['better_model_by_mape']}",
        "",
        "## 基线模型 Top 特征",
    ]
    for item in baseline["top_features"]:
        lines.append(f"- {item['feature']}: {item['importance']}")

    lines.extend(["", "## DGNN 关键注意力/通道"])
    for item in dgnn["top_features"][:5]:
        lines.append(f"- {item['feature']}: {item['importance']}")
    for item in dgnn.get("top_channels", [])[:5]:
        lines.append(f"- channel {item['feature']}: {item['importance']}")

    lines.extend(
        [
            "",
            "## 文件说明",
            f"- JSON 结果文件：{output_path.name}",
            f"- 当前摘要文件：{output_path.with_suffix('.md').name}",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()