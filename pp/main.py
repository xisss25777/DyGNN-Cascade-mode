import argparse
import json
import sys
import os
from pathlib import Path

# 添加父目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cascade_model.data import write_sample_csv
from cascade_model.dataset_profiles import load_dataset_and_config
from cascade_model import run_dgnn_pipeline
from cascade_model.pipeline import run_pipeline
from cascade_model.cache_utils import cache_manager, benchmark_cache_performance


def main() -> None:
    parser = argparse.ArgumentParser(description="信息级联动态图建模框架")
    parser.add_argument("--input", help="事件级联 CSV 文件路径")
    parser.add_argument(
        "--dataset",
        choices=["wikipedia", "reddit", "cascade", "synthetic"],
        required=True,
        help="当前运行的数据集类型；真实实验请显式选择 wikipedia 或 reddit",
    )
    parser.add_argument("--output", default="outputs/report.json", help="结果输出路径")
    parser.add_argument("--save-sample", action="store_true", help="将模拟数据保存为 sample_data/cascades.csv")
    parser.add_argument("--benchmark", action="store_true", help="运行缓存性能基准测试")
    args = parser.parse_args()

    dataset_name, cascades, config = load_dataset_and_config(args.input, args.dataset)

    # 添加数据集名称到配置
    config.dataset_name = dataset_name

    # 生成缓存键
    cache_key = f"{dataset_name}_{config.epochs}_epochs"

    # 检查是否已经有缓存的训练结果
    cached_result = cache_manager.get_training_result(cache_key)
    if cached_result:
        print(f"使用缓存的训练结果: {cache_key}")
        report = cached_result
    else:
        # 运行性能基准测试（如果指定）
        if args.benchmark:
            print("运行缓存性能基准测试...")
            benchmark_cache_performance(cascades[:10], config, iterations=3)

        # 运行训练
        if args.save_sample:
            write_sample_csv("sample_data/cascades.csv", cascades)

        if dataset_name in {"wikipedia", "reddit"}:
            report = run_dgnn_pipeline(cascades, config)
        else:
            report = run_pipeline(cascades, config)

        # 保存结果到缓存
        cache_manager.save_training_result(cache_key, report)
        print(f"训练结果已缓存: {cache_key}")

    report["dataset"] = dataset_name
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path = output_path.with_suffix(".md")
    summary_path.write_text(build_summary(report, output_path), encoding="utf-8")

    print(f"数据集: {dataset_name}")
    print(f"样本数: {report['sample_count']}")
    print(f"特征数: {report['feature_count']}")
    print(f"融合权重 alpha: {report.get('blend_alpha', 'N/A')}")
    print(
        "评估指标: "
        f"MAE={report['metrics']['mae']}, "
        f"RMSE={report['metrics']['rmse']}, "
        f"MAPE={report['metrics']['mape']}"
    )
    print("Top 3 特征:")
    for item in report["top_features"][:3]:
        print(f"  - {item['feature']}: {item['importance']}")
    print("代表性测试结果:")
    for item in report["test_reports"][:3]:
        main_pattern = item["patterns"][0]["pattern"] if item["patterns"] else "无"
        print(f"  - cascade {item['cascade_id']}: pred={item['prediction']}, pattern={main_pattern}")
    print(f"JSON 结果已输出到: {output_path}")
    print(f"摘要结果已输出到: {summary_path}")

def build_summary(report: dict, output_path: Path) -> str:
    lines = [
        f"# {report['dataset']} 数据集实验摘要",
        "",
        "## 核心结果",
        f"- 样本数：{report['sample_count']}",
        f"- 特征数：{report['feature_count']}",
        f"- 融合权重 alpha：{report.get('blend_alpha', 'N/A')}",
        f"- MAE：{report['metrics']['mae']}",
        f"- RMSE：{report['metrics']['rmse']}",
        f"- MAPE：{report['metrics']['mape']}",
        "",
        "## Top 特征",
    ]
    for item in report["top_features"][:8]:
        lines.append(f"- {item['feature']}: {item['importance']}")

    lines.extend(["", "## 代表性测试样本"])
    for item in report["test_reports"][:5]:
        main_pattern = item["patterns"][0]["pattern"] if item["patterns"] else "无"
        lines.append(
            f"- 级联 {item['cascade_id']}：预测规模 {item['prediction']}，"
            f"主导模式 {main_pattern}，"
            f"删除实验 delta={item['deletion_test']['delta']}"
        )

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