import json
from pathlib import Path


def main() -> None:
    wiki = json.loads(Path("outputs/wikipedia_model_compare.json").read_text(encoding="utf-8"))
    reddit = json.loads(Path("outputs/reddit_model_compare.json").read_text(encoding="utf-8"))

    summary = {
        "datasets": [wiki, reddit],
        "highlights": {
            "wikipedia_better_mape_model": wiki["improvement"]["better_model_by_mape"],
            "reddit_better_mape_model": reddit["improvement"]["better_model_by_mape"],
            "wikipedia_runtime_gap_seconds": round(
                wiki["dgnn"]["runtime_seconds"] - wiki["baseline"]["runtime_seconds"], 3
            ),
            "reddit_runtime_gap_seconds": round(
                reddit["dgnn"]["runtime_seconds"] - reddit["baseline"]["runtime_seconds"], 3
            ),
        },
    }

    out_json = Path("outputs/model_evaluation_summary.json")
    out_md = Path("outputs/model_evaluation_summary.md")
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(build_markdown(wiki, reddit), encoding="utf-8")
    print(f"评价总表已输出到: {out_json}")


def build_markdown(wiki: dict, reddit: dict) -> str:
    return "\n".join(
        [
            "# 模型准确性与性能评价",
            "",
            "## 评价指标",
            "- 准确性指标：MAE、RMSE、MAPE",
            "- 性能指标：运行时间（Runtime）",
            "",
            "## Wikipedia 数据集",
            f"- Baseline：MAE={wiki['baseline']['metrics']['mae']}，RMSE={wiki['baseline']['metrics']['rmse']}，MAPE={wiki['baseline']['metrics']['mape']}，Runtime={wiki['baseline']['runtime_seconds']}s",
            f"- DGNN：MAE={wiki['dgnn']['metrics']['mae']}，RMSE={wiki['dgnn']['metrics']['rmse']}，MAPE={wiki['dgnn']['metrics']['mape']}，Runtime={wiki['dgnn']['runtime_seconds']}s",
            f"- 结论：Wikipedia 上 Baseline 在 MAE、RMSE、MAPE 三项指标上均优于 DGNN。",
            "",
            "## Reddit 数据集",
            f"- Baseline：MAE={reddit['baseline']['metrics']['mae']}，RMSE={reddit['baseline']['metrics']['rmse']}，MAPE={reddit['baseline']['metrics']['mape']}，Runtime={reddit['baseline']['runtime_seconds']}s",
            f"- DGNN：MAE={reddit['dgnn']['metrics']['mae']}，RMSE={reddit['dgnn']['metrics']['rmse']}，MAPE={reddit['dgnn']['metrics']['mape']}，Runtime={reddit['dgnn']['runtime_seconds']}s",
            f"- 结论：Reddit 上 DGNN 在 MAE 与 RMSE 上优于 Baseline，但 Baseline 在 MAPE 上更优。",
            "",
            "## 综合分析",
            "- DGNN 在表达传播过程的结构演化与时序依赖方面更贴近论文题目设定。",
            "- 基线模型训练速度更快，并在 Wikipedia 上表现出更稳的整体预测精度。",
            "- DGNN 在 Reddit 这类重尾更强的数据上改善了绝对误差，但其相对误差仍偏高，说明还需要继续优化损失函数与训练策略。",
        ]
    )


if __name__ == "__main__":
    main()
