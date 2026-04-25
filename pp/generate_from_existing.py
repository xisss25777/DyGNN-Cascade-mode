"""
从现有 wikipedia_report_final.json 生成所有图表
包括补充 raw_targets / raw_predictions（从原始数据推算）
"""
import json, sys, os, math, numpy as np
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from generate_all_figures import (
    plot_pred_vs_true, plot_error_distribution, plot_attention_heatmap,
    plot_key_patterns, plot_feature_importance, plot_dashboard,
    plot_cascade_size_distribution, OUTPUT_DIR
)
from cascade_model.enhanced_evaluation import compute_all_metrics, error_distribution

# ── 读取现有报告
with open("outputs/wikipedia_report_final.json", "r", encoding="utf-8") as f:
    report = json.load(f)

dataset_name = "wikipedia"
test_reports = report.get("test_reports", [])

# ── 从 test_reports 提取预测值
predictions = [r["prediction"] for r in test_reports]
print(f"预测值样本: {predictions[:5]}")

# ── 从 pp/sample_data/wikipedia.csv 提取真实目标值
import pandas as pd
df = pd.read_csv("sample_data/wikipedia.csv",
                 names=["user_id","item_id","timestamp","label"],
                 usecols=[0,1,2,3], skiprows=1)
cascade_sizes = df.groupby("item_id")["user_id"].nunique().to_dict()

# 对应 test_reports 的 cascade_id
targets = []
valid_preds = []
for r in test_reports:
    cid = r["cascade_id"]
    size = cascade_sizes.get(int(cid) if str(cid).isdigit() else cid)
    if size is not None:
        targets.append(float(size))
        valid_preds.append(r["prediction"])

print(f"有效匹配: {len(targets)} 个级联")
print(f"真实值范围: [{min(targets):.0f}, {max(targets):.0f}]")
print(f"预测值范围: [{min(valid_preds):.2f}, {max(valid_preds):.2f}]")

# ── 计算增强指标
metrics = compute_all_metrics(targets, valid_preds)
dist    = error_distribution(targets, valid_preds)

print("\n=== 增强评估指标 ===")
print(f"MAE_log:   {metrics['mae_log']:.4f}")
print(f"RMSE_log:  {metrics['rmse_log']:.4f}")
print(f"Pearson r: {metrics['pearson_r']:.4f}")
print(f"R2:        {metrics['r2']:.4f}")
print(f"acc@0.5:   {metrics['acc_0.5']:.2%}")
print(f"bias_log:  {metrics['bias_log']:+.4f}")
print(f"高估比例:  {dist['overestimate_ratio']:.2%}")

# ── 补充报告字段
report["raw_targets"]     = [round(t, 4) for t in targets]
report["raw_predictions"] = [round(p, 4) for p in valid_preds]
report["metrics"]         = metrics
report["error_distribution"] = dist

# ── 生成全套图表
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n=== 开始生成图表 ===")

# 1. 规模分布
plot_cascade_size_distribution(
    {"wikipedia": [float(s) for s in cascade_sizes.values() if s >= 5]},
)

# 2. 预测散点图
plot_pred_vs_true(targets, valid_preds, dataset_name, metrics)

# 3. 误差分布图
plot_error_distribution(targets, valid_preds, dataset_name)

# 4. 注意力热力图
plot_attention_heatmap(test_reports, dataset_name)

# 5. 关键传播模式
plot_key_patterns(test_reports, dataset_name)

# 6. 特征重要性
top_feats = report.get("top_features", []) + [
    {"feature": c["feature"], "importance": c["importance"]}
    for c in report.get("top_channels", [])[:6]
]
plot_feature_importance(top_feats, dataset_name)

# 7. 综合仪表板
plot_dashboard(report, dataset_name)

# ── 保存更新后的报告
with open("outputs/wikipedia_report_enriched.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print("\nDone! 所有图表已保存至 outputs/figures/")
