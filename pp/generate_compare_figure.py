import json
import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="生成模型评价对比图")
    parser.add_argument("--input", default="outputs/model_evaluation_summary.json", help="评价汇总 JSON 路径")
    args = parser.parse_args()

    report = json.loads(Path(args.input).read_text(encoding="utf-8"))
    output_path = Path("outputs/figures/dual_dataset_compare.svg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_svg(report), encoding="utf-8")
    print(f"对比图已输出到: {output_path}")


def build_svg(report) -> str:
    datasets = report["datasets"]
    wikipedia = next(item for item in datasets if item["dataset"] == "wikipedia")
    reddit = next(item for item in datasets if item["dataset"] == "reddit")
    metric_names = [("mae", "MAE"), ("rmse", "RMSE"), ("mape", "MAPE")]
    max_metric = max(
        max(wikipedia["baseline"]["metrics"][key], wikipedia["dgnn"]["metrics"][key], reddit["baseline"]["metrics"][key], reddit["dgnn"]["metrics"][key])
        for key, _ in metric_names
    )

    bars = []
    for idx, (key, label) in enumerate(metric_names):
        y = 240 + idx * 150
        w1 = 420 * wikipedia["baseline"]["metrics"][key] / max_metric
        w2 = 420 * wikipedia["dgnn"]["metrics"][key] / max_metric
        w3 = 420 * reddit["baseline"]["metrics"][key] / max_metric
        w4 = 420 * reddit["dgnn"]["metrics"][key] / max_metric
        bars.append(f'<text x="130" y="{y-22}" font-size="24" font-weight="700" fill="#0f172a">{label}</text>')
        bars.append(f'<rect x="130" y="{y}" width="{w1:.1f}" height="18" rx="9" fill="#93c5fd"/>')
        bars.append(f'<text x="{130+w1+15:.1f}" y="{y+14}" font-size="16" fill="#334155">Wiki-B {wikipedia["baseline"]["metrics"][key]:.4f}</text>')
        bars.append(f'<rect x="130" y="{y+24}" width="{w2:.1f}" height="18" rx="9" fill="#2563eb"/>')
        bars.append(f'<text x="{130+w2+15:.1f}" y="{y+38}" font-size="16" fill="#334155">Wiki-D {wikipedia["dgnn"]["metrics"][key]:.4f}</text>')
        bars.append(f'<rect x="130" y="{y+52}" width="{w3:.1f}" height="18" rx="9" fill="#99f6e4"/>')
        bars.append(f'<text x="{130+w3+15:.1f}" y="{y+66}" font-size="16" fill="#334155">Reddit-B {reddit["baseline"]["metrics"][key]:.4f}</text>')
        bars.append(f'<rect x="130" y="{y+76}" width="{w4:.1f}" height="18" rx="9" fill="#0f766e"/>')
        bars.append(f'<text x="{130+w4+15:.1f}" y="{y+90}" font-size="16" fill="#334155">Reddit-D {reddit["dgnn"]["metrics"][key]:.4f}</text>')

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="900" viewBox="0 0 1400 900">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#f8fbff"/>
      <stop offset="100%" stop-color="#f8fafc"/>
    </linearGradient>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="16" stdDeviation="24" flood-color="#94a3b8" flood-opacity="0.22"/>
    </filter>
  </defs>
  <rect width="100%" height="100%" fill="url(#bg)"/>
  <rect x="60" y="50" width="1280" height="800" rx="30" fill="rgba(255,255,255,0.85)" filter="url(#shadow)"/>
  <text x="100" y="110" font-size="42" font-weight="700" fill="#0f172a">模型准确性与性能对比</text>
  <text x="100" y="145" font-size="21" fill="#64748b">Wikipedia 与 Reddit 上 Baseline / DGNN 的实验结果比较</text>
  <circle cx="900" cy="120" r="9" fill="#93c5fd"/><text x="920" y="127" font-size="18" fill="#334155">Wiki-Baseline</text>
  <circle cx="1100" cy="120" r="9" fill="#2563eb"/><text x="1120" y="127" font-size="18" fill="#334155">Wiki-DGNN</text>
  <circle cx="1260" cy="120" r="9" fill="#99f6e4"/><text x="1280" y="127" font-size="18" fill="#334155">Reddit-Baseline</text>
  <circle cx="900" cy="155" r="9" fill="#0f766e"/><text x="920" y="162" font-size="18" fill="#334155">Reddit-DGNN</text>
  {''.join(bars)}
  <text x="100" y="760" font-size="24" font-weight="700" fill="#0f172a">摘要结论</text>
  <text x="100" y="800" font-size="21" fill="#334155">Wikipedia 上更优 MAPE 模型：{report['highlights']['wikipedia_better_mape_model']}</text>
  <text x="100" y="835" font-size="21" fill="#334155">Reddit 上更优 MAPE 模型：{report['highlights']['reddit_better_mape_model']}</text>
</svg>"""


if __name__ == "__main__":
    main()
