import json
import argparse
from pathlib import Path
from typing import Dict


WIDTH = 1600
HEIGHT = 900


def main() -> None:
    parser = argparse.ArgumentParser(description="根据实验报告生成论文图表")
    parser.add_argument("--input", default="outputs/wikipedia_result.json", help="报告 JSON 路径")
    args = parser.parse_args()

    report_path = Path(args.input)
    output_dir = Path("outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    (output_dir / "model_framework.svg").write_text(build_framework_svg(), encoding="utf-8")
    (output_dir / "result_dashboard.svg").write_text(build_dashboard_svg(report), encoding="utf-8")
    (output_dir / "feature_importance.svg").write_text(build_feature_svg(report), encoding="utf-8")

    print(f"图像已输出到: {output_dir}")


def build_framework_svg() -> str:
    boxes = [
        ("01", "动态图构建", "按 item_id 聚合传播序列\n分时间片构造快照图"),
        ("02", "多维特征提取", "结构特征、时间特征\n高维交互特征汇总"),
        ("03", "传播规模预测", "标准化 + 回归建模\n输出最终交互规模"),
        ("04", "关键模式识别", "识别快速扩散、核心驱动\n平稳扩散等模式"),
        ("05", "验证与解释", "删除实验 + 重要特征排序\n支撑论文分析"),
    ]

    box_svg = []
    start_x = 90
    y = 300
    box_w = 250
    box_h = 220
    gap = 50
    for idx, (num, title, desc) in enumerate(boxes):
        x = start_x + idx * (box_w + gap)
        box_svg.append(card(x, y, box_w, box_h, title, desc, num))
        if idx < len(boxes) - 1:
            ax = x + box_w
            bx = x + box_w + gap
            box_svg.append(arrow(ax + 8, y + box_h / 2, bx - 8, y + box_h / 2))

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">
  {defs()}
  <rect width="100%" height="100%" fill="url(#bg)"/>
  <circle cx="140" cy="110" r="180" fill="url(#glow1)" opacity="0.65"/>
  <circle cx="1350" cy="760" r="240" fill="url(#glow2)" opacity="0.55"/>
  <text x="90" y="100" font-size="42" font-weight="700" fill="#0f172a">信息级联规模预测模型框架</text>
  <text x="90" y="145" font-size="22" fill="#475569">Dynamic Cascade Modeling, Prediction, Pattern Mining, and Validation</text>
  <rect x="90" y="185" rx="18" ry="18" width="490" height="62" fill="rgba(255,255,255,0.72)" stroke="rgba(148,163,184,0.25)"/>
  <text x="120" y="225" font-size="24" font-weight="600" fill="#0f172a">真实数据适配：Wikipedia 交互流 -> 按 item_id 构造级联</text>
  {''.join(box_svg)}
  <text x="90" y="780" font-size="20" fill="#334155">建议放在论文“研究框架”或“模型构建”部分，SVG 可直接导入 Word / PPT。</text>
</svg>"""


def build_dashboard_svg(report: Dict[str, object]) -> str:
    metrics = report["metrics"]
    sample_count = report["sample_count"]
    feature_count = report["feature_count"]
    top_features = report["top_features"][:6]
    test_reports = report["test_reports"][:8]

    pattern_counts: Dict[str, int] = {}
    for item in report["test_reports"]:
        for pattern in item["patterns"]:
            name = pattern["pattern"]
            pattern_counts[name] = pattern_counts.get(name, 0) + 1
    sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:4]

    cards_svg = []
    metric_cards = [
        ("样本数", str(sample_count), "#0f766e"),
        ("特征数", str(feature_count), "#1d4ed8"),
        ("MAE", f"{metrics['mae']:.2f}", "#c2410c"),
        ("MAPE", f"{metrics['mape']:.3f}", "#7c3aed"),
    ]
    for idx, (label, value, color) in enumerate(metric_cards):
        cards_svg.append(metric_card(90 + idx * 345, 150, 300, 120, label, value, color))

    bars = []
    max_importance = max(item["importance"] for item in top_features)
    for idx, item in enumerate(top_features):
        y = 390 + idx * 54
        bar_w = 360 * item["importance"] / max_importance
        bars.append(f'<text x="110" y="{y}" font-size="19" fill="#1e293b">{safe(item["feature"])}</text>')
        bars.append(f'<rect x="370" y="{y-18}" width="390" height="22" rx="11" fill="rgba(148,163,184,0.18)"/>')
        bars.append(f'<rect x="370" y="{y-18}" width="{bar_w:.1f}" height="22" rx="11" fill="url(#barGrad)"/>')
        bars.append(f'<text x="775" y="{y}" font-size="17" text-anchor="end" fill="#334155">{item["importance"]:.2f}</text>')

    pattern_svg = []
    pattern_colors = ["#2563eb", "#0f766e", "#7c3aed", "#ea580c"]
    max_count = max(count for _, count in sorted_patterns) if sorted_patterns else 1
    for idx, (name, count) in enumerate(sorted_patterns):
        y = 395 + idx * 70
        w = 280 * count / max_count
        pattern_svg.append(f'<text x="980" y="{y}" font-size="20" fill="#1e293b">{safe(name)}</text>')
        pattern_svg.append(f'<rect x="980" y="{y+14}" width="{w:.1f}" height="16" rx="8" fill="{pattern_colors[idx % len(pattern_colors)]}" opacity="0.88"/>')
        pattern_svg.append(f'<text x="{980 + w + 18:.1f}" y="{y+28}" font-size="17" fill="#475569">{count}</text>')

    rows = []
    for idx, item in enumerate(test_reports):
        y = 675 + idx * 24
        pattern_name = item["patterns"][0]["pattern"]
        rows.append(
            f'<text x="110" y="{y}" font-size="16" fill="#334155">{item["cascade_id"]}</text>'
            f'<text x="250" y="{y}" font-size="16" fill="#0f172a">{item["prediction"]:.1f}</text>'
            f'<text x="400" y="{y}" font-size="16" fill="#475569">{safe(pattern_name)}</text>'
            f'<text x="690" y="{y}" font-size="16" fill="#475569">{item["deletion_test"]["effect_direction"]}</text>'
        )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">
  {defs()}
  <rect width="100%" height="100%" fill="url(#bg)"/>
  <rect x="60" y="55" width="1480" height="790" rx="34" fill="rgba(255,255,255,0.78)" filter="url(#shadow)"/>
  <text x="90" y="105" font-size="40" font-weight="700" fill="#0f172a">Wikipedia 真实数据实验结果总览</text>
  <text x="90" y="132" font-size="20" fill="#64748b">Prediction Report · Early Dynamic Observation Window = 6 Hours</text>
  {''.join(cards_svg)}
  <text x="90" y="330" font-size="26" font-weight="700" fill="#0f172a">关键特征重要性</text>
  {''.join(bars)}
  <text x="980" y="330" font-size="26" font-weight="700" fill="#0f172a">关键传播模式分布</text>
  {''.join(pattern_svg)}
  <text x="90" y="630" font-size="26" font-weight="700" fill="#0f172a">测试样本代表结果</text>
  <rect x="90" y="645" width="700" height="170" rx="22" fill="#f8fafc" stroke="rgba(148,163,184,0.2)"/>
  <text x="110" y="668" font-size="15" font-weight="700" fill="#64748b">级联ID</text>
  <text x="250" y="668" font-size="15" font-weight="700" fill="#64748b">预测规模</text>
  <text x="400" y="668" font-size="15" font-weight="700" fill="#64748b">主导模式</text>
  <text x="690" y="668" font-size="15" font-weight="700" fill="#64748b">删除实验方向</text>
  {''.join(rows)}
</svg>"""


def build_feature_svg(report: Dict[str, object]) -> str:
    top_features = report["top_features"][:10]
    max_value = max(item["importance"] for item in top_features)
    bars = []
    for idx, item in enumerate(top_features):
        x = 170
        y = 160 + idx * 58
        width = 980 * item["importance"] / max_value
        bars.append(f'<text x="{x}" y="{y-12}" font-size="19" fill="#0f172a">{safe(item["feature"])}</text>')
        bars.append(f'<rect x="{x}" y="{y}" width="{width:.1f}" height="24" rx="12" fill="url(#barGrad)"/>')
        bars.append(f'<text x="{x + width + 16:.1f}" y="{y+18}" font-size="17" fill="#334155">{item["importance"]:.2f}</text>')

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="860" viewBox="0 0 1400 860">
  {defs()}
  <rect width="100%" height="100%" fill="url(#bg)"/>
  <rect x="60" y="50" width="1280" height="760" rx="30" fill="rgba(255,255,255,0.82)" filter="url(#shadow)"/>
  <text x="90" y="105" font-size="40" font-weight="700" fill="#0f172a">模型特征贡献 Top 10</text>
  <text x="90" y="135" font-size="20" fill="#64748b">Feature Importance Ranking for Cascade Size Prediction</text>
  {''.join(bars)}
</svg>"""


def defs() -> str:
    return """
<defs>
  <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0%" stop-color="#f8fbff"/>
    <stop offset="45%" stop-color="#eef6ff"/>
    <stop offset="100%" stop-color="#f8fafc"/>
  </linearGradient>
  <linearGradient id="barGrad" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0%" stop-color="#0f766e"/>
    <stop offset="50%" stop-color="#2563eb"/>
    <stop offset="100%" stop-color="#7c3aed"/>
  </linearGradient>
  <radialGradient id="glow1">
    <stop offset="0%" stop-color="#60a5fa"/>
    <stop offset="100%" stop-color="rgba(96,165,250,0)"/>
  </radialGradient>
  <radialGradient id="glow2">
    <stop offset="0%" stop-color="#34d399"/>
    <stop offset="100%" stop-color="rgba(52,211,153,0)"/>
  </radialGradient>
  <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
    <feDropShadow dx="0" dy="18" stdDeviation="28" flood-color="#94a3b8" flood-opacity="0.22"/>
  </filter>
</defs>
"""


def card(x: int, y: int, w: int, h: int, title: str, desc: str, num: str) -> str:
    lines = desc.split("\n")
    text_svg = "".join(
        f'<text x="{x+26}" y="{y+122+i*30}" font-size="19" fill="#475569">{safe(line)}</text>'
        for i, line in enumerate(lines)
    )
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="28" fill="rgba(255,255,255,0.82)" '
        f'stroke="rgba(148,163,184,0.24)" filter="url(#shadow)"/>'
        f'<circle cx="{x+42}" cy="{y+40}" r="22" fill="#0f766e"/>'
        f'<text x="{x+42}" y="{y+47}" text-anchor="middle" font-size="20" font-weight="700" fill="white">{num}</text>'
        f'<text x="{x+78}" y="{y+48}" font-size="28" font-weight="700" fill="#0f172a">{safe(title)}</text>'
        f'{text_svg}'
    )


def arrow(x1: float, y1: float, x2: float, y2: float) -> str:
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#2563eb" stroke-width="5" stroke-linecap="round"/>'
        f'<polygon points="{x2},{y2} {x2-18},{y2-10} {x2-18},{y2+10}" fill="#2563eb"/>'
    )


def metric_card(x: int, y: int, w: int, h: int, label: str, value: str, color: str) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="26" fill="white" filter="url(#shadow)"/>'
        f'<rect x="{x}" y="{y}" width="{w}" height="10" rx="10" fill="{color}"/>'
        f'<text x="{x+28}" y="{y+52}" font-size="22" fill="#64748b">{safe(label)}</text>'
        f'<text x="{x+28}" y="{y+98}" font-size="42" font-weight="700" fill="#0f172a">{safe(value)}</text>'
    )


def safe(text: object) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


if __name__ == "__main__":
    main()
