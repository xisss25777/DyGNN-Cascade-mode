import json
import argparse
import math
from pathlib import Path


W = 1800
H = 1280


def main() -> None:
    parser = argparse.ArgumentParser(description="生成论文风实验结果图")
    parser.add_argument("--wiki", default="wikipedia_result.json", help="Wikipedia 结果文件")
    parser.add_argument("--reddit", default="reddit_dgnn.json", help="Reddit 结果文件")
    args = parser.parse_args()

    # 获取脚本所在目录的绝对路径
    script_dir = Path(__file__).parent
    outputs_dir = script_dir / "outputs"
    
    # 构建完整的文件路径
    wiki_path = outputs_dir / args.wiki
    reddit_path = outputs_dir / args.reddit
    output = outputs_dir / "figures" / "publication_results_figure.svg"
    
    # 读取文件
    wiki = json.loads(wiki_path.read_text(encoding="utf-8"))
    reddit = json.loads(reddit_path.read_text(encoding="utf-8"))
    
    # 确保输出目录存在
    output.parent.mkdir(parents=True, exist_ok=True)
    
    # 写入结果
    output.write_text(build_svg(wiki, reddit), encoding="utf-8")
    print(f"论文结果图已输出到: {output}")

def build_svg(wiki: dict, reddit: dict) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
  {defs()}
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="80" y="72" font-family="Helvetica, Arial, sans-serif" font-size="34" font-weight="700" fill="#111827">
    Dynamic cascade prediction across Wikipedia and Reddit
  </text>
  <text x="80" y="106" font-family="Helvetica, Arial, sans-serif" font-size="18" fill="#4b5563">
    Early dynamic graph modeling, cross-platform evaluation, and interpretable propagation patterns
  </text>
  {panel_frame(70, 140, 800, 500, "A", "Cross-dataset predictive accuracy")}
  {panel_frame(930, 140, 800, 500, "B", "Dominant feature contributions")}
  {panel_frame(70, 700, 800, 500, "C", "Deletion-test response across representative cascades")}
  {panel_frame(930, 700, 800, 500, "D", "Propagation pattern composition")}
  {draw_accuracy_panel(110, 215, 720, 380, wiki, reddit)}
  {draw_feature_panel(970, 215, 720, 380, wiki, reddit)}
  {draw_deletion_panel(110, 775, 720, 380, wiki, reddit)}
  {draw_pattern_panel(970, 775, 720, 380, wiki, reddit)}
</svg>"""


def defs() -> str:
    return """
<defs>
  <filter id="softShadow" x="-10%" y="-10%" width="120%" height="120%">
    <feDropShadow dx="0" dy="3" stdDeviation="8" flood-color="#94a3b8" flood-opacity="0.12"/>
  </filter>
  <style>
    .title { font-family: Helvetica, Arial, sans-serif; fill: #111827; }
    .label { font-family: Helvetica, Arial, sans-serif; fill: #374151; }
    .muted { font-family: Helvetica, Arial, sans-serif; fill: #6b7280; }
    .tick { font-family: Helvetica, Arial, sans-serif; fill: #4b5563; font-size: 14px; }
    .panel { fill: #ffffff; stroke: #d1d5db; stroke-width: 1.1; }
    .grid { stroke: #e5e7eb; stroke-width: 1; }
    .axis { stroke: #6b7280; stroke-width: 1.2; }
  </style>
</defs>
"""


def panel_frame(x: int, y: int, w: int, h: int, letter: str, title: str) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" class="panel" filter="url(#softShadow)"/>'
        f'<text x="{x+24}" y="{y+34}" class="title" font-size="26" font-weight="700">{letter}</text>'
        f'<text x="{x+60}" y="{y+34}" class="title" font-size="24" font-weight="700">{safe(title)}</text>'
    )


def draw_accuracy_panel(x: int, y: int, w: int, h: int, wiki: dict, reddit: dict) -> str:
    keys = [("mae", "MAE"), ("rmse", "RMSE"), ("mape", "MAPE")]
    blocks = []
    max_log = max(math.log10(max(1e-6, dataset["metrics"][key])) for dataset in (wiki, reddit) for key, _ in keys[:2])
    max_mape = max(wiki["metrics"]["mape"], reddit["metrics"]["mape"])
    group_gap = w / 3

    for gx, (key, label) in enumerate(keys):
        base_x = x + 60 + gx * group_gap
        blocks.append(f'<text x="{base_x+65}" y="{y+h+34}" text-anchor="middle" class="label" font-size="18">{label}</text>')
        if key != "mape":
            blocks.extend(log_bar_group(base_x, y + 28, 44, h - 80, wiki["metrics"][key], reddit["metrics"][key], max_log))
        else:
            blocks.extend(linear_bar_group(base_x, y + 28, 44, h - 80, wiki["metrics"][key], reddit["metrics"][key], max_mape))

    legend_y = y - 8
    blocks.append(circle_legend(x + 10, legend_y, "#2b6cb0", "Wikipedia"))
    blocks.append(circle_legend(x + 165, legend_y, "#c05621", "Reddit"))
    blocks.append(f'<text x="{x+10}" y="{y+h+62}" class="muted" font-size="15">MAE and RMSE are drawn on a log scale to accommodate Reddit heavy-tail errors.</text>')
    return "".join(blocks)


def log_bar_group(base_x: float, top_y: float, bar_w: float, height: float, v1: float, v2: float, max_log: float) -> list:
    elems = axis_guides(base_x - 26, top_y, height, log_scale=True)
    vals = [v1, v2]
    colors = ["#2b6cb0", "#c05621"]
    for idx, value in enumerate(vals):
        scaled = math.log10(max(value, 1e-6)) / max_log
        bar_h = height * scaled
        x = base_x + idx * 76
        y = top_y + height - bar_h
        elems.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" rx="4" fill="{colors[idx]}"/>')
        elems.append(f'<text x="{x + bar_w/2}" y="{y-10:.1f}" text-anchor="middle" class="tick">{short_num(value)}</text>')
    return elems


def linear_bar_group(base_x: float, top_y: float, bar_w: float, height: float, v1: float, v2: float, vmax: float) -> list:
    elems = axis_guides(base_x - 26, top_y, height, log_scale=False, max_value=vmax)
    vals = [v1, v2]
    colors = ["#2b6cb0", "#c05621"]
    for idx, value in enumerate(vals):
        scaled = value / vmax if vmax else 0
        bar_h = height * scaled
        x = base_x + idx * 76
        y = top_y + height - bar_h
        elems.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" rx="4" fill="{colors[idx]}"/>')
        elems.append(f'<text x="{x + bar_w/2}" y="{y-10:.1f}" text-anchor="middle" class="tick">{value:.3f}</text>')
    return elems


def axis_guides(x: float, top_y: float, height: float, log_scale: bool, max_value: float = 1.0) -> list:
    elems = [f'<line x1="{x+18}" y1="{top_y}" x2="{x+18}" y2="{top_y+height}" class="axis"/>']
    if log_scale:
        ticks = [(0, "10^0"), (0.33, "10^1"), (0.66, "10^3"), (1.0, "10^5")]
        for frac, label in ticks:
            yy = top_y + height - height * frac
            elems.append(f'<line x1="{x+18}" y1="{yy:.1f}" x2="{x+220}" y2="{yy:.1f}" class="grid"/>')
            elems.append(f'<text x="{x}" y="{yy+5:.1f}" text-anchor="end" class="tick">{label}</text>')
    else:
        for frac in [0, 0.25, 0.5, 0.75, 1.0]:
            yy = top_y + height - height * frac
            elems.append(f'<line x1="{x+18}" y1="{yy:.1f}" x2="{x+220}" y2="{yy:.1f}" class="grid"/>')
            elems.append(f'<text x="{x}" y="{yy+5:.1f}" text-anchor="end" class="tick">{max_value*frac:.2f}</text>')
    return elems


def draw_feature_panel(x: int, y: int, w: int, h: int, wiki: dict, reddit: dict) -> str:
    elems = []
    wiki_feats = wiki["top_features"][:6]
    reddit_feats = reddit["top_features"][:6]
    left_x = x + 10
    right_x = x + w / 2 + 20
    elems.append(f'<text x="{left_x}" y="{y-6}" class="label" font-size="18" font-weight="700">Wikipedia</text>')
    elems.append(f'<text x="{right_x}" y="{y-6}" class="label" font-size="18" font-weight="700">Reddit</text>')
    elems.extend(feature_column(left_x, y + 24, 300, wiki_feats, "#2b6cb0"))
    elems.extend(feature_column(right_x, y + 24, 300, reddit_feats, "#c05621"))
    elems.append(f'<text x="{x+10}" y="{y+h+30}" class="muted" font-size="15">Both datasets rank temporal spacing as the top predictor, while Reddit is more sensitive to burstiness and event-volume terms.</text>')
    return "".join(elems)


def feature_column(x: float, y: float, max_w: float, feats: list, color: str) -> list:
    elems = []
    vmax = max(item["importance"] for item in feats) if feats else 1.0
    for idx, item in enumerate(feats):
        yy = y + idx * 56
        bw = max_w * item["importance"] / vmax
        elems.append(f'<text x="{x}" y="{yy-8}" class="tick">{safe(truncate(item["feature"], 24))}</text>')
        elems.append(f'<rect x="{x}" y="{yy}" width="{bw:.1f}" height="18" rx="3" fill="{color}" opacity="0.9"/>')
        elems.append(f'<text x="{x+bw+12:.1f}" y="{yy+14}" class="tick">{item["importance"]:.3f}</text>')
    return elems


def draw_deletion_panel(x: int, y: int, w: int, h: int, wiki: dict, reddit: dict) -> str:
    wiki_vals = [item["deletion_test"]["delta"] for item in wiki["test_reports"][:12]]
    reddit_vals = [item["deletion_test"]["delta"] for item in reddit["test_reports"][:12]]
    all_vals = wiki_vals + reddit_vals
    min_v = min(all_vals)
    max_v = max(all_vals)
    x0 = x + 60
    x1 = x + w - 30
    y_mid = y + h / 2
    elems = [
        f'<line x1="{x0}" y1="{y_mid}" x2="{x1}" y2="{y_mid}" class="axis"/>',
        f'<text x="{x0}" y="{y+10}" class="tick">Negative values: early dynamic signals suppress prediction after deletion</text>',
        f'<text x="{x0}" y="{y+h+30}" class="muted" font-size="15">Positive delta indicates that removing early dynamic cues decreases the predicted cascade size.</text>',
    ]
    for frac in [0, 0.25, 0.5, 0.75, 1.0]:
        val = min_v + (max_v - min_v) * frac
        xx = x0 + (x1 - x0) * frac
        elems.append(f'<line x1="{xx:.1f}" y1="{y+20}" x2="{xx:.1f}" y2="{y+h-25}" class="grid"/>')
        elems.append(f'<text x="{xx:.1f}" y="{y+h-4}" text-anchor="middle" class="tick">{short_num(val)}</text>')
    elems.extend(delta_series(x0, x1, y + 70, wiki_vals, min_v, max_v, "#2b6cb0", "Wikipedia"))
    elems.extend(delta_series(x0, x1, y + 240, reddit_vals, min_v, max_v, "#c05621", "Reddit"))
    return "".join(elems)


def delta_series(x0: float, x1: float, y: float, values: list, vmin: float, vmax: float, color: str, label: str) -> list:
    elems = [f'<text x="{x0}" y="{y-18}" class="label" font-size="18" font-weight="700">{label}</text>']
    for idx, value in enumerate(values):
        yy = y + idx * 18
        xx = map_value(value, vmin, vmax, x0, x1)
        base = map_value(0, vmin, vmax, x0, x1)
        elems.append(f'<line x1="{base:.1f}" y1="{yy}" x2="{xx:.1f}" y2="{yy}" stroke="{color}" stroke-width="2.2" opacity="0.8"/>')
        elems.append(f'<circle cx="{xx:.1f}" cy="{yy}" r="4.4" fill="{color}"/>')
    return elems


def draw_pattern_panel(x: int, y: int, w: int, h: int, wiki: dict, reddit: dict) -> str:
    wiki_counts = pattern_counts(wiki)
    reddit_counts = pattern_counts(reddit)
    all_names = sorted(set(wiki_counts) | set(reddit_counts))
    colors = {
        "平稳扩散模式": "#4c78a8",
        "早期快速扩散模式": "#f58518",
        "核心节点驱动放大型模式": "#54a24b",
        "桥接式跨层传播模式": "#b279a2",
        "局部聚集后外溢模式": "#e45756",
    }
    elems = []
    elems.extend(stacked_bar(x + 50, y + 80, 200, 40, wiki_counts, all_names, colors, "Wikipedia"))
    elems.extend(stacked_bar(x + 50, y + 190, 200, 40, reddit_counts, all_names, colors, "Reddit"))
    legend_x = x + 50
    legend_y = y + 300
    for idx, name in enumerate(all_names):
        col = idx % 2
        row = idx // 2
        lx = legend_x + col * 310
        ly = legend_y + row * 38
        elems.append(f'<rect x="{lx}" y="{ly-12}" width="18" height="18" fill="{colors.get(name, "#9ca3af")}"/>')
        elems.append(f'<text x="{lx+28}" y="{ly+2}" class="tick">{safe(name)}</text>')
    elems.append(f'<text x="{x+50}" y="{y+h+28}" class="muted" font-size="15">Wikipedia exhibits more diverse identifiable mechanisms, whereas Reddit test cases are dominated by a stable-diffusion signature.</text>')
    return "".join(elems)


def stacked_bar(x: float, y: float, w: float, h: float, counts: dict, names: list, colors: dict, label: str) -> list:
    total = sum(counts.values()) or 1
    elems = [f'<text x="{x}" y="{y-18}" class="label" font-size="18" font-weight="700">{label}</text>']
    cursor = x
    for name in names:
        frac = counts.get(name, 0) / total
        seg_w = w * frac
        if seg_w <= 0:
            continue
        elems.append(f'<rect x="{cursor:.1f}" y="{y}" width="{seg_w:.1f}" height="{h}" fill="{colors.get(name, "#9ca3af")}"/>')
        if seg_w > 38:
            elems.append(f'<text x="{cursor + seg_w/2:.1f}" y="{y + 25}" text-anchor="middle" font-size="13" fill="white">{counts.get(name,0)}</text>')
        cursor += seg_w
    elems.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" stroke="#9ca3af" stroke-width="1"/>')
    return elems


def pattern_counts(report: dict) -> dict:
    counts = {}
    for item in report["test_reports"]:
        for p in item["patterns"]:
            counts[p["pattern"]] = counts.get(p["pattern"], 0) + 1
    return counts


def circle_legend(x: float, y: float, color: str, label: str) -> str:
    return f'<circle cx="{x}" cy="{y}" r="6" fill="{color}"/><text x="{x+14}" y="{y+5}" class="tick">{label}</text>'


def map_value(value: float, vmin: float, vmax: float, out_min: float, out_max: float) -> float:
    if vmax <= vmin:
        return (out_min + out_max) / 2
    return out_min + (value - vmin) * (out_max - out_min) / (vmax - vmin)


def short_num(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 100000:
        return f"{value/1000:.0f}k"
    if abs_value >= 1000:
        return f"{value/1000:.1f}k"
    if abs_value >= 100:
        return f"{value:.0f}"
    if abs_value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def safe(text: object) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


if __name__ == "__main__":
    main()