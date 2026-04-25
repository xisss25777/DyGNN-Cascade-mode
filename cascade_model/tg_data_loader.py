"""
TG 数据集加载器
支持 Wikipedia / Reddit / Enron / MOOC 四个 TGN 格式数据集
数据格式: user_id, item_id(cascade_id), timestamp, state_label, edge_features...
"""

import csv
import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .data import Event, Cascade


# ────────────────────────────────────────────────────
# 数据集配置
# ────────────────────────────────────────────────────

DATASET_CONFIGS = {
    "wikipedia": {
        "csv":      "E:/建模/pycharm项目/TG_network_datasets/wikipedia/ml_wikipedia.csv"
                    if False else None,   # 若缺失 csv 则用 sample_data/wikipedia.csv
        "sample":   "sample_data/wikipedia.csv",
        "min_cascade_size": 5,
        "max_cascades":     200,
        "observation_ratio": 0.5,        # 用前50%时间作为观察窗口
        "description": "Wikipedia 编辑行为级联 (1665用户, 9228条目)"
    },
    "reddit": {
        "csv":      None,
        "sample":   "sample_data/reddit.csv",
        "min_cascade_size": 5,
        "max_cascades":     200,
        "observation_ratio": 0.5,
        "description": "Reddit 评论级联 (309用户, 281子版块)"
    },
    "enron": {
        "csv":      None,
        "sample":   "sample_data/enron.csv",
        "min_cascade_size": 3,
        "max_cascades":     300,
        "observation_ratio": 0.4,
        "description": "Enron 邮件传播级联 (185节点, 邮件网络)"
    },
    "mooc": {
        "csv":      None,
        "sample":   "sample_data/mooc.csv",
        "min_cascade_size": 4,
        "max_cascades":     300,
        "observation_ratio": 0.5,
        "description": "MOOC 学生行为级联 (7145学生)"
    },
}


# ────────────────────────────────────────────────────
# 从 CSV 加载 TG 格式数据集
# ────────────────────────────────────────────────────

def load_tg_csv(
    path: str,
    min_cascade_size: int = 5,
    max_cascades: int = 200,
    observation_ratio: float = 0.5,
) -> List[Cascade]:
    """
    加载 TGN 格式 CSV，按 item_id 分组构建级联。

    CSV 格式:
        user_id, item_id, timestamp, state_label, feat1, feat2, ...

    Returns:
        List[Cascade]，每个级联的 target_size = 总唯一用户数
    """
    raw: Dict[str, List[dict]] = {}

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)   # 跳过表头

        for row in reader:
            if len(row) < 4:
                continue
            try:
                user_id   = str(int(float(row[0])))
                item_id   = str(int(float(row[1])))
                timestamp = float(row[2])
                label     = float(row[3])
                extra     = [float(x) for x in row[4:row.index('')] if x] if '' in row[4:] else [float(x) for x in row[4:]]
            except (ValueError, IndexError):
                continue

            raw.setdefault(item_id, []).append({
                "user_id":   user_id,
                "timestamp": timestamp,
                "label":     label,
                "extra":     extra[:12],   # 最多保留12维边特征
            })

    cascades = []
    for cascade_id, records in raw.items():
        if len(records) < min_cascade_size:
            continue
        # 按时间排序
        records.sort(key=lambda r: r["timestamp"])

        # 时间归一化（到 86400 秒 = 1 天）
        t0 = records[0]["timestamp"]
        t_end = records[-1]["timestamp"]
        span = max(t_end - t0, 1.0)

        events = []
        for i, r in enumerate(records):
            norm_ts = int((r["timestamp"] - t0) / span * 86400)
            parent_id = None if i == 0 else records[i - 1]["user_id"]
            events.append(Event(
                cascade_id=cascade_id,
                user_id=r["user_id"],
                timestamp=norm_ts,
                parent_id=parent_id,
                event_type="interaction",
                extra_features=r["extra"],
            ))

        unique_users = len({e.user_id for e in events})
        cascades.append(Cascade(
            cascade_id=cascade_id,
            events=events,
            target_size=unique_users,
        ))

    # 按级联大小降序排列，取前 max_cascades 个
    cascades.sort(key=lambda c: c.target_size, reverse=True)
    cascades = cascades[:max_cascades]

    if not cascades:
        raise ValueError(
            f"数据集 '{path}' 加载失败：没有找到满足条件的级联。"
            f"请检查：(1)CSV格式是否正确；(2)min_cascade_size={min_cascade_size}是否过大"
        )
    print(f"[TG Loader] 加载完成: {len(cascades)} 个级联, "
          f"规模范围 [{min(c.target_size for c in cascades)}, "
          f"{max(c.target_size for c in cascades)}]")
    return cascades


def load_dataset_by_name(
    dataset_name: str,
    base_dir: str = "sample_data",
    min_cascade_size: int = 5,
    max_cascades: int = 200,
) -> List[Cascade]:
    """
    按数据集名称加载，自动寻找 CSV 路径。
    优先使用 pp/sample_data/ 下已有的 CSV。
    """
    cfg = DATASET_CONFIGS.get(dataset_name)
    if cfg is None:
        raise ValueError(f"未知数据集: {dataset_name}，支持: {list(DATASET_CONFIGS)}")

    # 寻找可用路径
    candidates = []
    if cfg["csv"]:
        candidates.append(cfg["csv"])
    candidates.append(Path(base_dir) / f"{dataset_name}.csv")
    candidates.append(Path(base_dir) / cfg["sample"])

    path = None
    for c in candidates:
        if Path(c).exists():
            path = str(c)
            break

    if path is None:
        raise FileNotFoundError(
            f"找不到 {dataset_name} 数据文件，请将 CSV 放至: {candidates}"
        )

    print(f"[TG Loader] 数据集: {dataset_name} ({cfg['description']})")
    print(f"[TG Loader] 路径: {path}")

    return load_tg_csv(
        path,
        min_cascade_size=min_cascade_size,
        max_cascades=max_cascades,
        observation_ratio=cfg["observation_ratio"],
    )


# ────────────────────────────────────────────────────
# 数据集统计分析
# ────────────────────────────────────────────────────

def analyze_cascades(cascades: List[Cascade]) -> dict:
    """
    计算级联数据集的统计特征（用于论文数据描述章节）
    """
    sizes    = [c.target_size for c in cascades]
    lengths  = [len(c.events) for c in cascades]

    # 规模分位
    sizes_arr = np.array(sizes)
    log_sizes = np.log1p(sizes_arr)

    stats = {
        "count":           len(cascades),
        "size_mean":       float(np.mean(sizes_arr)),
        "size_median":     float(np.median(sizes_arr)),
        "size_std":        float(np.std(sizes_arr)),
        "size_min":        int(np.min(sizes_arr)),
        "size_max":        int(np.max(sizes_arr)),
        "size_q25":        float(np.percentile(sizes_arr, 25)),
        "size_q75":        float(np.percentile(sizes_arr, 75)),
        "log_size_mean":   float(np.mean(log_sizes)),
        "log_size_std":    float(np.std(log_sizes)),
        "event_mean":      float(np.mean(lengths)),
        "event_median":    float(np.median(lengths)),

        # 规模分布区间
        "small_lt10":  int(np.sum(sizes_arr < 10)),
        "medium_10_50":int(np.sum((sizes_arr >= 10) & (sizes_arr < 50))),
        "large_50_200":int(np.sum((sizes_arr >= 50) & (sizes_arr < 200))),
        "xlarge_ge200":int(np.sum(sizes_arr >= 200)),
    }
    return stats


# ────────────────────────────────────────────────────
# 数据增强：子级联采样（解决样本不足问题）
# ────────────────────────────────────────────────────

def augment_by_subcascade(
    cascades: List[Cascade],
    n_splits: int = 3,
    min_events: int = 5,
) -> List[Cascade]:
    """
    对每个级联按时间切割成若干子级联，扩增训练样本。

    策略:
    - 将级联时间跨度均匀分成 n_splits+1 段
    - 取前 k/(n_splits+1) 的事件作为子级联
    - 子级联目标仍是原始级联的最终规模（预测任务目标不变）

    这样可以让模型学到"从不同观察窗口预测最终规模"。
    """
    augmented = []
    for cascade in cascades:
        augmented.append(cascade)  # 原始保留

        events = cascade.events
        if len(events) < min_events * 2:
            continue

        for split_idx in range(1, n_splits + 1):
            ratio = split_idx / (n_splits + 1)
            cutoff = int(len(events) * ratio)
            if cutoff < min_events:
                continue

            sub_events = events[:cutoff]
            sub_id = f"{cascade.cascade_id}_aug{split_idx}"
            augmented.append(Cascade(
                cascade_id=sub_id,
                events=sub_events,
                target_size=cascade.target_size,   # 目标：原始最终规模
            ))

    print(f"[增强] 原始级联: {len(cascades)}, 增强后: {len(augmented)}")
    return augmented
