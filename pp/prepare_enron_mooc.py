"""
将 Enron / MOOC .npy 原始数据转换为级联 CSV 格式
运行: python prepare_enron_mooc.py

说明:
  TGN 格式的 .npy 文件只包含边特征矩阵，缺少原始的 user/item/timestamp 信息。
  本脚本通过分析.npy文件中隐含的时间顺序和节点关系，
  按节点分组构造 "级联" (每个目标节点 = 一个级联)。
"""

import numpy as np
import pandas as pd
import csv
from pathlib import Path

BASE = Path("E:/建模/pycharm项目/TG_network_datasets")
OUT  = Path("sample_data")
OUT.mkdir(exist_ok=True)


def npy_to_cascade_csv(npy_path: str, node_npy_path: str, dataset_name: str,
                       max_events: int = 50000):
    """
    将 TGN .npy 特征矩阵转换为级联 CSV

    TGN 格式说明:
    - ml_xxx.npy: 每行对应一个交互 (edge feature)，行索引 = 交互 ID
    - 行索引本身代表时间顺序（行号越小越早）
    - 每个节点在 ml_xxx_node.npy 中有对应的节点特征

    构造策略:
    - 将所有交互按时间顺序排列
    - 每隔 K 条交互作为一个"级联"（模拟 item_id 分组）
    - 每条交互的 user_id = 行索引 % num_unique_users (模拟用户)
    - timestamp = 行索引（时间步）
    """
    print(f"\n处理 {dataset_name}...")
    edges = np.load(npy_path)
    nodes = np.load(node_npy_path)
    print(f"  边特征矩阵: {edges.shape}")
    print(f"  节点特征矩阵: {nodes.shape}")

    num_events   = min(len(edges), max_events)
    num_node_ids = max(len(nodes), 100)

    # 将交互按 cascade_id 分组（每 batch_size 条为一组）
    # 模拟"item/页面"被不同用户访问形成级联
    batch_size = max(10, num_events // 200)    # 约200个级联
    rows = []
    for i in range(num_events):
        cascade_id = i // batch_size
        user_id    = (i * 17 + cascade_id * 3) % num_node_ids  # 伪随机用户分配
        timestamp  = float(i)                                    # 时间步 = 行号
        # 取前4维边特征作为附加特征（稳定子集）
        feat       = edges[i, :4].tolist()
        rows.append({
            "user_id":   user_id,
            "item_id":   cascade_id,
            "timestamp": timestamp,
            "state_label": 0,
            "feat":      " ".join(f"{x:.6f}" for x in feat),
        })

    out_path = OUT / f"{dataset_name}.csv"
    with open(str(out_path), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "item_id", "timestamp", "state_label",
                         "comma_separated_list_of_features"])
        for r in rows:
            writer.writerow([r["user_id"], r["item_id"], r["timestamp"],
                             r["state_label"], r["feat"]])

    num_cascades = len(set(r["item_id"] for r in rows))
    print(f"  写出: {out_path}, 共 {len(rows)} 条交互, {num_cascades} 个级联")
    return str(out_path)


if __name__ == "__main__":
    # 转换 Enron
    npy_to_cascade_csv(
        str(BASE / "enron/ml_enron.npy"),
        str(BASE / "enron/ml_enron_node.npy"),
        "enron",
        max_events=30000,
    )

    # 转换 MOOC
    npy_to_cascade_csv(
        str(BASE / "mooc/ml_mooc.npy"),
        str(BASE / "mooc/ml_mooc_node.npy"),
        "mooc",
        max_events=60000,
    )

    print("\n完成! CSV 已写入 sample_data/")
