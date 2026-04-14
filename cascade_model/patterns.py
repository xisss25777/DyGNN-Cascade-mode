from typing import Dict, List

from .dynamic_graph import Snapshot


def identify_key_patterns(cascade_id: str, snapshots: List[Snapshot], prediction: float) -> List[Dict[str, float]]:
    patterns: List[Dict[str, float]] = []
    if not snapshots:
        return patterns

    first = snapshots[0]
    middle = snapshots[len(snapshots) // 2]
    last = snapshots[-1]

    first_growth = len(first.new_events)
    middle_growth = len(middle.new_events)
    last_growth = len(last.new_events)
    max_depth = max(last.depth_by_node.values(), default=0)
    max_width = max((list(last.depth_by_node.values()).count(depth) for depth in set(last.depth_by_node.values())), default=1)
    root = _infer_root(first, last)
    root_children = len(last.children_by_node.get(root, []))

    if first_growth >= max(3, middle_growth):
        patterns.append(
            {
                "pattern": "早期快速扩散模式",
                "score": round(first_growth / max(1, len(last.seen_nodes)), 4),
                "evidence": "前期时间片新增传播节点明显更高，说明信息在早期迅速破圈。",
            }
        )
    if max_depth >= 3 and max_width >= 2:
        patterns.append(
            {
                "pattern": "桥接式跨层传播模式",
                "score": round(max_depth / max(1, len(last.seen_nodes)), 4),
                "evidence": "传播深度持续增加，级联并非只在局部扩散，而是跨层延伸。",
            }
        )
    if middle_growth > first_growth * 0.6 and max_width >= 3:
        patterns.append(
            {
                "pattern": "局部聚集后外溢模式",
                "score": round(max_width / max(1, len(last.seen_nodes)), 4),
                "evidence": "中期宽度较大，说明局部聚集后出现外扩扩散。",
            }
        )
    if root_children >= max(2, len(last.edges) * 0.2):
        patterns.append(
            {
                "pattern": "核心节点驱动放大型模式",
                "score": round(root_children / max(1, len(last.edges)), 4),
                "evidence": "源节点直接带动较多后继传播，存在明显的核心驱动效应。",
            }
        )
    if not patterns:
        patterns.append(
            {
                "pattern": "平稳扩散模式",
                "score": round(prediction / max(1.0, len(last.seen_nodes) * 10.0), 4),
                "evidence": "未检测到强烈单一结构信号，传播更接近平稳累积。",
            }
        )
    return sorted(patterns, key=lambda item: item["score"], reverse=True)


def _infer_root(first: Snapshot, last: Snapshot) -> str:
    zero_depth_nodes = [node for node, depth in last.depth_by_node.items() if depth == 0]
    if zero_depth_nodes:
        return sorted(zero_depth_nodes)[0]
    if first.new_events:
        return first.new_events[0].user_id
    if first.seen_nodes:
        return sorted(first.seen_nodes)[0]
    return "root"


def rank_feature_importance(feature_names: List[str], weights: List[float], top_k: int = 10) -> List[Dict[str, float]]:
    pairs = [
        {"feature": feature, "importance": round(abs(weight), 6)}
        for feature, weight in zip(feature_names, weights)
    ]
    pairs.sort(key=lambda item: item["importance"], reverse=True)
    return pairs[:top_k]
