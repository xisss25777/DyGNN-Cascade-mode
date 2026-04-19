from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple

from .data import Cascade, Event


@dataclass
class Snapshot:
    cutoff_time: int
    seen_nodes: Set[str]
    edges: List[tuple]
    new_events: List[Event]
    depth_by_node: Dict[str, int]
    children_by_node: Dict[str, List[str]]


def build_snapshots(
    cascade: Cascade,
    observation_seconds: int,
    slice_seconds: int,
) -> List[Snapshot]:
    if not cascade.events:
        return []

    # 时间已经归一化，从0开始
    start_time = 0
    cutoff_limit = observation_seconds
    
    # 所有事件都应该在时间范围内
    usable_events = cascade.events
    if not usable_events:
        return []
    
    # 按时间排序，确保顺序正确
    usable_events = sorted(usable_events, key=lambda x: x.timestamp)

    snapshots: List[Snapshot] = []
    current_events: List[Event] = []
    seen_nodes: Set[str] = set()
    edges: List[tuple] = []
    children_by_node: Dict[str, List[str]] = defaultdict(list)
    depth_by_node: Dict[str, int] = {}
    parent_map: Dict[str, Optional[str]] = {}
    slice_count = max(1, (observation_seconds + slice_seconds - 1) // slice_seconds)
    event_index = 0

    for slice_idx in range(slice_count):
        cutoff_time = start_time + (slice_idx + 1) * slice_seconds
        new_events: List[Event] = []
        while event_index < len(usable_events) and usable_events[event_index].timestamp <= cutoff_time:
            event = usable_events[event_index]
            current_events.append(event)
            new_events.append(event)
            seen_nodes.add(event.user_id)
            if event.parent_id:
                edges.append((event.parent_id, event.user_id))
                children_by_node[event.parent_id].append(event.user_id)
                parent_map[event.user_id] = event.parent_id
            else:
                parent_map[event.user_id] = None
            event_index += 1

        depth_by_node = compute_depths(seen_nodes, parent_map)
        snapshots.append(
            Snapshot(
                cutoff_time=cutoff_time,
                seen_nodes=set(seen_nodes),
                edges=list(edges),
                new_events=new_events,
                depth_by_node=depth_by_node,
                children_by_node={key: list(value) for key, value in children_by_node.items()},
            )
        )
    return snapshots


def summarize_width(depth_by_node: Dict[str, int]) -> int:
    if not depth_by_node:
        return 0
    level_counts = Counter(depth_by_node.values())
    return max(level_counts.values())


@lru_cache(maxsize=1000)
def _compute_depths(
    seen_nodes: Tuple[str, ...],
    parent_map: Tuple[Tuple[str, Optional[str]], ...],
) -> Tuple[Tuple[str, int], ...]:
    """
    计算每个节点的深度，使用缓存提高性能
    
    Args:
        seen_nodes: 节点集合（转换为元组以支持缓存）
        parent_map: 父节点映射（转换为元组以支持缓存）
    
    Returns:
        节点深度的元组，每个元素是 (node, depth) 对
    """
    # 将元组转换回原始类型
    seen_nodes_set = set(seen_nodes)
    parent_map_dict = dict(parent_map)
    
    depths: Dict[str, int] = {}
    for node in seen_nodes_set:
        depth = 0
        cursor = node
        visited = set()
        while parent_map_dict.get(cursor) is not None and cursor not in visited:
            visited.add(cursor)
            cursor = parent_map_dict[cursor]
            depth += 1
        depths[node] = depth
    
    # 将结果转换为元组以支持缓存
    return tuple(sorted(depths.items()))


def compute_depths(
    seen_nodes: Set[str],
    parent_map: Dict[str, Optional[str]],
) -> Dict[str, int]:
    """
    计算每个节点的深度，使用缓存提高性能
    
    Args:
        seen_nodes: 节点集合
        parent_map: 父节点映射
    
    Returns:
        节点深度的字典
    """
    # 将参数转换为可哈希类型以支持缓存
    seen_nodes_tuple = tuple(sorted(seen_nodes))
    parent_map_tuple = tuple(sorted(parent_map.items()))
    
    # 调用缓存版本的函数
    depths_tuple = _compute_depths(seen_nodes_tuple, parent_map_tuple)
    
    # 将结果转换回字典
    return dict(depths_tuple)