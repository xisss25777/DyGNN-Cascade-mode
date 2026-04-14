from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

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

    start_time = cascade.events[0].timestamp
    cutoff_limit = start_time + observation_seconds
    usable_events = [event for event in cascade.events if event.timestamp <= cutoff_limit]
    if not usable_events:
        usable_events = [cascade.events[0]]

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

        depth_by_node = _compute_depths(seen_nodes, parent_map)
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


def _compute_depths(
    seen_nodes: Set[str],
    parent_map: Dict[str, Optional[str]],
) -> Dict[str, int]:
    depths: Dict[str, int] = {}
    for node in seen_nodes:
        depth = 0
        cursor = node
        visited = set()
        while parent_map.get(cursor) is not None and cursor not in visited:
            visited.add(cursor)
            cursor = parent_map[cursor]
            depth += 1
        depths[node] = depth
    return depths

