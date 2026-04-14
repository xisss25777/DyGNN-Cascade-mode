import math
from statistics import mean
from typing import Dict, List, Tuple

from .data import Cascade
from .dynamic_graph import Snapshot, build_snapshots, summarize_width


def build_feature_table(
    cascades: List[Cascade],
    observation_seconds: int,
    slice_seconds: int,
) -> Tuple[List[str], List[List[float]], List[float], Dict[str, List[Snapshot]]]:
    feature_rows: List[List[float]] = []
    target_values: List[float] = []
    snapshot_map: Dict[str, List[Snapshot]] = {}
    feature_names: List[str] = []

    for cascade in cascades:
        snapshots = build_snapshots(cascade, observation_seconds, slice_seconds)
        if not snapshots:
            continue
        snapshot_map[cascade.cascade_id] = snapshots
        names, row = extract_features(cascade, snapshots, slice_seconds)
        if not feature_names:
            feature_names = names
        feature_rows.append(row)
        target_values.append(float(cascade.final_size))
    return feature_names, feature_rows, target_values, snapshot_map


def extract_features(
    cascade: Cascade,
    snapshots: List[Snapshot],
    slice_seconds: int,
) -> Tuple[List[str], List[float]]:
    feature_names: List[str] = []
    values: List[float] = []
    last_nodes = 0
    last_edges = 0
    cumulative_events = 0
    first_time = cascade.events[0].timestamp
    inter_event_gaps = [
        cascade.events[idx].timestamp - cascade.events[idx - 1].timestamp
        for idx in range(1, len(cascade.events))
    ]
    observed_events = [event for snapshot in snapshots for event in snapshot.new_events]
    extra_feature_count = len(observed_events[0].extra_features) if observed_events and observed_events[0].extra_features else 0
    burstiness = 0.0
    if inter_event_gaps:
        gap_mean = mean(inter_event_gaps)
        gap_std = _std(inter_event_gaps)
        denominator = gap_std + gap_mean
        burstiness = (gap_std - gap_mean) / denominator if denominator else 0.0

    slice_new_nodes: List[float] = []
    slice_active_events: List[float] = []
    slice_depths: List[float] = []
    slice_widths: List[float] = []

    for idx, snapshot in enumerate(snapshots, start=1):
        node_count = len(snapshot.seen_nodes)
        edge_count = len(snapshot.edges)
        max_depth = max(snapshot.depth_by_node.values(), default=0)
        max_width = summarize_width(snapshot.depth_by_node)
        density = edge_count / max(1, node_count * max(1, node_count - 1))
        new_nodes = max(0, node_count - last_nodes)
        new_edges = max(0, edge_count - last_edges)
        growth_rate = new_nodes / max(1, node_count)
        leaf_count = sum(1 for node in snapshot.seen_nodes if not snapshot.children_by_node.get(node))
        leaf_ratio = leaf_count / max(1, node_count)
        root = cascade.events[0].user_id
        root_children = len(snapshot.children_by_node.get(root, []))
        root_influence = root_children / max(1, edge_count)
        avg_depth = sum(snapshot.depth_by_node.values()) / max(1, node_count)
        elapsed = snapshot.cutoff_time - first_time
        cumulative_events += len(snapshot.new_events)
        repeat_event_ratio = 1.0 - (node_count / max(1, cumulative_events))

        slice_feature_map = {
            f"slice_{idx}_node_count": float(node_count),
            f"slice_{idx}_edge_count": float(edge_count),
            f"slice_{idx}_max_depth": float(max_depth),
            f"slice_{idx}_max_width": float(max_width),
            f"slice_{idx}_density": density,
            f"slice_{idx}_new_nodes": float(new_nodes),
            f"slice_{idx}_new_edges": float(new_edges),
            f"slice_{idx}_growth_rate": growth_rate,
            f"slice_{idx}_leaf_ratio": leaf_ratio,
            f"slice_{idx}_root_influence": root_influence,
            f"slice_{idx}_avg_depth": avg_depth,
            f"slice_{idx}_active_events": float(len(snapshot.new_events)),
            f"slice_{idx}_cumulative_events": float(cumulative_events),
            f"slice_{idx}_repeat_event_ratio": repeat_event_ratio,
            f"slice_{idx}_elapsed_hours": elapsed / 3600.0,
        }
        for name, value in slice_feature_map.items():
            feature_names.append(name)
            values.append(value)
        slice_new_nodes.append(float(new_nodes))
        slice_active_events.append(float(len(snapshot.new_events)))
        slice_depths.append(float(max_depth))
        slice_widths.append(float(max_width))
        last_nodes = node_count
        last_edges = edge_count

    if extra_feature_count:
        extra_means = _average_extra_features(observed_events, extra_feature_count)
        extra_stds = _std_extra_features(observed_events, extra_feature_count, extra_means)
        for idx, value in enumerate(extra_means[:12], start=1):
            feature_names.append(f"extra_feature_mean_{idx}")
            values.append(value)
        for idx, value in enumerate(extra_stds[:8], start=1):
            feature_names.append(f"extra_feature_std_{idx}")
            values.append(value)

    final_snapshot = snapshots[-1]
    trend_feature_map = {
        "trend_new_node_slope": _linear_slope(slice_new_nodes),
        "trend_active_event_slope": _linear_slope(slice_active_events),
        "trend_depth_slope": _linear_slope(slice_depths),
        "trend_width_slope": _linear_slope(slice_widths),
        "trend_peak_active_events": max(slice_active_events) if slice_active_events else 0.0,
        "trend_peak_new_nodes": max(slice_new_nodes) if slice_new_nodes else 0.0,
        "trend_early_event_share": (
            sum(slice_active_events[: max(1, len(slice_active_events) // 3)]) / max(1.0, sum(slice_active_events))
        ),
        "trend_late_event_share": (
            sum(slice_active_events[-max(1, len(slice_active_events) // 3):]) / max(1.0, sum(slice_active_events))
        ),
    }
    global_feature_map = {
        "global_observed_size": float(len(final_snapshot.seen_nodes)),
        "global_observed_edges": float(len(final_snapshot.edges)),
        "global_observed_events": float(len(observed_events)),
        "global_burstiness": burstiness,
        "global_avg_gap_minutes": (mean(inter_event_gaps) / 60.0) if inter_event_gaps else 0.0,
        "global_log_observed_size": math.log1p(len(final_snapshot.seen_nodes)),
        "global_log_observed_events": math.log1p(len(observed_events)),
        "global_repeat_event_ratio": 1.0 - (len(final_snapshot.seen_nodes) / max(1, len(observed_events))),
        "global_depth_width_ratio": (
            max(final_snapshot.depth_by_node.values(), default=0)
            / max(1, summarize_width(final_snapshot.depth_by_node))
        ),
        "global_slice_count": float(len(snapshots)),
        "global_observation_hours": float(len(snapshots) * slice_seconds / 3600.0),
    }
    global_feature_map.update(trend_feature_map)
    for name, value in global_feature_map.items():
        feature_names.append(name)
        values.append(value)
    return feature_names, values


def _std(values: List[int]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    variance = sum((value - avg) ** 2 for value in values) / len(values)
    return variance ** 0.5


def _average_extra_features(events, feature_count: int) -> List[float]:
    totals = [0.0] * feature_count
    for event in events:
        for idx, value in enumerate(event.extra_features[:feature_count]):
            totals[idx] += value
    return [value / max(1, len(events)) for value in totals]


def _std_extra_features(events, feature_count: int, means: List[float]) -> List[float]:
    totals = [0.0] * feature_count
    for event in events:
        for idx, value in enumerate(event.extra_features[:feature_count]):
            diff = value - means[idx]
            totals[idx] += diff * diff
    return [(value / max(1, len(events))) ** 0.5 for value in totals]


def _linear_slope(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    x_mean = (len(values) - 1) / 2.0
    y_mean = sum(values) / len(values)
    numerator = 0.0
    denominator = 0.0
    for idx, value in enumerate(values):
        x_diff = idx - x_mean
        numerator += x_diff * (value - y_mean)
        denominator += x_diff * x_diff
    return numerator / denominator if denominator else 0.0
