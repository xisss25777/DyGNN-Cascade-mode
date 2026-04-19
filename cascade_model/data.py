import csv
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Event:
    cascade_id: str
    user_id: str
    timestamp: int
    parent_id: Optional[str] = None
    event_type: str = "repost"
    extra_features: List[float] = field(default_factory=list)


@dataclass
class Cascade:
    cascade_id: str
    events: List[Event] = field(default_factory=list)
    target_size: Optional[int] = None

    @property
    def final_size(self) -> int:
        if self.target_size is not None:
            return self.target_size
        return len({event.user_id for event in self.events})


def load_cascades_from_csv(path: str) -> List[Cascade]:
    cascades: Dict[str, List[Event]] = {}
    with open(path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            event = Event(
                cascade_id=row["cascade_id"],
                user_id=row["user_id"],
                parent_id=row.get("parent_id") or None,
                timestamp=int(row["timestamp"]),
                event_type=row.get("event_type", "repost"),
            )
            cascades.setdefault(event.cascade_id, []).append(event)

    result = []
    for cascade_id, events in cascades.items():
        ordered = sorted(events, key=lambda item: item.timestamp)
        result.append(Cascade(cascade_id=cascade_id, events=ordered))
    return sorted(result, key=lambda item: item.cascade_id)


def normalize_cascade_times(events: List[Event], target_span=86400) -> List[Event]:
    """
    将级联时间归一化到目标时间跨度
    
    Args:
        events: 事件列表
        target_span: 目标时间跨度（秒）
    
    Returns:
        归一化后的事件列表
    """
    if not events:
        return events
    
    # 获取时间范围
    timestamps = [event.timestamp for event in events]
    min_time = min(timestamps)
    max_time = max(timestamps)
    time_span = max_time - min_time
    
    # 如果时间跨度为0，保持不变
    if time_span == 0:
        for event in events:
            event.timestamp = 0
        return events
    
    # 归一化到目标时间跨度
    for event in events:
        normalized_time = (event.timestamp - min_time) / time_span * target_span
        event.timestamp = int(normalized_time)
    
    return events

def load_wikipedia_cascades(path: str) -> List[Cascade]:
    cascades: Dict[str, List[Event]] = {}
    with open(path, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader, None)
        for row in reader:
            if len(row) < 4:
                continue
            user_id = row[0]
            item_id = row[1]
            timestamp = int(float(row[2]))
            extra_features = [float(value) for value in row[4:]]
            event = Event(
                cascade_id=item_id,
                user_id=user_id,
                timestamp=timestamp,
                extra_features=extra_features,
                event_type="interaction",
            )
            cascades.setdefault(item_id, []).append(event)

    result: List[Cascade] = []
    for cascade_id, events in cascades.items():
        ordered = sorted(events, key=lambda item: item.timestamp)
        # 归一化时间到24小时窗口
        normalized = normalize_cascade_times(ordered, target_span=86400)
        enriched = _assign_parents(normalized)
        result.append(Cascade(cascade_id=cascade_id, events=enriched, target_size=len(enriched)))
    return sorted(result, key=lambda item: item.cascade_id)


def write_sample_csv(path: str, cascades: List[Cascade]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["cascade_id", "user_id", "parent_id", "timestamp", "event_type"],
        )
        writer.writeheader()
        for cascade in cascades:
            for event in cascade.events:
                writer.writerow(
                    {
                        "cascade_id": event.cascade_id,
                        "user_id": event.user_id,
                        "parent_id": event.parent_id or "",
                        "timestamp": event.timestamp,
                        "event_type": event.event_type,
                    }
                )


def generate_synthetic_cascades(count: int = 80, seed: int = 42) -> List[Cascade]:
    random.seed(seed)
    cascades: List[Cascade] = []
    for idx in range(count):
        cascade_id = f"cascade_{idx:03d}"
        base_time = 1_700_000_000 + idx * 600
        root_user = f"user_{idx}_0"
        events = [Event(cascade_id=cascade_id, user_id=root_user, timestamp=base_time)]
        frontier = [root_user]
        active_users = [root_user]
        total_steps = random.randint(8, 70)
        for step in range(1, total_steps):
            if not frontier:
                frontier = active_users[-3:] or active_users
            if random.random() < 0.35:
                parent = random.choice(frontier)
            else:
                parent = random.choice(active_users)
            user_id = f"user_{idx}_{step}"
            gap = random.randint(20, 900 if step < 10 else 1800)
            timestamp = events[-1].timestamp + gap
            events.append(
                Event(
                    cascade_id=cascade_id,
                    user_id=user_id,
                    parent_id=parent,
                    timestamp=timestamp,
                )
            )
            active_users.append(user_id)
            if random.random() < 0.6:
                frontier.append(user_id)
            if random.random() < 0.2 and len(frontier) > 2:
                frontier.pop(0)
        cascades.append(Cascade(cascade_id=cascade_id, events=events))
    return cascades


def _assign_parents(events: List[Event]) -> List[Event]:
    if not events:
        return events

    assigned: List[Event] = []
    recent_users: List[str] = []
    for idx, event in enumerate(events):
        parent_id = None
        if idx > 0:
            if recent_users:
                parent_id = recent_users[-1]
        assigned.append(
            Event(
                cascade_id=event.cascade_id,
                user_id=event.user_id,
                timestamp=event.timestamp,
                parent_id=parent_id,
                event_type=event.event_type,
                extra_features=list(event.extra_features),
            )
        )
        recent_users.append(event.user_id)
    return assigned