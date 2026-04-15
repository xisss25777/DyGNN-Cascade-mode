import math
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
from torch import nn

from .config import PipelineConfig
from .data import Cascade, Event
from .dynamic_graph import Snapshot, build_snapshots
from .evaluation import deletion_test, regression_metrics
from .patterns import identify_key_patterns
from .cache_utils import cache_manager


@dataclass
class GraphSnapshotData:
    node_features: torch.Tensor
    edge_index: torch.Tensor
    graph_features: torch.Tensor


@dataclass
class CascadeSequenceData:
    cascade_id: str
    snapshots: List[GraphSnapshotData]
    target: float
    raw_snapshots: List[Snapshot]


def run_dgnn_pipeline(cascades: List[Cascade], config: PipelineConfig) -> Dict[str, object]:
    dataset = build_dgnn_dataset(
        cascades,
        observation_seconds=config.observation_seconds,
        slice_seconds=config.slice_seconds,
    )
    if len(dataset) < 4:
        raise ValueError("可用级联样本过少，至少需要 4 条样本才能完成训练和测试。")

    indices = list(range(len(dataset)))
    random.seed(config.random_seed)
    random.shuffle(indices)
    split_at = max(1, int(len(indices) * (1 - config.test_ratio)))
    train_ids = indices[:split_at]
    test_ids = indices[split_at:]
    if not test_ids:
        test_ids = indices[-1:]
        train_ids = indices[:-1]

    train_data = [dataset[idx] for idx in train_ids]
    test_data = [dataset[idx] for idx in test_ids]
    input_dim = dataset[0].snapshots[0].node_features.shape[1]
    graph_dim = dataset[0].snapshots[0].graph_features.shape[0]

    model = DynamicCascadeGNN(input_dim=input_dim, graph_dim=graph_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.l2_penalty)
    loss_fn = nn.MSELoss()

    best_state = None
    best_loss = None
    # 修改后的代码
    for _ in range(config.epochs):
        random.shuffle(train_data)
        epoch_loss = 0.0
        model.train()
        for sample in train_data:
            optimizer.zero_grad()
            pred_log, attention_weights, channel_scores, spatial_masks, temporal_mask = model(sample.snapshots)
            target = torch.tensor([math.log1p(sample.target)], dtype=torch.float32)

            # 预测损失
            pred_loss = loss_fn(pred_log.view(1), target)

            # 解释损失 L_exp：掩码稀疏性正则化
            spatial_sparsity = sum([torch.mean(mask) for mask in spatial_masks]) / len(spatial_masks)
            temporal_sparsity = torch.mean(temporal_mask)
            exp_loss = 0.01 * (spatial_sparsity + temporal_sparsity)

            # 联合损失
            loss = pred_loss + exp_loss

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / max(1, len(train_data))
        if best_loss is None or avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    predictions: List[float] = []
    targets: List[float] = []
    attention_sum = None
    pattern_reports = []
    for sample in test_data:
        with torch.no_grad():
            pred_log, attention_weights, channel_scores, _, _ = model(sample.snapshots)
        prediction = max(1.0, math.expm1(pred_log.item()))
        predictions.append(prediction)
        targets.append(sample.target)

        attention_values = attention_weights.tolist()
        if attention_sum is None:
            attention_sum = [0.0] * len(attention_values)
        for idx, value in enumerate(attention_values):
            attention_sum[idx] += value

        important_slice_indices = [int(idx) for idx, value in enumerate(attention_values) if value >= max(attention_values) * 0.8]
        deletion_report = dgnn_deletion_test(model, sample, important_slice_indices)
        pattern_reports.append(
            {
                "cascade_id": sample.cascade_id,
                "prediction": round(prediction, 4),
                "patterns": identify_key_patterns(sample.cascade_id, sample.raw_snapshots, prediction)[:3],
                "deletion_test": deletion_report,
                "top_attention_slices": [
                    {"slice": idx + 1, "weight": round(value, 4)}
                    for idx, value in sorted(enumerate(attention_values), key=lambda item: item[1], reverse=True)[:3]
                ],
                "channel_importance": [
                    {"channel": item["feature"], "score": round(item["importance"], 4)}
                    for item in channel_scores[:4]
                ],
            }
        )

    metrics = regression_metrics(targets, predictions)
    avg_attention = [value / max(1, len(test_data)) for value in (attention_sum or [])]
    top_features = [
        {"feature": f"slice_{idx + 1}_attention", "importance": round(value, 6)}
        for idx, value in sorted(enumerate(avg_attention), key=lambda item: item[1], reverse=True)[:12]
    ]
    top_channels = model.channel_importance()

    return {
        "config": {
            "observation_seconds": config.observation_seconds,
            "slice_seconds": config.slice_seconds,
            "test_ratio": config.test_ratio,
            "learning_rate": config.learning_rate,
            "epochs": config.epochs,
            "l2_penalty": config.l2_penalty,
            "random_seed": config.random_seed,
            "use_log_target": True,
            "knn_neighbors": config.knn_neighbors,
            "model_type": "dgnn_gru_attention",
        },
        "sample_count": len(dataset),
        "feature_count": input_dim,
        "metrics": metrics,
        "top_features": top_features,
        "top_channels": top_channels,
        "test_reports": pattern_reports,
    }


# 修改 build_dgnn_dataset 函数
def build_dgnn_dataset(
        cascades: Sequence[Cascade],
        observation_seconds: int,
        slice_seconds: int,
) -> List[CascadeSequenceData]:
    dataset: List[CascadeSequenceData] = []
    config = type('Config', (), {
        'observation_seconds': observation_seconds,
        'slice_seconds': slice_seconds
    })()

    for cascade in cascades:
        # 使用缓存获取快照
        raw_snapshots = cache_manager.get_snapshots(cascade, config)
        if not raw_snapshots:
            continue
        graph_snapshots = [snapshot_to_graph_data(cascade, snapshot) for snapshot in raw_snapshots]
        dataset.append(
            CascadeSequenceData(
                cascade_id=cascade.cascade_id,
                snapshots=graph_snapshots,
                target=float(cascade.final_size),
                raw_snapshots=raw_snapshots,
            )
        )
    return dataset


def run_dgnn_pipeline(cascades: List[Cascade], config: PipelineConfig) -> Dict[str, object]:
    # 生成缓存键
    cache_key = f"{getattr(config, 'dataset_name', 'unknown')}_{config.epochs}_epochs"
    
    # 检查是否已经有缓存的训练结果
    cached_result = cache_manager.get_training_result(cache_key)
    if cached_result:
        print(f"使用缓存的训练结果: {cache_key}")
        return cached_result
    
    # 构建数据集
    dataset = build_dgnn_dataset(
        cascades,
        observation_seconds=config.observation_seconds,
        slice_seconds=config.slice_seconds,
    )
    if len(dataset) < 4:
        raise ValueError("可用级联样本过少，至少需要 4 条样本才能完成训练和测试。")

    # 划分训练和测试集
    indices = list(range(len(dataset)))
    random.seed(config.random_seed)
    random.shuffle(indices)
    split_at = max(1, int(len(indices) * (1 - config.test_ratio)))
    train_ids = indices[:split_at]
    test_ids = indices[split_at:]
    if not test_ids:
        test_ids = indices[-1:]
        train_ids = indices[:-1]

    train_data = [dataset[idx] for idx in train_ids]
    test_data = [dataset[idx] for idx in test_ids]
    input_dim = dataset[0].snapshots[0].node_features.shape[1]
    graph_dim = dataset[0].snapshots[0].graph_features.shape[0]

    # 初始化模型和优化器
    model = DynamicCascadeGNN(input_dim=input_dim, graph_dim=graph_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.l2_penalty)
    loss_fn = nn.MSELoss()

    # 训练循环
    best_state = None
    best_loss = None
    early_stopping_counter = 0
    early_stopping_patience = getattr(config, 'patience', 20)  # 使用配置中的早停耐心值，默认20
    
    for epoch in range(config.epochs):
        random.shuffle(train_data)
        epoch_loss = 0.0
        model.train()
        for sample in train_data:
            optimizer.zero_grad()
            pred_log, attention_weights, channel_scores, spatial_masks, temporal_mask = model(sample.snapshots)
            target = torch.tensor([math.log1p(sample.target)], dtype=torch.float32)

            # 预测损失
            pred_loss = loss_fn(pred_log.view(1), target)

            # 解释损失 L_exp：掩码稀疏性正则化
            spatial_sparsity = sum([torch.mean(mask) for mask in spatial_masks]) / len(spatial_masks)
            temporal_sparsity = torch.mean(temporal_mask)
            exp_loss = 0.01 * (spatial_sparsity + temporal_sparsity)

            # 联合损失
            loss = pred_loss + exp_loss

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / max(1, len(train_data))
        
        # 早停逻辑
        if best_loss is None or avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1} after {early_stopping_patience} epochs without improvement")
                break
        
        # 每100轮打印一次训练进度
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{config.epochs}, Loss: {avg_loss:.4f}")

    # 加载最佳模型
    if best_state is not None:
        model.load_state_dict(best_state)

    # 评估模型
    model.eval()
    predictions: List[float] = []
    targets: List[float] = []
    attention_sum = None
    pattern_reports = []
    for sample in test_data:
        with torch.no_grad():
            pred_log, attention_weights, channel_scores, _, _ = model(sample.snapshots)
        prediction = max(1.0, math.expm1(pred_log.item()))
        predictions.append(prediction)
        targets.append(sample.target)

        attention_values = attention_weights.tolist()
        if attention_sum is None:
            attention_sum = [0.0] * len(attention_values)
        for idx, value in enumerate(attention_values):
            attention_sum[idx] += value

        important_slice_indices = [int(idx) for idx, value in enumerate(attention_values) if value >= max(attention_values) * 0.8]
        deletion_report = dgnn_deletion_test(model, sample, important_slice_indices)
        pattern_reports.append(
            {
                "cascade_id": sample.cascade_id,
                "prediction": round(prediction, 4),
                "patterns": identify_key_patterns(sample.cascade_id, sample.raw_snapshots, prediction)[:3],
                "deletion_test": deletion_report,
                "top_attention_slices": [
                    {"slice": idx + 1, "weight": round(value, 4)}
                    for idx, value in sorted(enumerate(attention_values), key=lambda item: item[1], reverse=True)[:3]
                ],
                "channel_importance": [
                    {"channel": item["feature"], "score": round(item["importance"], 4)}
                    for item in channel_scores[:4]
                ],
            }
        )

    # 计算评估指标
    metrics = regression_metrics(targets, predictions)
    avg_attention = [value / max(1, len(test_data)) for value in (attention_sum or [])]
    top_features = [
        {"feature": f"slice_{idx + 1}_attention", "importance": round(value, 6)}
        for idx, value in sorted(enumerate(avg_attention), key=lambda item: item[1], reverse=True)[:12]
    ]
    top_channels = model.channel_importance()

    # 生成报告
    report = {
        "config": {
            "observation_seconds": config.observation_seconds,
            "slice_seconds": config.slice_seconds,
            "test_ratio": config.test_ratio,
            "learning_rate": config.learning_rate,
            "epochs": config.epochs,
            "l2_penalty": config.l2_penalty,
            "random_seed": config.random_seed,
            "use_log_target": True,
            "knn_neighbors": config.knn_neighbors,
            "model_type": "dgnn_gru_attention",
        },
        "sample_count": len(dataset),
        "feature_count": input_dim,
        "metrics": metrics,
        "top_features": top_features,
        "top_channels": top_channels,
        "test_reports": pattern_reports,
    }
    
    # 保存结果到缓存
    cache_manager.save_training_result(cache_key, report)
    print(f"训练结果已缓存: {cache_key}")

    return report


def snapshot_to_graph_data(cascade: Cascade, snapshot: Snapshot) -> GraphSnapshotData:
    ordered_nodes = sorted(snapshot.seen_nodes)
    node_to_idx = {node: idx for idx, node in enumerate(ordered_nodes)}
    root_user = cascade.events[0].user_id
    node_features: List[List[float]] = []
    for node in ordered_nodes:
        event = _latest_event_for_user(snapshot.new_events, cascade.events, node)
        extra = event.extra_features[:12] if event and event.extra_features else []
        extra = extra + [0.0] * (12 - len(extra))
        indegree = 0
        outdegree = len(snapshot.children_by_node.get(node, []))
        for parent, child in snapshot.edges:
            if child == node:
                indegree += 1
        depth = float(snapshot.depth_by_node.get(node, 0))
        is_root = 1.0 if node == root_user else 0.0
        is_leaf = 1.0 if outdegree == 0 else 0.0
        node_features.append(
            [
                is_root,
                is_leaf,
                depth,
                float(indegree),
                float(outdegree),
            ]
            + extra
        )

    edge_pairs = []
    for parent, child in snapshot.edges:
        if parent in node_to_idx and child in node_to_idx:
            src = node_to_idx[parent]
            dst = node_to_idx[child]
            edge_pairs.append((src, dst))
            edge_pairs.append((dst, src))
    if not edge_pairs:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()

    node_count = float(len(snapshot.seen_nodes))
    edge_count = float(len(snapshot.edges))
    active_events = float(len(snapshot.new_events))
    max_depth = float(max(snapshot.depth_by_node.values(), default=0))
    max_width = _max_width(snapshot.depth_by_node)
    density = edge_count / max(1.0, node_count * max(1.0, node_count - 1.0))
    avg_depth = sum(snapshot.depth_by_node.values()) / max(1.0, node_count)
    leaf_count = sum(1.0 for node in snapshot.seen_nodes if not snapshot.children_by_node.get(node))
    leaf_ratio = leaf_count / max(1.0, node_count)
    root_children = len(snapshot.children_by_node.get(root_user, []))
    root_influence = float(root_children) / max(1.0, edge_count)
    graph_features = torch.tensor(
        [
            node_count,
            edge_count,
            active_events,
            max_depth,
            max_width,
            density,
            avg_depth,
            leaf_ratio,
            root_influence,
        ],
        dtype=torch.float32,
    )
    return GraphSnapshotData(
        node_features=torch.tensor(node_features, dtype=torch.float32),
        edge_index=edge_index,
        graph_features=graph_features,
    )


def _latest_event_for_user(new_events: List[Event], all_events: List[Event], user_id: str) -> Event:
    for event in reversed(new_events):
        if event.user_id == user_id:
            return event
    for event in reversed(all_events):
        if event.user_id == user_id:
            return event
    return None


class GraphConvLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.neigh_linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return torch.relu(self.self_linear(x))
        src, dst = edge_index
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, x[src])
        degree = torch.zeros(x.size(0), device=x.device)
        degree.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
        degree = degree.clamp(min=1.0).unsqueeze(1)
        neigh = agg / degree
        return torch.relu(self.self_linear(x) + self.neigh_linear(neigh))


class DynamicCascadeGNN(nn.Module):
    def __init__(self, input_dim: int, graph_dim: int, hidden_dim: int = 48) -> None:
        super().__init__()
        self.conv1 = GraphConvLayer(input_dim, hidden_dim)
        self.conv2 = GraphConvLayer(hidden_dim, hidden_dim)
        self.graph_proj = nn.Linear(graph_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim * 2, hidden_dim, batch_first=True)
        self.attn = nn.Linear(hidden_dim, 1)
        # 添加时空掩码模块
        self.spatial_mask = nn.Linear(hidden_dim, hidden_dim)  # 空间掩码
        self.temporal_mask = nn.Linear(hidden_dim, 1)  # 时间掩码
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.channel_names = ["is_root", "is_leaf", "depth", "indegree", "outdegree"] + [f"extra_{idx}" for idx in range(1, 13)]

    def forward(self, snapshots: Sequence[GraphSnapshotData]) -> Tuple[
        torch.Tensor, torch.Tensor, List[Dict[str, float]], List[torch.Tensor], torch.Tensor]:
        sequence = []
        spatial_masks = []
        temporal_masks = []

        for snapshot in snapshots:
            x = snapshot.node_features
            h = self.conv1(x, snapshot.edge_index)
            h = self.conv2(h, snapshot.edge_index)

            # 计算空间掩码
            spatial_mask = torch.sigmoid(self.spatial_mask(h))
            h = h * spatial_mask
            spatial_masks.append(spatial_mask)

            pooled = h.mean(dim=0)
            graph_state = torch.relu(self.graph_proj(snapshot.graph_features))
            sequence.append(torch.cat([pooled, graph_state], dim=0))

        sequence_tensor = torch.stack(sequence, dim=0).unsqueeze(0)
        gru_out, _ = self.gru(sequence_tensor)

        # 计算时间掩码
        temp_mask = torch.sigmoid(self.temporal_mask(gru_out)).squeeze(-1)
        temporal_masks.append(temp_mask)

        scores = self.attn(gru_out).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(gru_out * weights.unsqueeze(-1) * temp_mask.unsqueeze(-1), dim=1)
        pred = self.regressor(context).squeeze(-1)
        channel_scores = self.channel_importance()

        return pred.squeeze(0), weights.squeeze(0), channel_scores, spatial_masks, temp_mask.squeeze(0)

    def channel_importance(self) -> List[Dict[str, float]]:
        weight_norms = self.conv1.self_linear.weight.abs().mean(dim=0)
        pairs = []
        for idx, name in enumerate(self.channel_names):
            pairs.append({"feature": name, "importance": round(weight_norms[idx].item(), 6)})
        pairs.sort(key=lambda item: item["importance"], reverse=True)
        return pairs[:8]


def dgnn_deletion_test(
        model: DynamicCascadeGNN,
        sample: CascadeSequenceData,
        slice_indices: List[int],
) -> Dict[str, float]:
    with torch.no_grad():
        base_log, _, _, base_spatial_masks, base_temporal_mask = model(sample.snapshots)

    # 基于掩码的反事实干预
    ablated_snapshots = []
    for idx, snapshot in enumerate(sample.snapshots):
        if idx in slice_indices:
            # 使用掩码进行更精细的干预
            ablated_snapshots.append(
                GraphSnapshotData(
                    node_features=snapshot.node_features * 0.1,  # 保留10%的特征
                    edge_index=snapshot.edge_index,
                    graph_features=snapshot.graph_features * 0.1,  # 保留10%的特征
                )
            )
        else:
            ablated_snapshots.append(snapshot)

    with torch.no_grad():
        ablated_log, _, _, ablated_spatial_masks, ablated_temporal_mask = model(ablated_snapshots)

    base_prediction = max(1.0, math.expm1(base_log.item()))
    ablated_prediction = max(1.0, math.expm1(ablated_log.item()))
    delta = base_prediction - ablated_prediction

    # 计算掩码变化
    mask_change = {
        "spatial_mask_change": sum(
            [torch.mean(torch.abs(b - a)).item() for b, a in zip(base_spatial_masks, ablated_spatial_masks)]) / len(
            base_spatial_masks),
        "temporal_mask_change": torch.mean(torch.abs(base_temporal_mask - ablated_temporal_mask)).item()
    }

    return {
        "base_prediction": round(base_prediction, 4),
        "ablated_prediction": round(ablated_prediction, 4),
        "drop_ratio": round(delta / max(1.0, base_prediction), 4),
        "delta": round(delta, 4),
        "effect_direction": "decrease" if delta >= 0 else "increase",
        "mask_change": mask_change
    }


def _max_width(depth_by_node: Dict[str, int]) -> float:
    counts: Dict[int, int] = {}
    for depth in depth_by_node.values():
        counts[depth] = counts.get(depth, 0) + 1
    return float(max(counts.values(), default=0))