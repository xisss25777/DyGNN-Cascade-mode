import math
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Optional, Any

import numpy as np
import torch
from torch import nn

from .config import PipelineConfig
from .data import Cascade, Event
from .dynamic_graph import Snapshot, build_snapshots
from .evaluation import deletion_test, regression_metrics
from .patterns import identify_key_patterns
from .cache_utils import cache_manager

# 导入时空掩码模块
from 掩码.spatio_temporal_mask import SpatioTemporalMask, extract_key_propagation_patterns_from_mask


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
    # 使用AdamW优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.l2_penalty)
    # 使用Huber损失函数，对异常值更鲁棒
    loss_fn = nn.HuberLoss()

    best_state = None
    best_loss = None
    # 修改后的代码
    for _ in range(config.epochs):
        random.shuffle(train_data)
        epoch_loss = 0.0
        model.train()
        for sample in train_data:
            optimizer.zero_grad()
            pred_log, attention_weights, channel_scores, spatial_masks, temporal_mask, edge_masks, node_masks = model(sample.snapshots)
            target = torch.tensor([math.log1p(sample.target)], dtype=torch.float32)

            # 预测损失
            pred_loss = loss_fn(pred_log.view(1), target)

            # 解释损失 L_exp：掩码稀疏性正则化
            spatial_sparsity = sum([torch.mean(mask) for mask in spatial_masks]) / len(spatial_masks)
            temporal_sparsity = torch.mean(temporal_mask)
            edge_sparsity = sum([torch.mean(mask) for mask in edge_masks]) / len(edge_masks)
            node_sparsity = sum([torch.mean(mask) for mask in node_masks]) / len(node_masks)
            exp_loss = 0.01 * (spatial_sparsity + temporal_sparsity + edge_sparsity + node_sparsity)

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
            size_pred, _, attention_weights, channel_scores, _, _, _, _ = model(sample.snapshots)
        prediction = max(1.0, math.expm1(size_pred.item()))
        predictions.append(prediction)
        targets.append(sample.target)

        attention_values = attention_weights.tolist()
        if attention_sum is None:
            attention_sum = [0.0] * len(attention_values)
        for idx, value in enumerate(attention_values):
            attention_sum[idx] += value

        # 生成解释性分析
        explanation = model.generate_explanation(sample.snapshots)
        
        important_slice_indices = [int(idx) for idx, value in enumerate(attention_values) if value >= max(attention_values) * 0.8]
        deletion_report = dgnn_deletion_test(model, sample, important_slice_indices)
        
        # 提取关键模式信息
        key_edge_patterns = []
        for t, pattern in enumerate(explanation["edge_patterns"]):
            if pattern["num_selected_elements"] > 0:
                key_edge_patterns.append({
                    "time_step": t + 1,
                    "edges": pattern["selected_edges_indices"].tolist() if pattern["selected_edges_indices"].numel() > 0 else [],
                    "active_nodes": torch.where(pattern["active_nodes_mask"])[0].tolist(),
                    "importance": pattern["num_selected_elements"]
                })
        
        key_node_patterns = []
        for t, pattern in enumerate(explanation["node_patterns"]):
            if pattern["num_selected_elements"] > 0:
                key_node_patterns.append({
                    "time_step": t + 1,
                    "active_nodes": torch.where(pattern["active_nodes_mask"])[0].tolist(),
                    "importance": pattern["num_selected_elements"]
                })
        
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
                "explanation": {
                    "key_edge_patterns": key_edge_patterns[:3],
                    "key_node_patterns": key_node_patterns[:3],
                    "temporal_weights": explanation["temporal_weights"]
                }
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
def calculate_proper_observation_window(cascade: Cascade, min_observation=21600, max_observation=604800):
    """
    根据级联的实际时间跨度计算合适的观察窗口
    
    Args:
        cascade: 级联对象
        min_observation: 最小观察窗口（秒）
        max_observation: 最大观察窗口（秒）
    
    Returns:
        observation_seconds: 观察窗口大小
        slice_seconds: 时间片大小
    """
    if not cascade.events:
        return min_observation, min_observation // 12
    
    # 获取事件时间范围
    timestamps = [event.timestamp for event in cascade.events]
    min_time = min(timestamps)
    max_time = max(timestamps)
    time_span = max_time - min_time
    
    print(f"级联 {cascade.cascade_id} 时间分析:")
    print(f"  事件数: {len(cascade.events)}")
    print(f"  时间范围: [{min_time}, {max_time}]")
    print(f"  时间跨度: {time_span}秒 ({time_span/3600:.1f}小时)")
    
    # 确定观察窗口
    if time_span <= min_observation:
        # 时间跨度小于最小观察窗口，使用完整时间
        observation_seconds = time_span
    elif time_span <= 86400:  # 小于1天
        observation_seconds = time_span
    elif time_span <= 604800:  # 小于1周
        observation_seconds = 86400  # 观察1天
    else:  # 超过1周
        observation_seconds = 604800  # 观察1周
    
    # 确保观察窗口在合理范围内
    observation_seconds = max(min_observation, min(observation_seconds, max_observation))
    
    # 计算时间片数量（保持12个时间片）
    slice_seconds = max(60, observation_seconds // 12)  # 最小60秒
    
    print(f"  观察窗口: {observation_seconds}秒 ({observation_seconds/3600:.1f}小时)")
    print(f"  时间片大小: {slice_seconds}秒 ({slice_seconds/60:.1f}分钟)")
    
    return observation_seconds, slice_seconds

def build_dgnn_dataset(
        cascades: Sequence[Cascade],
        observation_seconds: int,
        slice_seconds: int,
) -> List[CascadeSequenceData]:
    dataset: List[CascadeSequenceData] = []
    
    # 第一步：构建原始快照并收集特征
    temp_dataset = []
    for cascade in cascades:
        # 动态计算合适的观察窗口
        cascade_observation, cascade_slice = calculate_proper_observation_window(cascade)
        
        config = type('Config', (), {
            'observation_seconds': cascade_observation,
            'slice_seconds': cascade_slice
        })()
        
        # 使用缓存获取快照
        raw_snapshots = cache_manager.get_snapshots(cascade, config)
        if not raw_snapshots:
            continue
        # 先构建未缩放的图快照
        graph_snapshots = [snapshot_to_graph_data(cascade, snapshot) for snapshot in raw_snapshots]
        temp_dataset.append(
            CascadeSequenceData(
                cascade_id=cascade.cascade_id,
                snapshots=graph_snapshots,
                target=float(cascade.final_size),
                raw_snapshots=raw_snapshots,
            )
        )
    
    # 第二步：计算全局特征统计
    if temp_dataset:
        calculate_global_feature_stats(temp_dataset)
    
    # 第三步：使用全局统计重新构建数据集
    final_dataset = []
    for cascade in cascades:
        # 动态计算合适的观察窗口
        cascade_observation, cascade_slice = calculate_proper_observation_window(cascade)
        
        config = type('Config', (), {
            'observation_seconds': cascade_observation,
            'slice_seconds': cascade_slice
        })()
        
        # 使用缓存获取快照
        raw_snapshots = cache_manager.get_snapshots(cascade, config)
        if not raw_snapshots:
            continue
        # 使用全局统计构建缩放后的图快照
        graph_snapshots = [snapshot_to_graph_data(cascade, snapshot) for snapshot in raw_snapshots]
        final_dataset.append(
            CascadeSequenceData(
                cascade_id=cascade.cascade_id,
                snapshots=graph_snapshots,
                target=float(cascade.final_size),
                raw_snapshots=raw_snapshots,
            )
        )
    
    return final_dataset


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
    last_pred = 0.0  # 初始化上一个预测值
    
    for epoch in range(config.epochs):
        random.shuffle(train_data)
        epoch_loss = 0.0
        model.train()
        for sample_idx, sample in enumerate(train_data):
            optimizer.zero_grad()
            size_pred, growth_pred, attention_weights, channel_scores, spatial_masks, temporal_mask, edge_masks, node_masks = model(sample.snapshots)
            
            # 计算增长率目标
            raw_snapshots = sample.raw_snapshots
            if len(raw_snapshots) > 1:
                first_size = len(raw_snapshots[0].seen_nodes)
                last_size = len(raw_snapshots[-1].seen_nodes)
                # 改进增长率计算，避免数值过小
                growth_rate = math.log1p(last_size) - math.log1p(first_size)
            else:
                growth_rate = 0.0
            
            size_target = torch.tensor([math.log1p(sample.target)], dtype=torch.float32)
            growth_target = torch.tensor([growth_rate], dtype=torch.float32)

            # 单任务损失：专注于级联大小预测
            size_loss = loss_fn(size_pred.view(1), size_target)
            pred_loss = size_loss  # 暂时禁用多任务学习

            # 解释损失 L_exp：掩码稀疏性正则化
            spatial_sparsity = sum([torch.mean(mask) for mask in spatial_masks]) / len(spatial_masks)
            temporal_sparsity = torch.mean(temporal_mask)
            exp_loss = 0.01 * (spatial_sparsity + temporal_sparsity)

            # 添加注意力多样性损失
            diversity_loss = getattr(model, 'attention_diversity_loss', 0.0)
            attention_diversity_loss = 0.001 * diversity_loss  # 再降10倍，避免过度正则化

            # 添加时间注意力平衡损失，防止过度关注单个时间片
            attention_balance_loss = torch.var(attention_weights) * 0.001  # 再降10倍，避免过度正则化

            # 添加batch级别的预测多样性损失
            if sample_idx % 32 == 31:  # 每32个样本计算一次
                # 简单的多样性损失：鼓励预测值有合理的范围
                if hasattr(model, 'prediction_buffer'):
                    if len(model.prediction_buffer) > 0:
                        pred_std = torch.std(torch.tensor(model.prediction_buffer))
                        target_std = torch.tensor(15.0, device=size_pred.device)
                        diversity_promoting_loss = torch.nn.functional.mse_loss(pred_std, target_std) * 0.001
                    else:
                        diversity_promoting_loss = 0.0
                else:
                    model.prediction_buffer = []
                    diversity_promoting_loss = 0.0
            else:
                diversity_promoting_loss = 0.0
                # 累积预测值
                if not hasattr(model, 'prediction_buffer'):
                    model.prediction_buffer = []
                model.prediction_buffer.append(size_pred.item())
                # 保持buffer大小
                if len(model.prediction_buffer) > 64:
                    model.prediction_buffer = model.prediction_buffer[-64:]

            # 联合损失
            loss = pred_loss + exp_loss + attention_diversity_loss + attention_balance_loss + diversity_promoting_loss
            
            # 梯度检查（每100个样本打印一次）
            if (epoch + 1) % 100 == 0 and sample_idx % 100 == 0:
                print(f"Epoch {epoch+1}, Sample {sample_idx}, Loss: {loss.item():.4f}, Gradient norms:")
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        if grad_norm > 0.0:
                            print(f"  {name}: {grad_norm:.4f}")

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
            size_pred, _, attention_weights, channel_scores, _, _, _, _ = model(sample.snapshots)
        prediction = max(1.0, math.expm1(size_pred.item()))
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


# 全局特征统计（用于标准化）
_global_feature_stats = {
    'node': {'mean': 0.0, 'std': 1.0},
    'graph': {'mean': 0.0, 'std': 1.0}
}


def calculate_global_feature_stats(dataset):
    """
    计算全局特征统计信息
    """
    node_features_list = []
    graph_features_list = []
    
    for sample in dataset:
        for snapshot in sample.snapshots:
            if snapshot.node_features is not None and snapshot.node_features.numel() > 0:
                node_features_list.append(snapshot.node_features.numpy())
            if snapshot.graph_features is not None and snapshot.graph_features.numel() > 0:
                graph_features_list.append(snapshot.graph_features.numpy())
    
    if node_features_list:
        all_node_features = np.concatenate(node_features_list, axis=0)
        _global_feature_stats['node']['mean'] = all_node_features.mean()
        _global_feature_stats['node']['std'] = all_node_features.std()
    
    if graph_features_list:
        all_graph_features = np.concatenate(graph_features_list, axis=0)
        _global_feature_stats['graph']['mean'] = all_graph_features.mean()
        _global_feature_stats['graph']['std'] = all_graph_features.std()
    
    print(f"全局特征统计:")
    print(f"  节点特征: 均值={_global_feature_stats['node']['mean']:.4f}, 标准差={_global_feature_stats['node']['std']:.4f}")
    print(f"  图特征: 均值={_global_feature_stats['graph']['mean']:.4f}, 标准差={_global_feature_stats['graph']['std']:.4f}")


def rescale_features(features, feature_type='node', target_mean=50, target_std=25):
    """
    将特征缩放到合理的范围
    目标：均值50，标准差25，匹配标签范围
    """
    # 1. 使用全局统计信息
    current_mean = _global_feature_stats[feature_type]['mean']
    current_std = _global_feature_stats[feature_type]['std']
    
    # 2. 标准化到N(0,1)
    features_normalized = (features - current_mean) / (current_std + 1e-8)
    
    # 3. 缩放到目标范围
    features_rescaled = features_normalized * target_std + target_mean
    
    return features_rescaled


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
    
    # 添加增强特征
    degrees = [len(snapshot.children_by_node.get(node, [])) for node in snapshot.seen_nodes]
    avg_degree = np.mean(degrees) if degrees else 0.0
    std_degree = np.std(degrees) if degrees else 0.0
    max_degree = np.max(degrees) if degrees else 0.0
    
    enhanced_features = [
        node_count,              # 节点数
        edge_count,              # 边数
        density,                 # 边密度
        active_events,            # 活跃事件数
        active_events / max(1.0, node_count),  # 活跃度
        max_depth,               # 最大深度
        avg_depth,                # 平均深度
        leaf_count,               # 叶子节点数
        leaf_ratio,               # 叶子比例
        avg_degree,               # 平均度
        std_degree,               # 度标准差
        max_degree,               # 最大度
    ]
    
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
    
    # 将增强特征添加到节点特征中
    node_features_array = np.array(node_features)
    enhanced_features_array = np.tile(np.array(enhanced_features), (node_features_array.shape[0], 1))
    node_features_enhanced = np.column_stack([node_features_array, enhanced_features_array])
    
    # 特征重缩放
    node_features_rescaled = rescale_features(node_features_enhanced, feature_type='node', target_mean=50, target_std=25)
    graph_features_array = np.array([
        node_count,
        edge_count,
        active_events,
        max_depth,
        max_width,
        density,
        avg_depth,
        leaf_ratio,
        root_influence,
    ])
    graph_features_rescaled = rescale_features(graph_features_array, feature_type='graph', target_mean=50, target_std=25)
    
    return GraphSnapshotData(
        node_features=torch.tensor(node_features_rescaled, dtype=torch.float32),
        edge_index=edge_index,
        graph_features=torch.tensor(graph_features_rescaled, dtype=torch.float32),
    )


def _latest_event_for_user(new_events: List[Event], all_events: List[Event], user_id: str) -> Event:
    """
    只从当前时间片的新事件中查找用户的最新事件
    不使用 all_events，确保时间片特征的正确性
    """
    for event in reversed(new_events):
        if event.user_id == user_id:
            return event
    # 只返回当前时间片的事件，不回退到整个级联
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
    def __init__(self, input_dim: int, graph_dim: int, hidden_dim: int = 128) -> None:  # 恢复适中的模型容量
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.2)
        self.conv1 = GraphConvLayer(input_dim, hidden_dim)
        self.conv2 = GraphConvLayer(hidden_dim, hidden_dim)
        self.graph_proj = nn.Linear(graph_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim * 2, hidden_dim, batch_first=True)
        # 改进注意力机制
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        # 位置编码层
        self.position_embedding = nn.Embedding(100, hidden_dim * 2)
        # 使用高级时空掩码模块
        self.spatio_temporal_mask = SpatioTemporalMask(input_dim, hidden_dim=64, mask_type='edge', time_aware=True)
        self.node_mask = SpatioTemporalMask(input_dim, hidden_dim=64, mask_type='node', time_aware=True)
        # 多任务学习：预测级联大小和增长率
        self.size_regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # 使用Softplus避免梯度消失
        )
        self.growth_regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # 使用Softplus避免梯度消失
        )
        self.channel_names = ["is_root", "is_leaf", "depth", "indegree", "outdegree"] + [f"extra_{idx}" for idx in range(1, 13)]

    def forward(self, snapshots: Sequence[GraphSnapshotData]) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, List[Dict[str, float]], List[torch.Tensor], torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        sequence = []
        spatial_masks = []
        temporal_masks = []
        edge_masks = []
        node_masks = []

        # 准备边索引和节点特征列表
        edge_indices = []
        node_features_list = []
        for snapshot in snapshots:
            edge_indices.append(snapshot.edge_index)
            node_features_list.append(snapshot.node_features)

        # 生成时空掩码
        edge_masks = self.spatio_temporal_mask(None, edge_indices, node_features_list)
        node_masks = self.node_mask(None, node_features=node_features_list)

        for i, snapshot in enumerate(snapshots):
            x = snapshot.node_features
            
            # 应用节点掩码
            if i < len(node_masks):
                node_mask = node_masks[i]
                if isinstance(node_mask, torch.Tensor) and node_mask.ndim == 1 and len(node_mask) == len(x):
                    x = x * node_mask.unsqueeze(1)
            
            h = self.conv1(x, snapshot.edge_index)
            h = self.dropout(h)
            h = self.conv2(h, snapshot.edge_index)
            h = self.dropout(h)

            # 计算空间掩码
            spatial_mask = torch.ones_like(h)
            if i < len(edge_masks) and snapshot.edge_index.numel() > 0:
                # 应用边掩码到节点表示
                edge_mask = edge_masks[i]
                if len(edge_mask) == snapshot.edge_index.shape[1]:
                    # 根据边掩码加权聚合
                    src, dst = snapshot.edge_index
                    agg = torch.zeros_like(h)
                    agg.index_add_(0, dst, h[src] * edge_mask.unsqueeze(1))
                    degree = torch.zeros(h.size(0), device=h.device)
                    degree.index_add_(0, dst, edge_mask)
                    degree = degree.clamp(min=1.0).unsqueeze(1)
                    spatial_mask = agg / degree
            
            h = h * spatial_mask
            spatial_masks.append(spatial_mask)

            pooled = h.mean(dim=0)
            graph_state = torch.relu(self.graph_proj(snapshot.graph_features))
            sequence.append(torch.cat([pooled, graph_state], dim=0))

        # 添加位置编码
        seq_len = len(sequence)
        positions = torch.arange(0, seq_len, device=sequence[0].device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)
        
        sequence_tensor = torch.stack(sequence, dim=0).unsqueeze(0)
        sequence_tensor = sequence_tensor + pos_emb  # 添加位置编码
        
        gru_out, _ = self.gru(sequence_tensor)

        # 计算时间掩码
        temp_mask = torch.sigmoid(torch.nn.Linear(self.hidden_dim, 1)(gru_out)).squeeze(-1)
        temporal_masks.append(temp_mask)

        # 改进注意力机制 - 添加多样性正则化
        scores = self.attn(gru_out).squeeze(-1)
        
        # 添加增强的位置偏置（鼓励关注近期时间片）
        seq_len = scores.shape[1]
        # 创建线性递增的位置权重，近期时间片权重更高
        position_weights = torch.linspace(1.0, 3.0, seq_len, device=scores.device)
        position_bias = position_weights * 1.0  # 增加偏置强度到1.0
        scores = scores + position_bias
        
        # 添加温度参数，控制注意力分布的尖锐程度
        temperature = 0.5  # 进一步降低温度，增加区分度
        scores = scores / temperature
        
        weights = torch.softmax(scores, dim=1)
        
        # 添加注意力多样性损失（在训练时）
        if self.training:
            # 鼓励注意力分布更加多样化
            # 1. 熵正则化（鼓励分布均匀）
            entropy = -torch.sum(weights * torch.log(weights + 1e-10), dim=1)
            max_entropy = torch.log(torch.tensor(scores.shape[1], dtype=torch.float, device=scores.device))
            entropy_loss = (1.0 - entropy / max_entropy).mean()
            
            # 2. 稀疏性正则化（惩罚过度稀疏）
            sparsity_loss = torch.sum(weights**2, dim=1).mean()
            
            # 3. 组合损失
            diversity_loss = entropy_loss + sparsity_loss
            self.attention_diversity_loss = diversity_loss
        else:
            self.attention_diversity_loss = 0.0
        
        context = torch.sum(gru_out * weights.unsqueeze(-1) * temp_mask.unsqueeze(-1), dim=1)
        
        # 多任务学习：预测级联大小和增长率
        size_pred = self.size_regressor(context).squeeze(-1)
        growth_pred = self.growth_regressor(context).squeeze(-1)
        channel_scores = self.channel_importance()

        return size_pred.squeeze(0), growth_pred.squeeze(0), weights.squeeze(0), channel_scores, spatial_masks, temp_mask.squeeze(0), edge_masks, node_masks

    def channel_importance(self) -> List[Dict[str, float]]:
        weight_norms = self.conv1.self_linear.weight.abs().mean(dim=0)
        pairs = []
        for idx, name in enumerate(self.channel_names):
            pairs.append({"feature": name, "importance": round(weight_norms[idx].item(), 6)})
        pairs.sort(key=lambda item: item["importance"], reverse=True)
        return pairs[:8]
    
    def generate_explanation(self, snapshots: Sequence[GraphSnapshotData]) -> Dict[str, Any]:
        """
        生成解释性分析结果
        """
        with torch.no_grad():
            size_pred, growth_pred, weights, channel_scores, spatial_masks, temporal_mask, edge_masks, node_masks = self(snapshots)
        
        # 准备边索引和节点特征
        edge_indices = [snapshot.edge_index for snapshot in snapshots]
        node_features_list = [snapshot.node_features for snapshot in snapshots]
        
        # 构建掩码张量用于模式提取
        num_time_steps = len(snapshots)
        max_nodes = max([sf.node_features.shape[0] for sf in snapshots])
        
        # 构建边掩码张量 (K, N, N)
        edge_mask_tensor = torch.zeros((num_time_steps, max_nodes, max_nodes), device=snapshots[0].node_features.device)
        for t, (edge_idx, mask) in enumerate(zip(edge_indices, edge_masks)):
            if edge_idx.numel() > 0 and len(mask) == edge_idx.shape[1]:
                src, dst = edge_idx
                edge_mask_tensor[t, src, dst] = mask
        
        # 构建节点掩码张量 (K, N)
        node_mask_tensor = torch.zeros((num_time_steps, max_nodes), device=snapshots[0].node_features.device)
        for t, (features, mask) in enumerate(zip(node_features_list, node_masks)):
            if isinstance(mask, torch.Tensor) and mask.ndim == 1 and len(mask) == features.shape[0]:
                node_mask_tensor[t, :features.shape[0]] = mask
        
        # 提取关键传播模式
        edge_patterns = extract_key_propagation_patterns_from_mask(
            edge_mask_tensor,
            importance_threshold_for_selection=0.6,
            is_interpretation_on_edges=True
        )
        
        node_patterns = extract_key_propagation_patterns_from_mask(
            node_mask_tensor,
            importance_threshold_for_selection=0.6,
            is_interpretation_on_edges=False
        )
        
        return {
            "edge_patterns": edge_patterns,
            "node_patterns": node_patterns,
            "temporal_weights": weights.tolist(),
            "channel_importance": channel_scores
        }


def dgnn_deletion_test(
        model: DynamicCascadeGNN,
        sample: CascadeSequenceData,
        slice_indices: List[int],
) -> Dict[str, float]:
    with torch.no_grad():
        base_size_pred, _, _, _, base_spatial_masks, base_temporal_mask, base_edge_masks, base_node_masks = model(sample.snapshots)

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
        ablated_size_pred, _, _, _, ablated_spatial_masks, ablated_temporal_mask, ablated_edge_masks, ablated_node_masks = model(ablated_snapshots)

    base_prediction = max(1.0, math.expm1(base_size_pred.item()))
    ablated_prediction = max(1.0, math.expm1(ablated_size_pred.item()))
    delta = base_prediction - ablated_prediction

    # 计算掩码变化
    mask_change = {
        "spatial_mask_change": sum(
            [torch.mean(torch.abs(b - a)).item() for b, a in zip(base_spatial_masks, ablated_spatial_masks)]) / len(
            base_spatial_masks),
        "temporal_mask_change": torch.mean(torch.abs(base_temporal_mask - ablated_temporal_mask)).item(),
        "edge_mask_change": sum(
            [torch.mean(torch.abs(b - a)).item() for b, a in zip(base_edge_masks, ablated_edge_masks)]) / len(
            base_edge_masks),
        "node_mask_change": sum(
            [torch.mean(torch.abs(b - a)).item() for b, a in zip(base_node_masks, ablated_node_masks)]) / len(
            base_node_masks)
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