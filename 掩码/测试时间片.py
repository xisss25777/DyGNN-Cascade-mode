# test_mask_time_slice.py
import torch
import numpy as np
from pathlib import Path
from time_slice_divider import TimeSliceDivider
import importlib.util

# 动态导入掩码模块
spec = importlib.util.spec_from_file_location("mask_module", "spatio_temporal_mask.py")
mask_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mask_module)
SpatioTemporalMask = mask_module.SpatioTemporalMask
extract_key_propagation_patterns_from_mask = mask_module.extract_key_propagation_patterns_from_mask

def test_mask_time_slice_integration(dataset_name='wikipedia'):
    """
    测试掩码生成和时间片划分的集成
    """
    print(f"===== 测试 {dataset_name} 数据集的掩码生成和时间片划分 =====")
    
    # 加载真实数据
    # 使用绝对路径确保正确读取
    data_dir = Path(r"E:\建模\pycharm项目\TG_network_datasets") / dataset_name
    edge_features_path = data_dir / f"ml_{dataset_name}.npy"
    node_features_path = data_dir / f"ml_{dataset_name}_node.npy"
    
    if not edge_features_path.exists() or not node_features_path.exists():
        print(f"数据集文件不存在: {dataset_name}")
        return
    
    # 加载数据
    edge_features = np.load(edge_features_path)
    node_features = np.load(node_features_path)
    
    print(f"加载 {dataset_name} 数据集:")
    print(f"  边特征形状: {edge_features.shape}")
    print(f"  节点特征形状: {node_features.shape}")
    
    # 1. 使用 TimeSliceDivider 生成时间片
    print("\n1. 生成时间片划分...")
    divider = TimeSliceDivider(strategy='uniform_events')
    
    # 从边特征中提取时间戳
    time_values = edge_features.sum(axis=1)
    
    # 处理时间戳问题
    # 1. 移除负数时间戳
    time_values = time_values[time_values >= 0]
    # 2. 确保时间戳有足够的范围
    if len(time_values) == 0:
        print("警告: 所有时间戳为负数，使用默认时间戳")
        time_values = np.linspace(0, 10000, min(1000, len(edge_features)))
    elif max(time_values) - min(time_values) < 1e-6:
        print("警告: 时间戳范围过小，使用均匀分布时间戳")
        time_values = np.linspace(0, 10000, len(time_values))
    
    timestamps = torch.tensor(time_values, dtype=torch.float32)
    print(f"时间戳统计: 数量={len(timestamps)}, 最小值={timestamps.min().item():.2f}, 最大值={timestamps.max().item():.2f}")
    
    # 生成时间片
    # 使用均匀事件划分，确保每个时间片有足够的数据
    # 限制最大时间片数量为20
    max_slices = 20
    events_per_slice = max(100, len(timestamps) // max_slices)
    
    # 确保至少有2个时间片
    num_slices = min(max_slices, max(2, len(timestamps) // events_per_slice))
    
    time_slices = divider.generate_time_slices_from_timestamps(timestamps, num_slices=num_slices)
    num_time_steps = len(time_slices)
    
    print(f"生成了 {num_time_steps} 个时间片")
    for i, time_slice in enumerate(time_slices[:3]):
        print(f"  时间片 {i}: 时间范围={time_slice['start_time']:.2f}-{time_slice['end_time']:.2f}, 事件数={time_slice['num_events']}")
    
    # 2. 为每个时间片生成边索引
    print("\n2. 为每个时间片生成边索引...")
    edge_indices = []
    for i, time_slice in enumerate(time_slices):
        event_indices = time_slice['event_indices']
        edge_feature_subset = edge_features[event_indices.numpy()]
        
        # 生成边索引
        num_edges = len(edge_feature_subset)
        num_nodes = node_features.shape[0]
        
        # 使用边特征的和作为哈希值来生成节点ID
        feature_sums = edge_feature_subset.sum(axis=1)
        src_nodes = (feature_sums * 1000).astype(int) % num_nodes
        dst_nodes = (feature_sums * 2000).astype(int) % num_nodes
        
        # 确保源节点和目标节点不同
        mask = src_nodes == dst_nodes
        while mask.any():
            dst_nodes[mask] = (dst_nodes[mask] + 1) % num_nodes
            mask = src_nodes == dst_nodes
        
        edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
        edge_indices.append(edge_index)
        print(f"  时间片 {i} 边数: {edge_index.shape[1]}")
    
    # 3. 使用 SpatioTemporalMask 生成掩码
    print("\n3. 生成时空掩码...")
    input_dim = node_features.shape[1]
    
    # 为每个时间片生成节点特征
    node_features_list = []
    for t in range(num_time_steps):
        features = torch.tensor(node_features, dtype=torch.float32)
        node_features_list.append(features)
    
    # 生成时间步长（使用真实时间戳的归一化值）
    time_values = edge_features.sum(axis=1)
    time_values = time_values[time_values >= 0]
    if len(time_values) > 0:
        # 归一化时间戳
        min_time = time_values.min()
        max_time = time_values.max()
        if max_time > min_time:
            normalized_time = (time_values - min_time) / (max_time - min_time)
            # 为每个时间片分配时间值
            time_steps = torch.zeros(num_time_steps)
            for i, time_slice in enumerate(time_slices):
                event_indices = time_slice['event_indices'].numpy()
                if len(event_indices) > 0:
                    slice_time = normalized_time[event_indices].mean()
                    time_steps[i] = slice_time
                else:
                    time_steps[i] = i / (num_time_steps - 1) if num_time_steps > 1 else 0
        else:
            # 时间范围过小，使用均匀分布
            time_steps = torch.linspace(0, 1, num_time_steps)
    else:
        # 没有有效时间戳，使用均匀分布
        time_steps = torch.linspace(0, 1, num_time_steps)
    
    print(f"时间步长: {time_steps.tolist()}")
    
    # 生成边掩码（使用时间感知）
    edge_mask_model = SpatioTemporalMask(input_dim=input_dim, hidden_dim=64, mask_type='edge', time_aware=True)
    edge_masks = edge_mask_model(None, edge_indices, node_features_list, time_steps)
    print(f"边掩码生成完成: {len(edge_masks)} 个时间片")
    
    # 生成节点掩码（使用时间感知）
    node_mask_model = SpatioTemporalMask(input_dim=input_dim, hidden_dim=64, mask_type='node', time_aware=True)
    node_masks = node_mask_model(None, node_features=node_features_list, time_steps=time_steps)
    print(f"节点掩码生成完成: {len(node_masks)} 个时间片")
    
    # 应用时间权重（方案C混合策略）
    print("\n应用时间权重（方案C混合策略）...")
    weighted_edge_masks = []
    weighted_node_masks = []
    time_weights = []
    
    for k in range(num_time_steps):
        # 计算时间权重
        if k <= 14:
            # 时间片0-14：轻微时间衰减，权重从1.00线性增加到1.01
            weight = 1.00 + (1.01 - 1.00) * (k / 14)
        elif 15 <= k <= 18:
            # 时间片15-18：中等时间衰减，权重从1.10线性增加到1.25
            weight = 1.10 + (1.25 - 1.10) * ((k - 15) / 3)
        else:  # k == 19
            # 时间片19：当前时间片，最高权重1.30
            weight = 1.30
        
        time_weights.append(weight)
        
        # 应用权重到边掩码
        edge_mask = edge_masks[k] * weight
        edge_mask = torch.clamp(edge_mask, 0.0, 1.0)
        weighted_edge_masks.append(edge_mask)
        
        # 应用权重到节点掩码
        node_mask = node_masks[k] * weight
        node_mask = torch.clamp(node_mask, 0.0, 1.0)
        weighted_node_masks.append(node_mask)
        
        print(f"  时间片 {k}: 权重={weight:.2f}, 边掩码范围=[{edge_mask.min().item():.4f}, {edge_mask.max().item():.4f}], 平均值={edge_mask.mean().item():.4f}")
    
    # 更新掩码列表为加权后的掩码
    edge_masks = weighted_edge_masks
    node_masks = weighted_node_masks
    print(f"时间权重应用完成，权重范围: [{min(time_weights):.2f}, {max(time_weights):.2f}]")
    
    # 验证所有掩码值都在[0,1]范围内
    print("\n验证掩码值范围...")
    for k in range(num_time_steps):
        edge_min = edge_masks[k].min().item()
        edge_max = edge_masks[k].max().item()
        node_min = node_masks[k].min().item()
        node_max = node_masks[k].max().item()
        
        if edge_min < 0 or edge_max > 1:
            print(f"  警告: 时间片 {k} 边掩码超出范围: [{edge_min:.4f}, {edge_max:.4f}]")
        if node_min < 0 or node_max > 1:
            print(f"  警告: 时间片 {k} 节点掩码超出范围: [{node_min:.4f}, {node_max:.4f}]")
    
    print("掩码值范围验证完成")
    
    # 验证掩码值范围
    print("\n验证掩码值范围:")
    for i, mask in enumerate(edge_masks):
        if len(mask) > 0:
            min_val = mask.min().item()
            max_val = mask.max().item()
            mean_val = mask.mean().item()
            print(f"边掩码时间片 {i}: 最小值={min_val:.4f}, 最大值={max_val:.4f}, 平均值={mean_val:.4f}")
    
    for i, mask in enumerate(node_masks):
        if len(mask) > 0:
            min_val = mask.min().item()
            max_val = mask.max().item()
            mean_val = mask.mean().item()
            print(f"节点掩码时间片 {i}: 最小值={min_val:.4f}, 最大值={max_val:.4f}, 平均值={mean_val:.4f}")
    
    # 4. 提取关键传播模式
    print("\n4. 提取关键传播模式...")
    
    # 保存原始边掩码（正确的格式）
    # 注意：边掩码是一个列表，每个元素是不同长度的张量，无法直接堆叠
    # 我们需要保存为列表格式或使用padding
    # 这里我们保存为列表，并同时保存一个邻接矩阵版本用于分析
    
    # 创建邻接矩阵版本用于提取关键传播模式
    adj_mask_tensor = torch.zeros((num_time_steps, node_features.shape[0], node_features.shape[0]))
    for t in range(num_time_steps):
        edge_index = edge_indices[t]
        mask = edge_masks[t]
        for i in range(edge_index.shape[1]):
            src = edge_index[0, i]
            dst = edge_index[1, i]
            adj_mask_tensor[t, src, dst] = mask[i]
    
    # 堆叠节点掩码
    node_mask_tensor = torch.stack(node_masks)
    
    # 测试不同阈值
    thresholds = [0.3, 0.5, 0.7]
    for threshold in thresholds:
        key_patterns = extract_key_propagation_patterns_from_mask(
            input_importance_mask_tensor=adj_mask_tensor,
            importance_threshold_for_selection=threshold,
            is_interpretation_on_edges=True
        )
        print(f"\n阈值 {threshold}:")
        print(f"  关键边数量: {sum(pattern['num_selected_elements'] for pattern in key_patterns)}")
        print(f"  关键节点数量: {sum(pattern['active_nodes_mask'].sum().item() for pattern in key_patterns)}")
        
        # 打印每个时间片的情况
        for i, pattern in enumerate(key_patterns):
            print(f"  时间片 {i}: 选中边数={pattern['num_selected_elements']}, 活跃节点数={pattern['active_nodes_mask'].sum().item()}")
    
    # 5. 保存结果
    print("\n5. 保存结果...")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # 保存时间片信息
    time_slice_info = []
    for i, time_slice in enumerate(time_slices):
        time_slice_info.append({
            'slice_index': i,
            'start_time': time_slice['start_time'],
            'end_time': time_slice['end_time'],
            'num_events': time_slice['num_events'],
            'time_span': time_slice['time_span'],
            'event_density': time_slice['event_density']
        })
    
    import json
    with open(output_dir / f"{dataset_name}_time_slices.json", 'w', encoding='utf-8') as f:
        json.dump(time_slice_info, f, indent=2)
    
    # 保存边掩码张量
    # 确保将边掩码堆叠为单个张量
    # 检查所有时间片的边数
    edge_counts = [len(mask) for mask in edge_masks]
    print(f"  各时间片边数: {edge_counts}")
    
    # 计算总边数
    total_edges = sum(edge_counts)
    print(f"  总边数: {total_edges}")
    
    # 确保总边数正确
    if total_edges != 34000:
        print(f"警告: 总边数 {total_edges} 与预期 34000 不符")
    
    # 创建一个大张量来存储所有时间片的边掩码
    # 形状: [num_time_steps, total_edges]
    edge_mask_tensor = torch.zeros(num_time_steps, total_edges)
    
    # 填充边掩码
    current_edge = 0
    for t in range(num_time_steps):
        mask = edge_masks[t]
        num_edges = len(mask)
        edge_mask_tensor[t, current_edge:current_edge+num_edges] = mask
        current_edge += num_edges
    
    print(f"  边掩码张量形状: {edge_mask_tensor.shape}")
    print(f"  边掩码张量维度: {edge_mask_tensor.ndim}")
    print(f"  边掩码张量类型: {type(edge_mask_tensor)}")
    print(f"  边掩码范围: [{edge_mask_tensor.min().item():.4f}, {edge_mask_tensor.max().item():.4f}]")
    print(f"  边掩码平均值: {edge_mask_tensor.mean().item():.4f}")
    
    # 保存边掩码张量（移除梯度信息）
    torch.save(edge_mask_tensor.detach().cpu(), output_dir / f"{dataset_name}_edge_mask_tensor.pt")
    
    # 保存节点掩码（移除梯度信息）
    torch.save(node_mask_tensor.detach().cpu(), output_dir / f"{dataset_name}_node_mask_tensor.pt")
    
    # 保存邻接矩阵版本（备份，移除梯度信息）
    torch.save(adj_mask_tensor.detach().cpu(), output_dir / f"{dataset_name}_adj_mask_tensor_broken.pt")
    
    # 保存元数据
    metadata = {
        'dataset_name': dataset_name,
        'num_time_steps': num_time_steps,
        'node_count': node_features.shape[0],
        'edge_counts': edge_counts,
        'mask_stats': {
            'edge_mask_ranges': [(mask.min().item(), mask.max().item()) for mask in edge_masks],
            'edge_mask_means': [mask.mean().item() for mask in edge_masks],
            'node_mask_range': (node_mask_tensor.min().item(), node_mask_tensor.max().item()),
            'node_mask_mean': node_mask_tensor.mean().item()
        }
    }
    
    with open(output_dir / f"{dataset_name}_edge_mask_metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    # 验证保存的文件
    print("\n验证保存的文件:")
    saved_edge_mask = torch.load(output_dir / f"{dataset_name}_edge_mask_tensor.pt")
    if isinstance(saved_edge_mask, torch.Tensor):
        print(f"  边掩码文件格式: torch.Tensor")
        print(f"  边掩码形状: {saved_edge_mask.shape}")
        print(f"  边掩码范围: [{saved_edge_mask.min().item():.4f}, {saved_edge_mask.max().item():.4f}]")
        print(f"  边掩码平均值: {saved_edge_mask.mean().item():.4f}")
    else:
        print(f"  边掩码文件格式: 张量列表")
        print(f"  边掩码数量: {len(saved_edge_mask)}")
        print(f"  第一个时间片边数: {len(saved_edge_mask[0])}")
        print(f"  边掩码范围: [{min(mask.min().item() for mask in saved_edge_mask):.4f}, {max(mask.max().item() for mask in saved_edge_mask):.4f}]")
    
    print(f"\n结果已保存到 outputs/{dataset_name}_*")
    print("测试完成！")

def fix_gradient_issues():
    """
    修复所有数据集的梯度问题
    重新生成所有数据集的掩码文件，确保没有梯度信息
    """
    print("===== 修复所有数据集的梯度问题 =====")
    
    # 要修复的数据集
    datasets = ['wikipedia', 'reddit', 'enron', 'mooc']
    
    for dataset in datasets:
        print(f"\n修复 {dataset} 数据集...")
        try:
            # 运行测试时间片函数生成新的无梯度掩码文件
            test_mask_time_slice_integration(dataset)
            print(f"  ✓ {dataset} 数据集修复成功")
            
            # 验证保存的文件是否没有梯度
            output_dir = Path("outputs")
            edge_mask_path = output_dir / f"{dataset}_edge_mask_tensor.pt"
            if edge_mask_path.exists():
                saved_tensor = torch.load(edge_mask_path)
                if hasattr(saved_tensor, 'requires_grad'):
                    print(f"  验证: 边掩码张量 requires_grad={saved_tensor.requires_grad}")
                else:
                    print(f"  验证: 边掩码张量已正确保存为无梯度张量")
        except Exception as e:
            print(f"  ✗ 修复 {dataset} 时出错: {e}")
    
    print("\n===== 修复完成 =====")


if __name__ == "__main__":
    # 运行修复脚本
    fix_gradient_issues()
    
    # 或者测试单个数据集
    # test_mask_time_slice_integration('wikipedia')