import torch
import numpy as np
import os
import sys
import importlib.util
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入111.py模块
spec = importlib.util.spec_from_file_location("mask_module", "spatio_temporal_mask.py")
mask_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mask_module)
SpatioTemporalMask = mask_module.SpatioTemporalMask
extract_key_propagation_patterns_from_mask = mask_module.extract_key_propagation_patterns_from_mask


def load_dataset(dataset_name):
    """
    加载TG_network_datasets中的数据集
    
    Args:
        dataset_name: 数据集名称，如 'wikipedia', 'reddit' 等
    
    Returns:
        edge_features: 边特征 numpy 数组
        node_features: 节点特征 numpy 数组
    """
    data_dir = Path("E:\建模\pycharm项目\TG_network_datasets") / dataset_name
    
    # 加载边特征和节点特征
    edge_features_path = data_dir / f"ml_{dataset_name}.npy"
    node_features_path = data_dir / f"ml_{dataset_name}_node.npy"
    
    if not edge_features_path.exists() or not node_features_path.exists():
        raise FileNotFoundError(f"数据集文件不存在: {dataset_name}")
    
    edge_features = np.load(edge_features_path)
    node_features = np.load(node_features_path)
    
    print(f"加载 {dataset_name} 数据集:")
    print(f"  边特征形状: {edge_features.shape}")
    print(f"  节点特征形状: {node_features.shape}")
    
    return edge_features, node_features


def create_dynamic_graph(edge_features, node_features, num_time_steps=5, dataset_name=None):
    """
    创建动态图序列
    
    Args:
        edge_features: 边特征数组
        node_features: 节点特征数组
        num_time_steps: 时间片数量
        dataset_name: 数据集名称，用于动态调整时间片数量
    
    Returns:
        snapshots: 动态图序列
        edge_indices: 各时间片的边索引列表
        node_features_list: 各时间片的节点特征列表
    """
    num_nodes = node_features.shape[0]
    num_edges = edge_features.shape[0]
    
    # 根据数据集动态调整时间片数量
    if dataset_name:
        # 基于数据集大小动态确定时间片数量
        if num_edges > 10000:
            num_time_steps = 10
        elif num_edges > 5000:
            num_time_steps = 8
        elif num_edges > 1000:
            num_time_steps = 6
        # 否则使用默认值5
    
    print(f"动态确定时间片数量: {num_time_steps}")
    
    # 从真实边特征中生成边索引
    edge_indices = []
    edges_per_time = num_edges // num_time_steps
    
    for t in range(num_time_steps):
        # 从真实边特征中选择连续的边
        start_idx = t * edges_per_time
        end_idx = (t + 1) * edges_per_time if t < num_time_steps - 1 else num_edges
        
        # 生成边索引（使用节点ID的哈希值作为源节点和目标节点）
        # 这里使用边特征的哈希值来生成合理的边索引
        edge_feature_subset = edge_features[start_idx:end_idx]
        num_edges_per_time = len(edge_feature_subset)
        
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
        print(f"  时间片 {t} 边数: {edge_index.shape[1]}")
    
    # 为每个时间片创建节点特征（使用真实节点特征）
    node_features_list = []
    for t in range(num_time_steps):
        # 使用真实节点特征，不添加噪声
        features = torch.tensor(node_features, dtype=torch.float32)
        node_features_list.append(features)
    
    # 创建快照列表
    snapshots = [None] * num_time_steps
    
    return snapshots, edge_indices, node_features_list


def process_dataset(dataset_name):
    """
    处理数据集并提取关键传播模式
    
    Args:
        dataset_name: 数据集名称
    """
    print(f"\n===== 处理 {dataset_name} 数据集 =====")
    
    # 加载数据集
    edge_features, node_features = load_dataset(dataset_name)
    
    # 创建动态图（不指定时间片数量，让函数动态确定）
    snapshots, edge_indices, node_features_list = create_dynamic_graph(
        edge_features, node_features, dataset_name=dataset_name
    )
    num_time_steps = len(snapshots)
    
    # 提取特征维度
    input_dim = node_features.shape[1]
    print(f"\n特征维度: {input_dim}")
    
    # 创建SpatioTemporalMask实例
    edge_masker = SpatioTemporalMask(input_dim, hidden_dim=64, mask_type='edge')
    node_masker = SpatioTemporalMask(input_dim, hidden_dim=64, mask_type='node')
    
    # 生成边掩码
    print("\n生成边掩码...")
    edge_masks = edge_masker(snapshots, edge_indices, node_features_list)
    
    # 生成节点掩码
    print("生成节点掩码...")
    node_masks = node_masker(snapshots, node_features=node_features_list)
    
    # 验证掩码值范围
    print("\n验证掩码值范围:")
    for i, mask in enumerate(edge_masks):
        min_val = mask.min().item()
        max_val = mask.max().item()
        mean_val = mask.mean().item()
        std_val = mask.std().item()
        print(f"边掩码时间片 {i+1}: 最小值={min_val:.4f}, 最大值={max_val:.4f}, 平均值={mean_val:.4f}, 标准差={std_val:.4f}")
    
    for i, mask in enumerate(node_masks):
        min_val = mask.min().item()
        max_val = mask.max().item()
        mean_val = mask.mean().item()
        std_val = mask.std().item()
        print(f"节点掩码时间片 {i+1}: 最小值={min_val:.4f}, 最大值={max_val:.4f}, 平均值={mean_val:.4f}, 标准差={std_val:.4f}")
    
    # 检查掩码是否不同
    print("\n检查掩码差异性:")
    for i in range(1, num_time_steps):
        edge_mask_diff = torch.abs(edge_masks[i] - edge_masks[0]).mean().item()
        node_mask_diff = torch.abs(node_masks[i] - node_masks[0]).mean().item()
        print(f"时间片 0 与 {i} 的边掩码差异: {edge_mask_diff:.4f}")
        print(f"时间片 0 与 {i} 的节点掩码差异: {node_mask_diff:.4f}")
    
    # 构建边掩码张量用于提取关键传播模式
    print("\n构建边掩码张量...")
    num_nodes = node_features.shape[0]
    edge_mask_tensor = torch.zeros((num_time_steps, num_nodes, num_nodes))
    
    for t in range(num_time_steps):
        edge_index = edge_indices[t]
        mask = edge_masks[t]
        for i in range(edge_index.shape[1]):
            src = edge_index[0, i]
            dst = edge_index[1, i]
            edge_mask_tensor[t, src, dst] = mask[i]
    
    # 构建节点掩码张量
    print("构建节点掩码张量...")
    node_mask_tensor = torch.stack(node_masks)
    
    # 保存掩码张量到文件，以便后续分析
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    torch.save(edge_mask_tensor, output_dir / f"{dataset_name}_edge_mask_tensor.pt")
    torch.save(node_mask_tensor, output_dir / f"{dataset_name}_node_mask_tensor.pt")
    print(f"掩码张量已保存到 outputs/{dataset_name}_*_mask_tensor.pt")
    
    # 测试不同阈值的效果
    print("\n测试不同阈值的效果:")
    thresholds = [0.3, 0.5, 0.7]
    
    for threshold in thresholds:
        print(f"\n阈值: {threshold}")
        
        # 提取关键传播模式（边解释模式）
        edge_patterns = extract_key_propagation_patterns_from_mask(
            input_importance_mask_tensor=edge_mask_tensor,
            importance_threshold_for_selection=threshold,
            is_interpretation_on_edges=True,
            input_graph_structure_information={"edge_index_tensor": edge_indices[0]}
        )
        
        # 提取关键传播模式（节点解释模式）
        node_patterns = extract_key_propagation_patterns_from_mask(
            input_importance_mask_tensor=node_mask_tensor,
            importance_threshold_for_selection=threshold,
            is_interpretation_on_edges=False
        )
        
        # 计算统计信息
        total_selected_edges = sum(p['num_selected_elements'] for p in edge_patterns)
        total_selected_nodes = sum(p['num_selected_elements'] for p in node_patterns)
        
        print(f"  总选中边数: {total_selected_edges}")
        print(f"  总选中节点数: {total_selected_nodes}")
        print(f"  平均每个时间片选中边数: {total_selected_edges / num_time_steps:.2f}")
        print(f"  平均每个时间片选中节点数: {total_selected_nodes / num_time_steps:.2f}")
    
    # 使用最佳阈值（默认0.5）提取最终结果
    best_threshold = 0.5
    print(f"\n使用最佳阈值 {best_threshold} 提取最终结果...")
    edge_patterns = extract_key_propagation_patterns_from_mask(
        input_importance_mask_tensor=edge_mask_tensor,
        importance_threshold_for_selection=best_threshold,
        is_interpretation_on_edges=True,
        input_graph_structure_information={"edge_index_tensor": edge_indices[0]}
    )
    
    node_patterns = extract_key_propagation_patterns_from_mask(
        input_importance_mask_tensor=node_mask_tensor,
        importance_threshold_for_selection=best_threshold,
        is_interpretation_on_edges=False
    )
    
    # 输出结果
    print("\n===== 结果分析 =====")
    print("边解释模式结果:")
    for i, pattern in enumerate(edge_patterns):
        print(f"时间片 {i}:")
        print(f"  选中边数: {pattern['num_selected_elements']}")
        print(f"  活跃节点数: {pattern['active_nodes_mask'].sum().item()}")
    
    print("\n节点解释模式结果:")
    for i, pattern in enumerate(node_patterns):
        print(f"时间片 {i}:")
        print(f"  选中节点数: {pattern['num_selected_elements']}")
    
    # 计算统计信息
    total_selected_edges = sum(p['num_selected_elements'] for p in edge_patterns)
    total_selected_nodes = sum(p['num_selected_elements'] for p in node_patterns)
    
    print("\n===== 统计信息 =====")
    print(f"总选中边数: {total_selected_edges}")
    print(f"总选中节点数: {total_selected_nodes}")
    print(f"平均每个时间片选中边数: {total_selected_edges / num_time_steps:.2f}")
    print(f"平均每个时间片选中节点数: {total_selected_nodes / num_time_steps:.2f}")


if __name__ == "__main__":
    # 处理多个数据集
    datasets = ['wikipedia', 'reddit', 'enron', 'mooc']
    
    for dataset in datasets:
        try:
            process_dataset(dataset)
        except FileNotFoundError as e:
            print(f"跳过 {dataset}: {e}")
        except Exception as e:
            print(f"处理 {dataset} 时出错: {e}")
        print("\n" + "="*60 + "\n")
    
    # 处理hyperparameter_search_results.json数据
    print("\n===== 处理 hyperparameter_search_results.json 数据 =====")
    try:
        import json
        
        # 加载JSON文件
        json_path = Path("../cascade_model/outputs/hyperparameter_search_results.json")
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            print(f"加载了 {len(results)} 条超参数搜索结果")
            print("最佳结果:")
            best_result = results[0] if results else None
            if best_result:
                print(f"  学习率: {best_result['learning_rate']}")
                print(f"  批次大小: {best_result['batch_size']}")
                print(f"  隐藏层维度: {best_result['hidden_dim']}")
                print(f"  训练轮数: {best_result['epochs']}")
                print(f"  MAE: {best_result['metrics']['mae']}")
                print(f"  RMSE: {best_result['metrics']['rmse']}")
                print(f"  MAPE: {best_result['metrics']['mape']}")
        else:
            print("hyperparameter_search_results.json 文件不存在")
    except Exception as e:
        print(f"处理JSON数据时出错: {e}")
    print("\n" + "="*60 + "\n")