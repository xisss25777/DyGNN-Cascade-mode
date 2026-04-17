import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path


class SpatioTemporalMask(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, mask_type='edge', time_aware=True, dropout=0.3):
        """
        初始化时空掩码模块
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            mask_type: 'edge'（边掩码）或'node'（节点掩码）
            time_aware: 是否使用时间感知机制
            dropout: Dropout概率，增加随机性
        """
        super(SpatioTemporalMask, self).__init__()
        self.mask_type = mask_type
        self.time_aware = time_aware
        
        # 掩码生成网络
        if mask_type == 'edge':
            # 边掩码网络：输入为边特征或节点对特征
            if time_aware:
                # 添加时间感知输入
                self.mask_net = nn.Sequential(
                    nn.Linear(input_dim * 2 + 1, hidden_dim),  # 两个节点特征 + 时间特征
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1)
                )
            else:
                self.mask_net = nn.Sequential(
                    nn.Linear(input_dim * 2, hidden_dim),  # 两个节点特征拼接
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1)
                )
        elif mask_type == 'node':
            # 节点掩码网络：输入为节点特征
            if time_aware:
                # 添加时间感知输入
                self.mask_net = nn.Sequential(
                    nn.Linear(input_dim + 1, hidden_dim),  # 节点特征 + 时间特征
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1)
                )
            else:
                self.mask_net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1)
                )
        else:
            raise ValueError(f"不支持的掩码类型: {mask_type}")
    
    def forward(self, snapshots, edge_indices=None, node_features=None, time_steps=None):
        """
        为每个时间片生成重要性掩码
        
        Args:
            snapshots: 动态图序列 [G_1, G_2, ..., G_K] 或 None
            edge_indices: 各时间片的边索引列表
            node_features: 各时间片的节点特征列表
            time_steps: 时间步长列表，用于时间感知
        
        Returns:
            masks: 掩码序列 [M_1, M_2, ..., M_K]，每个M_k形状取决于mask_type
        """
        masks = []
        
        # 确定时间片数量
        if snapshots is not None:
            num_time_steps = len(snapshots)
        elif edge_indices is not None:
            num_time_steps = len(edge_indices)
        elif node_features is not None:
            num_time_steps = len(node_features)
        else:
            raise ValueError("至少需要提供snapshots、edge_indices或node_features之一")
        
        # 生成时间特征
        if time_steps is None:
            # 如果没有提供时间步长，使用相对时间
            time_steps = torch.linspace(0, 1, num_time_steps)
        
        # 遍历每个时间片
        for i in range(num_time_steps):
            if self.mask_type == 'edge':
                # 生成边掩码
                if edge_indices is None or node_features is None:
                    raise ValueError("边掩码需要edge_indices和node_features")
                
                edge_index = edge_indices[i]
                features = node_features[i]
                
                # 获取边的两个节点的特征
                src_features = features[edge_index[0]]
                dst_features = features[edge_index[1]]
                
                # 拼接特征
                edge_features = torch.cat([src_features, dst_features], dim=1)
                
                # 添加时间特征
                if self.time_aware:
                    time_feature = torch.full((edge_features.shape[0], 1), time_steps[i], device=edge_features.device)
                    edge_features = torch.cat([edge_features, time_feature], dim=1)
                
                # 生成掩码并应用sigmoid
                raw_masks = self.mask_net(edge_features)
                
                # 调整温度参数，使分布更均匀（温度越低，分布越尖锐）
                temperature = 0.2
                edge_masks = torch.sigmoid(raw_masks / temperature).squeeze()
                
                # 调整掩码分布，增加范围
                # 先将sigmoid输出映射到更宽的范围
                edge_masks = edge_masks * 0.8  # 扩展到0-0.8范围
                
                # 添加时间片特定的偏置，确保当前时间片的边重要性略高于其他边
                # 时间步长从0到1，所以时间偏置从0到0.2
                time_bias = 0.2 * time_steps[i]
                edge_masks = edge_masks + time_bias
                
                # 确保掩码值在(0,1)之间合理分布
                edge_masks = torch.clamp(edge_masks, 0.05, 0.95)
                
                # 添加时间片特定的噪声，增加每条边的区分度
                # 增加噪声幅度，使掩码值更分散
                noise = torch.randn_like(edge_masks) * 0.1
                edge_masks = edge_masks + noise
                edge_masks = torch.clamp(edge_masks, 0.05, 0.95)  # 再次确保范围
                
                # 保持时间相关性：如果不是第一个时间片，与前一个时间片的掩码保持相似
                if i > 0 and len(masks) > 0:
                    prev_mask = masks[-1]
                    # 确保prev_mask与当前mask长度相同
                    if len(prev_mask) == len(edge_masks):
                        # 平滑过渡，当前掩码80%来自当前计算，20%来自前一个时间片
                        # 减少前一个时间片的影响，增加当前时间片的独特性
                        edge_masks = edge_masks * 0.8 + prev_mask * 0.2
                        edge_masks = torch.clamp(edge_masks, 0.05, 0.95)
                
                masks.append(edge_masks)
                
            elif self.mask_type == 'node':
                # 生成节点掩码
                if node_features is None:
                    raise ValueError("节点掩码需要node_features")
                
                features = node_features[i]
                
                # 添加时间特征
                if self.time_aware:
                    time_feature = torch.full((features.shape[0], 1), time_steps[i], device=features.device)
                    features = torch.cat([features, time_feature], dim=1)
                
                # 生成掩码并应用sigmoid
                raw_masks = self.mask_net(features)
                
                # 调整温度参数，使分布更均匀
                temperature = 0.2
                node_masks = torch.sigmoid(raw_masks / temperature).squeeze()
                
                # 调整掩码分布，增加范围
                node_masks = node_masks * 0.8  # 扩展到0-0.8范围
                
                # 添加时间片特定的偏置，确保当前时间片的节点重要性略高于其他节点
                time_bias = 0.2 * time_steps[i]
                node_masks = node_masks + time_bias
                
                # 确保掩码值在(0,1)之间合理分布
                node_masks = torch.clamp(node_masks, 0.05, 0.95)
                
                # 添加时间片特定的噪声，增加每个节点的区分度
                noise = torch.randn_like(node_masks) * 0.1
                node_masks = node_masks + noise
                node_masks = torch.clamp(node_masks, 0.05, 0.95)  # 再次确保范围
                
                # 保持时间相关性：如果不是第一个时间片，与前一个时间片的掩码保持相似
                if i > 0 and len(masks) > 0:
                    prev_mask = masks[-1]
                    # 确保prev_mask与当前mask长度相同
                    if isinstance(prev_mask, torch.Tensor) and isinstance(node_masks, torch.Tensor):
                        if prev_mask.ndim == 1 and node_masks.ndim == 1 and len(prev_mask) == len(node_masks):
                            # 平滑过渡，当前掩码80%来自当前计算，20%来自前一个时间片
                            node_masks = node_masks * 0.8 + prev_mask * 0.2
                            node_masks = torch.clamp(node_masks, 0.05, 0.95)
                
                masks.append(node_masks)
        
        return masks


def extract_key_propagation_patterns_from_mask(
    input_importance_mask_tensor: torch.Tensor,
    importance_threshold_for_selection: float,
    is_interpretation_on_edges: bool = True,
    input_graph_structure_information: Optional[Dict[str, torch.Tensor]] = None
) -> List[Dict[str, Any]]:
    """
    从解释模块输出的时空掩码张量中提取关键子图序列
    
    基于论文6.2节"时空掩码定义"的数学定义，实现关键传播模式的提取
    
    数学公式对应关系：
    - 输入掩码: ℳ = {M₁, M₂, ..., Mₖ}
    - 关键边集: Ēₖ = {(i, j) ∈ Eₖ | Mₖ(i, j) > δ}
    - 关键子图: Ĝₖ = (Vₖ, Ēₖ)
    - 关键传播模式序列: Ĝ⁽ᵀ⁾ = {Ĝ₁, ..., Ĝₖ}
    
    Args:
        input_importance_mask_tensor: 解释模块输出的重要性掩码张量
            - 边解释模式: 形状为 (num_time_steps_K, num_nodes_N, num_nodes_N)
            - 节点解释模式: 形状为 (num_time_steps_K, num_nodes_N)
        importance_threshold_for_selection: 浮点数，选择关键元素的阈值 δ ∈ [0, 1]
        is_interpretation_on_edges: 布尔值，True 表示对边解释，False 表示对节点解释
        input_graph_structure_information: 可选，包含输入图结构信息的字典
            - "edge_index_tensor": 形状 (2, num_edges_E) 的原始边索引
            - "num_nodes_per_time_step_list": 各时间片节点数列表（用于动态图）
    
    Returns:
        sequence_of_key_subgraphs: 长度为 K 的列表，每个元素是一个字典，表示第 k 个时间片的关键子图：
            {
                "selected_edges_indices": torch.Tensor,  # 形状 (2, num_selected_edges)，关键边索引
                "active_nodes_mask": torch.Tensor,       # 形状 (N,)，布尔掩码，标记活跃节点
                "time_step_index": int,                   # 当前时间片索引 k (0-based)
                "num_selected_elements": int             # 被选中的元素数量（边数或节点数）
            }
    
    Examples:
        # 边解释模式
        input_mask_from_explainer = torch.rand(5, 10, 10)  # 5个时间片，10个节点
        original_graph_edges = torch.tensor([[0,1,2,3], [1,2,3,4]])  # 4条边
        
        result_edges = extract_key_propagation_patterns_from_mask(
            input_importance_mask_tensor=input_mask_from_explainer,
            importance_threshold_for_selection=0.7,
            is_interpretation_on_edges=True,
            input_graph_structure_information={"edge_index_tensor": original_graph_edges}
        )
        
        # 节点解释模式
        node_importance_mask = torch.rand(5, 10)  # 5个时间片，10个节点
        
        result_nodes = extract_key_propagation_patterns_from_mask(
            input_importance_mask_tensor=node_importance_mask,
            importance_threshold_for_selection=0.5,
            is_interpretation_on_edges=False
        )
    """
    # 输入验证
    if not isinstance(input_importance_mask_tensor, torch.Tensor):
        raise TypeError("input_importance_mask_tensor 必须是 torch.Tensor 类型")
    
    if not (0 <= importance_threshold_for_selection <= 1):
        raise ValueError("importance_threshold_for_selection 必须在 [0, 1] 范围内")
    
    # 检查张量维度
    num_time_steps_K = input_importance_mask_tensor.shape[0]
    
    if is_interpretation_on_edges:
        if len(input_importance_mask_tensor.shape) != 3:
            raise ValueError("边解释模式下，input_importance_mask_tensor 必须是 3 维张量 (K, N, N)")
        num_nodes_N = input_importance_mask_tensor.shape[1]
        if input_importance_mask_tensor.shape[2] != num_nodes_N:
            raise ValueError("边解释模式下，input_importance_mask_tensor 必须是方阵 (K, N, N)")
    else:
        if len(input_importance_mask_tensor.shape) != 2:
            raise ValueError("节点解释模式下，input_importance_mask_tensor 必须是 2 维张量 (K, N)")
        num_nodes_N = input_importance_mask_tensor.shape[1]
    
    # 获取设备信息
    device = input_importance_mask_tensor.device
    
    # 准备返回结果
    sequence_of_key_subgraphs = []
    
    # 处理每个时间片
    for time_step_index in range(num_time_steps_K):
        # 初始化当前时间片的结果
        current_subgraph = {
            "selected_edges_indices": torch.empty((2, 0), dtype=torch.long, device=device),
            "active_nodes_mask": torch.zeros(num_nodes_N, dtype=torch.bool, device=device),
            "time_step_index": time_step_index,
            "num_selected_elements": 0
        }
        
        if is_interpretation_on_edges:
            # 边解释模式
            current_time_step_mask = input_importance_mask_tensor[time_step_index]  # 形状 (N, N)
            
            # 找到重要边
            selected_edges_mask = (current_time_step_mask > importance_threshold_for_selection)
            
            if input_graph_structure_information and "edge_index_tensor" in input_graph_structure_information:
                # 从原始边索引映射
                edge_index_tensor = input_graph_structure_information["edge_index_tensor"]
                num_edges_E = edge_index_tensor.shape[1]
                
                # 计算每条边的重要性
                edge_importance = torch.zeros(num_edges_E, device=device)
                for edge_idx in range(num_edges_E):
                    src_node = edge_index_tensor[0, edge_idx]
                    dst_node = edge_index_tensor[1, edge_idx]
                    edge_importance[edge_idx] = current_time_step_mask[src_node, dst_node]
                
                # 选择重要边
                selected_edges_indices = edge_index_tensor[:, edge_importance > importance_threshold_for_selection]
                current_subgraph["selected_edges_indices"] = selected_edges_indices
                current_subgraph["num_selected_elements"] = selected_edges_indices.shape[1]
                
                # 确定活跃节点
                if selected_edges_indices.shape[1] > 0:
                    active_nodes = torch.unique(selected_edges_indices.flatten())
                    current_subgraph["active_nodes_mask"][active_nodes] = True
            else:
                # 直接从邻接矩阵提取边索引
                selected_edge_positions = torch.nonzero(selected_edges_mask, as_tuple=False)
                if selected_edge_positions.shape[0] > 0:
                    selected_edges_indices = selected_edge_positions.t()
                    current_subgraph["selected_edges_indices"] = selected_edges_indices
                    current_subgraph["num_selected_elements"] = selected_edges_indices.shape[1]
                    
                    # 确定活跃节点
                    active_nodes = torch.unique(selected_edges_indices.flatten())
                    current_subgraph["active_nodes_mask"][active_nodes] = True
        else:
            # 节点解释模式
            current_node_importance = input_importance_mask_tensor[time_step_index]  # 形状 (N,)
            
            # 找到重要节点
            active_nodes_mask = (current_node_importance > importance_threshold_for_selection)
            current_subgraph["active_nodes_mask"] = active_nodes_mask
            current_subgraph["num_selected_elements"] = active_nodes_mask.sum().item()
            
            if input_graph_structure_information and "edge_index_tensor" in input_graph_structure_information:
                # 提取重要节点的诱导子图边
                edge_index_tensor = input_graph_structure_information["edge_index_tensor"]
                active_nodes = torch.where(active_nodes_mask)[0]
                
                if len(active_nodes) > 0:
                    # 创建活跃节点的掩码
                    active_node_set = set(active_nodes.tolist())
                    
                    # 筛选两个端点都在活跃节点中的边
                    selected_edges_indices = []
                    for edge_idx in range(edge_index_tensor.shape[1]):
                        src_node = edge_index_tensor[0, edge_idx].item()
                        dst_node = edge_index_tensor[1, edge_idx].item()
                        if src_node in active_node_set and dst_node in active_node_set:
                            selected_edges_indices.append([src_node, dst_node])
                    
                    if selected_edges_indices:
                        current_subgraph["selected_edges_indices"] = torch.tensor(selected_edges_indices, dtype=torch.long, device=device).t()
        
        # 添加到结果列表
        sequence_of_key_subgraphs.append(current_subgraph)
    
    return sequence_of_key_subgraphs


# 示例用法
if __name__ == "__main__":
    # 创建示例数据
    input_dim = 16
    hidden_dim = 64
    
    # 模拟3个时间片的动态图
    num_snapshots = 3
    num_nodes = 10
    num_edges = 20
    
    # 生成随机节点特征
    node_features = []
    for _ in range(num_snapshots):
        features = torch.randn(num_nodes, input_dim)
        node_features.append(features)
    
    # 生成随机边索引
    edge_indices = []
    for _ in range(num_snapshots):
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_indices.append(edge_index)
    
    # 模拟快照（实际应用中可能包含更多信息）
    snapshots = [None] * num_snapshots
    
    # 测试边掩码
    edge_masker = SpatioTemporalMask(input_dim, hidden_dim, mask_type='edge')
    edge_masks = edge_masker(snapshots, edge_indices, node_features)
    print("边掩码形状:")
    for i, mask in enumerate(edge_masks):
        print(f"时间片 {i+1}: {mask.shape}")
    
    # 测试节点掩码
    node_masker = SpatioTemporalMask(input_dim, hidden_dim, mask_type='node')
    node_masks = node_masker(snapshots, node_features=node_features)
    print("\n节点掩码形状:")
    for i, mask in enumerate(node_masks):
        print(f"时间片 {i+1}: {mask.shape}")
    
    # 验证掩码值范围
    print("\n掩码值范围验证:")
    for i, mask in enumerate(edge_masks):
        min_val = mask.min().item()
        max_val = mask.max().item()
        print(f"边掩码时间片 {i+1}: 最小值={min_val:.4f}, 最大值={max_val:.4f}")
    
    for i, mask in enumerate(node_masks):
        min_val = mask.min().item()
        max_val = mask.max().item()
        print(f"节点掩码时间片 {i+1}: 最小值={min_val:.4f}, 最大值={max_val:.4f}")
    
    # 测试关键传播模式提取
    print("\n测试关键传播模式提取:")
    
    # 边解释模式测试
    input_mask_from_explainer = torch.rand(5, 10, 10)  # 5个时间片，10个节点
    original_graph_edges = torch.tensor([[0,1,2,3,4,5], [1,2,3,4,5,6]])  # 6条边
    
    result_edges = extract_key_propagation_patterns_from_mask(
        input_importance_mask_tensor=input_mask_from_explainer,
        importance_threshold_for_selection=0.7,
        is_interpretation_on_edges=True,
        input_graph_structure_information={"edge_index_tensor": original_graph_edges}
    )
    
    print("边解释模式结果:")
    for i, subgraph in enumerate(result_edges):
        print(f"时间片 {i}:")
        print(f"  选中边数: {subgraph['num_selected_elements']}")
        print(f"  选中边索引: {subgraph['selected_edges_indices']}")
        print(f"  活跃节点: {torch.where(subgraph['active_nodes_mask'])[0].tolist()}")
    
    # 节点解释模式测试
    node_importance_mask = torch.rand(5, 10)  # 5个时间片，10个节点
    
    result_nodes = extract_key_propagation_patterns_from_mask(
        input_importance_mask_tensor=node_importance_mask,
        importance_threshold_for_selection=0.5,
        is_interpretation_on_edges=False
    )
    
    print("\n节点解释模式结果:")
    for i, subgraph in enumerate(result_nodes):
        print(f"时间片 {i}:")
        print(f"  选中节点数: {subgraph['num_selected_elements']}")
        print(f"  活跃节点: {torch.where(subgraph['active_nodes_mask'])[0].tolist()}")