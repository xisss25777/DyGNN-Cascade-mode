# predict_explain.py
"""
级联预测和解释脚本
功能：
1. 加载CSV格式的级联数据
2. 构建图结构
3. 使用GCN模型预测传播路径
4. 提取关键传播边
5. 可视化关键传播路径
6. 输出预测结果和可视化图像
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class SimpleGCN(nn.Module):
    """简单的图卷积网络模型"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(SimpleGCN, self).__init__()
        self.conv1 = nn.Linear(input_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        """前向传播
        
        Args:
            x: 节点特征 [N, input_dim]
            edge_index: 边索引 [2, E]
            
        Returns:
            节点表示 [N, output_dim]
        """
        # 图卷积操作
        num_nodes = x.size(0)
        edge_index = edge_index.long()
        
        # 计算度矩阵
        degrees = torch.zeros(num_nodes, dtype=torch.float32, device=x.device)
        degrees.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1), device=x.device))
        degrees.scatter_add_(0, edge_index[1], torch.ones(edge_index.size(1), device=x.device))
        
        # 避免除以零
        degrees = torch.max(degrees, torch.ones_like(degrees))
        degrees_inv_sqrt = 1.0 / torch.sqrt(degrees)
        
        # 构建对称归一化矩阵
        row, col = edge_index
        edge_weight = degrees_inv_sqrt[row] * degrees_inv_sqrt[col]
        
        # 消息传递
        out = self.conv1(x)
        out = self.relu(out)
        
        # 聚合邻居特征 (使用更高效的实现)
        out_agg = torch.zeros_like(out)
        src_features = out[row] * edge_weight.unsqueeze(1)
        out_agg.scatter_add_(0, col.unsqueeze(1).expand_as(src_features), src_features)
        
        out = self.conv2(out_agg)
        return out


def load_cascade_data(csv_path: str) -> Dict:
    """
    加载级联数据
    
    Args:
        csv_path: CSV文件路径
    
    Returns:
        包含数据的字典
    """
    print(f"加载级联数据: {csv_path}")
    
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 检查必要的列
    required_columns = ['user_id', 'item_id', 'timestamp']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV文件缺少必要的列: {col}")
    
    # 构建节点映射（将user_id和item_id都作为节点）
    all_nodes = set(df['user_id'].unique()) | set(df['item_id'].unique())
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    idx_to_node = {i: node for node, i in node_to_idx.items()}
    
    # 按用户-项目组合生成级联ID
    # 这里简单地将每个用户作为一个级联
    cascades = {}
    for user_id, group in df.groupby('user_id'):
        cascade_id = f"user_{user_id}"
        edges = []
        edge_times = []
        
        for _, row in group.iterrows():
            src = node_to_idx[row['user_id']]
            dst = node_to_idx[row['item_id']]
            edges.append([src, dst])
            edge_times.append(row['timestamp'])
        
        # 只保留有边的级联
        if edges:
            cascades[cascade_id] = {
                'edges': torch.tensor(edges).t(),  # 转换为(2, E)格式
                'edge_times': torch.tensor(edge_times),
                'nodes': list(set([e[0] for e in edges] + [e[1] for e in edges]))
            }
    
    print(f"加载完成，共 {len(cascades)} 个级联，{len(node_to_idx)} 个节点")
    
    return {
        'cascades': cascades,
        'node_to_idx': node_to_idx,
        'idx_to_node': idx_to_node,
        'num_nodes': len(node_to_idx)
    }


def build_node_features(num_nodes: int) -> torch.Tensor:
    """
    构建节点特征
    
    Args:
        num_nodes: 节点数量
    
    Returns:
        节点特征矩阵
    """
    # 简单的one-hot特征
    features = torch.eye(num_nodes)
    return features


def predict_next_nodes(model: SimpleGCN, features: torch.Tensor, edge_index: torch.Tensor, 
                      current_nodes: List[int], top_k: int = 5) -> Dict[int, float]:
    """
    预测下一个可能传播到的节点
    
    Args:
        model: GCN模型
        features: 节点特征
        edge_index: 边索引
        current_nodes: 当前活跃节点
        top_k: 预测的节点数量
    
    Returns:
        预测节点及其概率
    """
    # 前向传播
    with torch.no_grad():
        output = model(features, edge_index)
    
    # 计算每个节点的活跃度分数
    node_scores = F.softmax(output, dim=1)[:, 0].detach().numpy()
    
    # 排除当前活跃节点
    candidate_nodes = [i for i in range(len(node_scores)) if i not in current_nodes]
    
    # 选择分数最高的节点
    candidate_scores = [(i, node_scores[i]) for i in candidate_nodes]
    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    
    # 返回前top_k个预测
    predictions = {i: score for i, score in candidate_scores[:top_k]}
    return predictions


def extract_key_edges(model: SimpleGCN, features: torch.Tensor, edge_index: torch.Tensor, 
                     target_node: int, top_k: int = 10) -> List[Tuple[int, int, float]]:
    """
    提取对预测影响最大的关键边
    
    Args:
        model: GCN模型
        features: 节点特征
        edge_index: 边索引
        target_node: 目标节点
        top_k: 关键边数量
    
    Returns:
        关键边列表 (src, dst, importance)
    """
    # 启用梯度
    features.requires_grad = True
    
    # 前向传播
    output = model(features, edge_index)
    target_score = output[target_node, 0]
    
    # 反向传播计算梯度
    target_score.backward()
    
    # 计算边的重要性（基于节点特征梯度）
    edge_importance = []
    for i in range(edge_index.size(1)):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        # 使用源节点和目标节点的梯度之和作为边重要性
        src_importance = torch.norm(features.grad[src]).item() if features.grad is not None else 0
        dst_importance = torch.norm(features.grad[dst]).item() if features.grad is not None else 0
        importance = (src_importance + dst_importance) / 2
        edge_importance.append((src, dst, importance))
    
    # 按重要性排序
    edge_importance.sort(key=lambda x: x[2], reverse=True)
    
    return edge_importance[:top_k]


def visualize_key_paths(cascade_id: str, edges: torch.Tensor, key_edges: List[Tuple[int, int, float]], 
                       idx_to_node: Dict[int, str], output_path: str):
    """
    可视化关键传播路径
    
    Args:
        cascade_id: 级联ID
        edges: 所有边
        key_edges: 关键边
        idx_to_node: 节点索引到节点ID的映射
        output_path: 输出图像路径
    """
    print(f"生成 {cascade_id} 的关键传播路径可视化...")
    
    # 创建图
    G = nx.DiGraph()
    
    # 添加所有边
    for i in range(edges.size(1)):
        src = edges[0, i].item()
        dst = edges[1, i].item()
        G.add_edge(idx_to_node[src], idx_to_node[dst])
    
    # 准备关键边的样式
    key_edge_set = set((e[0], e[1]) for e in key_edges)
    edge_colors = []
    edge_widths = []
    
    for u, v in G.edges():
        src_idx = [k for k, node_id in idx_to_node.items() if node_id == u][0]
        dst_idx = [k for k, node_id in idx_to_node.items() if node_id == v][0]
        
        if (src_idx, dst_idx) in key_edge_set:
            edge_colors.append('red')
            edge_widths.append(2.5)
        else:
            edge_colors.append('gray')
            edge_widths.append(1.0)
    
    # 绘制图
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.3, iterations=50)
    
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightblue')
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title(f'Cascade {cascade_id} - Key Propagation Paths')
    plt.axis('off')
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"可视化保存到: {output_path}")


def run_prediction_explanation(csv_path: str, output_dir: str = 'outputs'):
    """
    运行预测和解释流程
    
    Args:
        csv_path: CSV文件路径
        output_dir: 输出目录
    """
    try:
        # 创建输出目录
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 加载数据
        data = load_cascade_data(csv_path)
        cascades = data['cascades']
        node_to_idx = data['node_to_idx']
        idx_to_node = data['idx_to_node']
        num_nodes = data['num_nodes']
        
        # 构建节点特征
        features = build_node_features(num_nodes)
        
        # 初始化模型
        input_dim = num_nodes
        hidden_dim = 64
        output_dim = 1
        model = SimpleGCN(input_dim, hidden_dim, output_dim)
        
        # 预测结果文件
        prediction_file = output_dir / 'prediction.txt'
        
        with open(prediction_file, 'w', encoding='utf-8') as f:
            f.write('cascade_id,target_node,probability\n')
        
        # 处理每个级联
        for cascade_id, cascade_data in cascades.items():
            print(f"处理级联: {cascade_id}")
            
            edge_index = cascade_data['edges']
            current_nodes = cascade_data['nodes']
            
            # 预测下一个节点
            try:
                predictions = predict_next_nodes(model, features, edge_index, current_nodes)
                
                # 提取关键边
                if predictions:
                    top_pred_node = list(predictions.keys())[0]
                    key_edges = extract_key_edges(model, features, edge_index, top_pred_node)
                    
                    # 保存预测结果
                    with open(prediction_file, 'a', encoding='utf-8') as f:
                        for node_idx, prob in predictions.items():
                            node_id = idx_to_node[node_idx]
                            f.write(f"{cascade_id},{node_id},{prob:.4f}\n")
                    
                    # 可视化关键路径
                    visualization_path = output_dir / f'{cascade_id}_explanation.png'
                    visualize_key_paths(cascade_id, edge_index, key_edges, idx_to_node, visualization_path)
                else:
                    print(f"级联 {cascade_id} 没有预测结果")
            except Exception as e:
                print(f"处理级联 {cascade_id} 时出错: {e}")
                continue
        
        print(f"\n预测结果保存到: {prediction_file}")
        print("分析完成!")
    except Exception as e:
        print(f"运行过程中出错: {e}")
        import traceback
        traceback.print_exc()


def main():
    """
    主函数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='级联预测和解释')
    parser.add_argument('--input', type=str, required=True, help='输入CSV文件路径')
    parser.add_argument('--output', type=str, default='outputs', help='输出目录')
    
    args = parser.parse_args()
    
    run_prediction_explanation(args.input, args.output)


if __name__ == "__main__":
    main()