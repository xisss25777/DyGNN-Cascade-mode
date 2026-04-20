# D:\Users\HP\Documents\GitHub\DyGNN-Cascade-mode\advanced_feature_engineering.py
import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cascade_model.data import Cascade
from cascade_model.dynamic_graph import build_snapshots
from cascade_model.config import PipelineConfig


def extract_advanced_features(cascade: Cascade, config: PipelineConfig):
    """提取高级特征"""
    # 构建快照
    snapshots = build_snapshots(
        cascade, 
        observation_seconds=config.observation_seconds, 
        slice_seconds=config.slice_seconds
    )
    
    if not snapshots:
        return []
    
    advanced_features = []
    
    # 1. 时间序列特征
    time_series_features = extract_time_series_features(snapshots)
    advanced_features.extend(time_series_features)
    
    # 2. 图结构特征
    graph_structure_features = extract_graph_structure_features(snapshots)
    advanced_features.extend(graph_structure_features)
    
    # 3. 传播模式特征
    propagation_features = extract_propagation_features(snapshots)
    advanced_features.extend(propagation_features)
    
    # 4. 统计特征
    statistical_features = extract_statistical_features(snapshots)
    advanced_features.extend(statistical_features)
    
    return advanced_features


def extract_time_series_features(snapshots):
    """提取时间序列特征"""
    features = []
    
    # 节点数时间序列
    node_counts = [len(snapshot.seen_nodes) for snapshot in snapshots]
    features.extend([
        np.mean(node_counts),           # 平均节点数
        np.std(node_counts),            # 节点数标准差
        np.max(node_counts),            # 最大节点数
        np.min(node_counts),            # 最小节点数
    ])
    
    # 边数时间序列
    edge_counts = [len(snapshot.edges) for snapshot in snapshots]
    features.extend([
        np.mean(edge_counts),           # 平均边数
        np.std(edge_counts),            # 边数标准差
        np.max(edge_counts),            # 最大边数
    ])
    
    # 活跃事件数时间序列
    active_counts = [len(snapshot.new_events) for snapshot in snapshots]
    features.extend([
        np.mean(active_counts),          # 平均活跃事件数
        np.std(active_counts),           # 活跃事件数标准差
        np.max(active_counts),           # 最大活跃事件数
    ])
    
    # 增长率特征
    if len(node_counts) > 1:
        growth_rates = []
        for i in range(1, len(node_counts)):
            if node_counts[i-1] > 0:
                growth_rates.append((node_counts[i] - node_counts[i-1]) / node_counts[i-1])
        features.extend([
            np.mean(growth_rates) if growth_rates else 0,  # 平均增长率
            np.std(growth_rates) if growth_rates else 0,   # 增长率标准差
            np.max(growth_rates) if growth_rates else 0,   # 最大增长率
        ])
    else:
        features.extend([0, 0, 0])
    
    return features


def extract_graph_structure_features(snapshots):
    """提取图结构特征"""
    features = []
    
    # 聚合系数（平均聚类系数）
    clustering_coeffs = []
    for snapshot in snapshots:
        if len(snapshot.seen_nodes) > 0:
            # 计算聚类系数
            clustering = calculate_clustering_coefficient(snapshot)
            clustering_coeffs.append(clustering)
    
    if clustering_coeffs:
        features.extend([
            np.mean(clustering_coeffs),      # 平均聚类系数
            np.std(clustering_coeffs),       # 聚类系数标准差
        ])
    else:
        features.extend([0, 0])
    
    # 直径特征
    diameters = []
    for snapshot in snapshots:
        if len(snapshot.seen_nodes) > 0:
            diameter = calculate_graph_diameter(snapshot)
            diameters.append(diameter)
    
    if diameters:
        features.extend([
            np.mean(diameters),          # 平均直径
            np.max(diameters),           # 最大直径
        ])
    else:
        features.extend([0, 0])
    
    # 连通分量数
    components = []
    for snapshot in snapshots:
        if len(snapshot.seen_nodes) > 0:
            component_count = count_connected_components(snapshot)
            components.append(component_count)
    
    if components:
        features.extend([
            np.mean(components),          # 平均连通分量数
            np.max(components),           # 最大连通分量数
        ])
    else:
        features.extend([0, 0])
    
    return features


def extract_propagation_features(snapshots):
    """提取传播模式特征"""
    features = []
    
    if not snapshots:
        return features
    
    # 传播速度
    if len(snapshots) > 1:
        first_nodes = len(snapshots[0].seen_nodes)
        last_nodes = len(snapshots[-1].seen_nodes)
        propagation_speed = (last_nodes - first_nodes) / len(snapshots)
        features.append(propagation_speed)
    else:
        features.append(0)
    
    # 传播加速度
    if len(snapshots) > 2:
        speeds = []
        for i in range(1, len(snapshots)):
            if i > 0:
                prev_nodes = len(snapshots[i-1].seen_nodes)
                curr_nodes = len(snapshots[i].seen_nodes)
                if prev_nodes > 0:
                    speeds.append((curr_nodes - prev_nodes))
        
        if len(speeds) > 1:
            accelerations = []
            for i in range(1, len(speeds)):
                accelerations.append(speeds[i] - speeds[i-1])
            features.extend([
                np.mean(accelerations),      # 平均加速度
                np.std(accelerations),       # 加速度标准差
            ])
        else:
            features.extend([0, 0])
    else:
        features.extend([0, 0])
    
    # 传播模式类型
    propagation_pattern = classify_propagation_pattern(snapshots)
    features.extend([
        float(propagation_pattern),     # 传播模式类型
    ])
    
    return features


def extract_statistical_features(snapshots):
    """提取统计特征"""
    features = []
    
    # 收集所有统计量
    all_node_counts = [len(snapshot.seen_nodes) for snapshot in snapshots]
    all_edge_counts = [len(snapshot.edges) for snapshot in snapshots]
    all_active_counts = [len(snapshot.new_events) for snapshot in snapshots]
    
    # 分布特征
    features.extend([
        np.percentile(all_node_counts, 25),   # 节点数25分位数
        np.percentile(all_node_counts, 50),   # 节点数50分位数
        np.percentile(all_node_counts, 75),   # 节点数75分位数
        np.percentile(all_edge_counts, 25),   # 边数25分位数
        np.percentile(all_edge_counts, 50),   # 边数50分位数
        np.percentile(all_edge_counts, 75),   # 边数75分位数
    ])
    
    # 偏度和峰度
    features.extend([
        calculate_skewness(all_node_counts),    # 节点数偏度
        calculate_kurtosis(all_node_counts),    # 节点数峰度
        calculate_skewness(all_edge_counts),    # 边数偏度
        calculate_kurtosis(all_edge_counts),    # 边数峰度
    ])
    
    # 熵特征
    node_entropy = calculate_entropy(all_node_counts)
    edge_entropy = calculate_entropy(all_edge_counts)
    features.extend([
        node_entropy,                      # 节点数熵
        edge_entropy,                       # 边数熵
    ])
    
    return features


def calculate_clustering_coefficient(snapshot):
    """计算聚类系数"""
    if len(snapshot.seen_nodes) == 0:
        return 0.0
    
    clustering_coeffs = []
    for node in snapshot.seen_nodes:
        neighbors = snapshot.children_by_node.get(node, [])
        if not neighbors:
            clustering_coeffs.append(0.0)
            continue
        
        # 计算邻居之间的连接数
        neighbor_connections = 0
        total_possible = len(neighbors) * (len(neighbors) - 1) / 2
        
        for i, neighbor1 in enumerate(neighbors):
            for neighbor2 in neighbors[i+1:]:
                if neighbor2 in snapshot.children_by_node.get(neighbor1, []):
                    neighbor_connections += 1
        
        if total_possible > 0:
            clustering_coeffs.append(neighbor_connections / total_possible)
        else:
            clustering_coeffs.append(0.0)
    
    return np.mean(clustering_coeffs) if clustering_coeffs else 0.0


def calculate_graph_diameter(snapshot):
    """计算图的直径"""
    if len(snapshot.seen_nodes) == 0:
        return 0
    
    # 简化计算：使用最大深度作为直径近似
    max_depth = max(snapshot.depth_by_node.values(), default=0)
    return max_depth


def count_connected_components(snapshot):
    """计算连通分量数"""
    if len(snapshot.seen_nodes) == 0:
        return 0
    
    # 使用BFS计算连通分量
    visited = set()
    components = 0
    
    for node in snapshot.seen_nodes:
        if node not in visited:
            components += 1
            # BFS遍历
            queue = [node]
            visited.add(node)
            while queue:
                current = queue.pop(0)
                neighbors = snapshot.children_by_node.get(current, [])
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
    
    return components


def classify_propagation_pattern(snapshots):
    """分类传播模式"""
    if len(snapshots) < 3:
        return 0  # 未知模式
    
    # 计算增长趋势
    node_counts = [len(snapshot.seen_nodes) for snapshot in snapshots]
    
    # 线性增长
    if np.std(node_counts) < np.mean(node_counts) * 0.1:
        return 1  # 线性增长
    
    # 指数增长
    growth_rates = []
    for i in range(1, len(node_counts)):
        if node_counts[i-1] > 0:
            growth_rates.append((node_counts[i] - node_counts[i-1]) / node_counts[i-1])
    
    if growth_rates and np.mean(growth_rates) > 0.5:
        return 2  # 指数增长
    
    # 对数增长
    if growth_rates and 0.1 < np.mean(growth_rates) < 0.5:
        return 3  # 对数增长
    
    # 饱和增长
    if len(node_counts) > 5:
        recent_growth = np.mean(node_counts[-3:]) - np.mean(node_counts[-6:-3])
        if recent_growth < np.mean(node_counts) * 0.1:
            return 4  # 饱和增长
    
    return 0  # 未知模式


def calculate_skewness(data):
    """计算偏度"""
    if len(data) < 3:
        return 0.0
    
    mean_val = np.mean(data)
    std_val = np.std(data)
    
    if std_val == 0:
        return 0.0
    
    skewness = np.mean(((data - mean_val) / std_val) ** 3)
    return skewness


def calculate_kurtosis(data):
    """计算峰度"""
    if len(data) < 4:
        return 0.0
    
    mean_val = np.mean(data)
    std_val = np.std(data)
    
    if std_val == 0:
        return 0.0
    
    kurtosis = np.mean(((data - mean_val) / std_val) ** 4) - 3
    return kurtosis


def calculate_entropy(data):
    """计算熵"""
    if len(data) == 0:
        return 0.0
    
    # 计算概率分布
    hist, bin_edges = np.histogram(data, bins=10, density=True)
    hist = hist + 1e-10  # 避免log(0)
    prob = hist / np.sum(hist)
    
    # 计算熵
    entropy = -np.sum(prob * np.log(prob))
    return entropy


def create_advanced_dataset(cascades, config):
    """创建高级特征数据集"""
    advanced_features = []
    labels = []
    
    for cascade in cascades:
        features = extract_advanced_features(cascade, config)
        if features:
            advanced_features.append(features)
            labels.append(cascade.final_size)
    
    return np.array(advanced_features), np.array(labels)


if __name__ == "__main__":
    from cascade_model.dataset_profiles import load_dataset_and_config
    
    # 加载数据
    dataset_name, cascades, config = load_dataset_and_config("pp/sample_data/wikipedia.csv", "wikipedia")
    
    # 创建高级特征
    features, labels = create_advanced_dataset(cascades, config)
    
    print(f"高级特征矩阵形状: {features.shape}")
    print(f"标签数组形状: {labels.shape}")
    
    # 计算特征相关性
    correlations = []
    for i in range(features.shape[1]):
        try:
            corr = np.corrcoef(features[:, i], labels)[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))
        except:
            pass
    
    if correlations:
        avg_corr = np.mean(correlations)
        max_corr = np.max(correlations)
        print(f"\n高级特征-标签平均相关性: {avg_corr:.4f}")
        print(f"高级特征-标签最大相关性: {max_corr:.4f}")
        
        if avg_corr > 0.2:
            print("✅ 高级特征工程成功！")
        elif avg_corr > 0.1:
            print("⚠️ 高级特征有一定效果，但仍需改进")
        else:
            print("❌ 高级特征效果仍然有限")
