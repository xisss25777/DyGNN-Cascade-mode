import os
import numpy as np
from .data import load_cascade_data

def load_dataset_and_config(input_path, dataset_name):
    """
    加载数据集和配置
    参数：
        input_path: str, 输入文件路径
        dataset_name: str, 数据集名称
    返回：
        dataset_name: str, 数据集名称
        cascades: list, 级联数据
        config: dict, 配置信息
    """
    # 数据集根目录
    dataset_root = 'D:\\python\\pycharm\\TG_network_datasets'
    
    # 加载级联数据
    cascades = load_cascade_data(dataset_root, dataset_name)
    
    # 加载节点特征以获取特征维度
    node_feat_path = os.path.join(dataset_root, dataset_name, f'ml_{dataset_name}_node.npy')
    node_features = np.load(node_feat_path)
    feature_count = node_features.shape[1]
    
    # 配置信息
    config = {
        'feature_count': feature_count,
        'hidden_channels': 64,
        'learning_rate': 0.001,
        'epochs': 100,
        'batch_size': 32,
        'T': 10.0  # 观测窗口
    }
    
    return dataset_name, cascades, config
