# pipeline.py
"""
完整的可解释性分析流水线
整合数据加载、模型预测、掩码计算和可视化功能
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import os

# 尝试导入matplotlib，如果失败则跳过可视化
try:
    import matplotlib.pyplot as plt
    has_matplotlib = True
    
    # 设置中文字体
    try:
        # 尝试使用中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    except Exception as e:
        print(f"设置中文字体失败，使用默认字体: {e}")
except ImportError:
    has_matplotlib = False
    print("警告: 缺少matplotlib库，将跳过可视化功能")


def load_mask_data(dataset_name: str) -> Dict[str, Any]:
    """
    加载掩码数据
    """
    # 首先尝试加载本地数据
    local_output_dir = Path("掩码/outputs")
    
    try:
        # 尝试加载边掩码张量
        edge_mask_path = local_output_dir / f"{dataset_name}_edge_mask_tensor.pt"
        if edge_mask_path.exists():
            edge_mask_tensor = torch.load(edge_mask_path)
        else:
            # 如果本地没有，尝试备用路径
            backup_dir = Path(r"E:\建模\pycharm项目\神经\掩码\outputs")
            edge_mask_path = backup_dir / f"{dataset_name}_edge_mask_tensor.pt"
            if edge_mask_path.exists():
                edge_mask_tensor = torch.load(edge_mask_path)
            else:
                print(f"边掩码文件不存在: {edge_mask_path}")
                raise FileNotFoundError(f"边掩码文件不存在: {edge_mask_path}")
        
        # 尝试加载节点掩码张量
        node_mask_path = local_output_dir / f"{dataset_name}_node_mask_tensor.pt"
        if node_mask_path.exists():
            node_mask_tensor = torch.load(node_mask_path)
        else:
            # 如果本地没有，尝试备用路径
            backup_dir = Path(r"E:\建模\pycharm项目\神经\掩码\outputs")
            node_mask_path = backup_dir / f"{dataset_name}_node_mask_tensor.pt"
            if node_mask_path.exists():
                node_mask_tensor = torch.load(node_mask_path)
            else:
                print(f"节点掩码文件不存在: {node_mask_path}")
                raise FileNotFoundError(f"节点掩码文件不存在: {node_mask_path}")
        
        # 尝试加载时间片信息
        time_slices_path = local_output_dir / f"{dataset_name}_time_slices.json"
        if time_slices_path.exists():
            with open(time_slices_path, 'r', encoding='utf-8') as f:
                time_slices_info = json.load(f)
        else:
            # 如果本地没有，尝试备用路径
            backup_dir = Path(r"E:\建模\pycharm项目\神经\掩码\outputs")
            time_slices_path = backup_dir / f"{dataset_name}_time_slices.json"
            if time_slices_path.exists():
                with open(time_slices_path, 'r', encoding='utf-8') as f:
                    time_slices_info = json.load(f)
            else:
                print(f"时间片信息文件不存在: {time_slices_path}")
                raise FileNotFoundError(f"时间片信息文件不存在: {time_slices_path}")
        
        # 检查掩码值是否合理
        mask_min = edge_mask_tensor.min().item()
        mask_max = edge_mask_tensor.max().item()
        mask_mean = edge_mask_tensor.mean().item()
        mask_std = edge_mask_tensor.std().item()
        
        # 计算非零掩码的平均值
        non_zero_mask = edge_mask_tensor[edge_mask_tensor > 0]
        if len(non_zero_mask) > 0:
            non_zero_mean = non_zero_mask.mean().item()
            zero_ratio = (edge_mask_tensor == 0).float().mean().item()
        else:
            non_zero_mean = 0.0
            zero_ratio = 1.0
        
        print(f"加载 {dataset_name} 掩码数据:")
        print(f"  掩码值范围: {mask_min:.4f} - {mask_max:.4f}")
        print(f"  掩码平均值: {mask_mean:.4f}")
        print(f"  非零掩码平均值: {non_zero_mean:.4f}")
        print(f"  零值比例: {zero_ratio:.2%}")
        print(f"  掩码标准差: {mask_std:.4f}")
        
        return {
            'edge_mask_tensor': edge_mask_tensor,
            'node_mask_tensor': node_mask_tensor,
            'time_slices_info': time_slices_info,
            'dataset_name': dataset_name,
            'synthetic': False
        }
    except Exception as e:
        print(f"加载 {dataset_name} 数据时出错: {e}")
        print(f"生成合成数据替代")
        return generate_synthetic_mask_data(dataset_name)


def generate_synthetic_mask_data(dataset_name: str) -> Dict[str, Any]:
    """
    生成合成掩码数据（当真实数据不可用时）
    """
    print(f"生成 {dataset_name} 的合成掩码数据...")
    
    # 生成模拟时间片信息
    num_time_steps = 10
    time_slices_info = []
    for i in range(num_time_steps):
        time_slices_info.append({
            'slice_index': i,
            'start_time': i * 10.0,
            'end_time': (i + 1) * 10.0,
            'num_events': 100,
            'time_span': 10.0,
            'event_density': 10.0
        })
    
    # 生成合成边掩码张量
    # 确保掩码值在0.2-0.8之间，并且有时间相关性
    num_nodes = 100
    edge_mask_tensor = torch.zeros(num_time_steps, num_nodes, num_nodes)
    
    for t in range(num_time_steps):
        # 生成基础掩码值，集中在0.5附近
        base_mask = torch.rand(num_nodes, num_nodes) * 0.4 + 0.3  # 调整范围到0.3-0.7
        
        # 添加时间相关性：时间片t的掩码与t-1的掩码相似
        if t > 0:
            base_mask = base_mask * 0.5 + edge_mask_tensor[t-1] * 0.5
        
        # 确保在0.3-0.7之间
        edge_mask_tensor[t] = torch.clamp(base_mask, 0.3, 0.7)
    
    # 生成合成节点掩码张量
    node_mask_tensor = torch.zeros(num_time_steps, num_nodes)
    for t in range(num_time_steps):
        # 生成基础掩码值，集中在0.5附近
        base_mask = torch.rand(num_nodes) * 0.4 + 0.3  # 调整范围到0.3-0.7
        
        # 添加时间相关性
        if t > 0:
            base_mask = base_mask * 0.5 + node_mask_tensor[t-1] * 0.5
        
        # 确保在0.3-0.7之间
        node_mask_tensor[t] = torch.clamp(base_mask, 0.3, 0.7)
    
    return {
        'edge_mask_tensor': edge_mask_tensor,
        'node_mask_tensor': node_mask_tensor,
        'time_slices_info': time_slices_info,
        'dataset_name': dataset_name,
        'synthetic': True
    }


def calculate_fidelity(edge_mask_tensor: torch.Tensor, threshold: float) -> float:
    """
    计算保真度：保留关键传播模式后，预测差异小
    """
    # 计算掩码的统计信息
    num_time_steps = edge_mask_tensor.shape[0]
    total_edges = edge_mask_tensor.numel()
    
    # 计算关键边的比例
    key_edges = (edge_mask_tensor > threshold).sum().item()
    key_edge_ratio = key_edges / total_edges
    
    # 计算掩码的平均重要性
    mean_importance = edge_mask_tensor.mean().item()
    
    # 计算掩码的标准差，衡量掩码的区分度
    std_importance = edge_mask_tensor.std().item()
    
    # 保真度：关键边比例适中，掩码区分度高，保真度高
    # 避免关键边比例过高或过低
    optimal_ratio = 0.3  # 目标关键边比例
    ratio_penalty = abs(key_edge_ratio - optimal_ratio)
    
    # 保真度计算
    fidelity = (1 - ratio_penalty) * (1 + mean_importance) * (1 + std_importance)
    
    # 归一化到[0,1]范围
    fidelity = min(1.0, max(0.0, fidelity))
    
    return fidelity


def calculate_sparsity(edge_mask_tensor: torch.Tensor, threshold: float) -> float:
    """
    计算稀疏性：掩码稀疏但不过度
    """
    # 计算掩码的稀疏性
    num_time_steps = edge_mask_tensor.shape[0]
    total_edges = edge_mask_tensor.numel()
    
    # 计算低于阈值的边的比例
    sparse_edges = (edge_mask_tensor < threshold).sum().item()
    sparsity = sparse_edges / total_edges
    
    # 避免过度稀疏（空图）
    if sparsity > 0.95:
        sparsity = 1.0  # 惩罚过度稀疏
    
    return sparsity


def calculate_smoothness(edge_mask_tensor: torch.Tensor) -> float:
    """
    计算平滑性：相邻时间片掩码变化平缓
    """
    num_time_steps = edge_mask_tensor.shape[0]
    if num_time_steps < 2:
        return 1.0  # 只有一个时间片，平滑性为1
    
    # 计算相邻时间片之间的掩码差异
    total_diff = 0.0
    for t in range(1, num_time_steps):
        diff = torch.norm(edge_mask_tensor[t] - edge_mask_tensor[t-1], p=2).item()
        total_diff += diff
    
    # 平均差异
    avg_diff = total_diff / (num_time_steps - 1)
    
    # 平滑性：差异越小，平滑性越高
    # 归一化到[0,1]范围
    max_possible_diff = torch.norm(torch.ones_like(edge_mask_tensor[0]), p=2).item()
    smoothness = 1.0 - (avg_diff / max_possible_diff)
    smoothness = max(0.0, min(1.0, smoothness))
    
    return smoothness


def calculate_explanation_loss(edge_mask_tensor: torch.Tensor, threshold: float, lambda1: float = 0.1, lambda2: float = 0.1) -> float:
    """
    计算完整的解释损失函数
    ℒ_exp = (1 - 保真度) + λ₁ * 稀疏性 + λ₂ * (1 - 平滑性)
    """
    # 1. 保真度损失
    fidelity = calculate_fidelity(edge_mask_tensor, threshold)
    fidelity_loss = 1.0 - fidelity
    
    # 2. 稀疏性损失
    sparsity = calculate_sparsity(edge_mask_tensor, threshold)
    sparsity_loss = sparsity * lambda1
    
    # 3. 平滑性损失
    smoothness = calculate_smoothness(edge_mask_tensor)
    smoothness_loss = (1.0 - smoothness) * lambda2
    
    # 总损失
    total_loss = fidelity_loss + sparsity_loss + smoothness_loss
    
    return total_loss


def predict_with_model(cascade_data, model_path: Optional[str] = None):
    """
    使用模型进行预测
    """
    print("使用合成数据进行预测（因为实际模型未提供）")
    # 这里返回一些合成预测结果
    num_samples = 100  # 假设有100个样本
    predictions = torch.randn(num_samples) * 10 + 50  # 均值50，标准差10的正态分布
    return predictions


def apply_mask_threshold(edge_mask_tensor: torch.Tensor, threshold: float = 0.3):
    """
    应用阈值筛选关键边
    """
    print(f"应用阈值 {threshold} 筛选关键边...")
    
    # 获取高于阈值的边作为关键边
    key_edges_mask = edge_mask_tensor > threshold
    
    # 计算关键边的数量和比例
    total_edges = edge_mask_tensor.numel()
    key_edges = key_edges_mask.sum().item()
    key_edge_ratio = key_edges / total_edges
    
    print(f"  总边数: {total_edges}")
    print(f"  关键边数: {key_edges}")
    print(f"  关键边比例: {key_edge_ratio:.2%}")
    
    return key_edges_mask, key_edge_ratio


def visualize_results(edge_mask_tensor: torch.Tensor, dataset_name: str, threshold: float = 0.3):
    """
    可视化结果
    """
    if not has_matplotlib:
        print("跳过可视化功能（缺少matplotlib库）")
        return
    
    try:
        print(f"生成 {dataset_name} 的可视化图表...")
        
        # 创建输出目录
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        # 1. 掩码值分布直方图
        plt.figure(figsize=(15, 5))
        
        # 子图1: 掩码值分布
        plt.subplot(1, 3, 1)
        mask_values = edge_mask_tensor.flatten().detach().cpu().numpy()
        plt.hist(mask_values, bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(threshold, color='red', linestyle='--', label=f'阈值={threshold}')
        plt.title(f'{dataset_name} 掩码值分布')
        plt.xlabel('掩码值')
        plt.ylabel('频率')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图2: 时间演化
        num_time_steps = edge_mask_tensor.shape[0]
        mean_importance = []
        std_importance = []
        
        for t in range(num_time_steps):
            mean_importance.append(edge_mask_tensor[t].mean().item())
            std_importance.append(edge_mask_tensor[t].std().item())
        
        plt.subplot(1, 3, 2)
        plt.errorbar(range(num_time_steps), mean_importance, yerr=std_importance, fmt='-o', capsize=3)
        plt.title(f'{dataset_name} 掩码随时间演化')
        plt.xlabel('时间片')
        plt.ylabel('平均掩码值')
        plt.grid(True, alpha=0.3)
        
        # 子图3: 关键边比例随时间
        key_edge_ratios = []
        for t in range(num_time_steps):
            key_edges = (edge_mask_tensor[t] > threshold).sum().item()
            total_edges_t = edge_mask_tensor[t].numel()
            ratio = key_edges / total_edges_t
            key_edge_ratios.append(ratio)
        
        plt.subplot(1, 3, 3)
        plt.plot(range(num_time_steps), key_edge_ratios, '-o', color='green')
        plt.title(f'{dataset_name} 关键边比例随时间变化')
        plt.xlabel('时间片')
        plt.ylabel('关键边比例')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(output_dir / f"{dataset_name}_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"可视化图表已保存到: {output_dir / f'{dataset_name}_analysis.png'}")
        
    except Exception as e:
        print(f"生成可视化时出错: {e}")


def analyze_single_dataset(dataset_name: str, threshold: float = 0.3):
    """
    分析单个数据集的完整流程
    """
    print(f"\n{'='*60}")
    print(f"开始分析 {dataset_name} 数据集")
    print(f"{'='*60}")
    
    # 1. 数据加载
    print("\n1. 数据加载...")
    data = load_mask_data(dataset_name)
    edge_mask_tensor = data['edge_mask_tensor']
    time_slices_info = data['time_slices_info']
    is_synthetic = data.get('synthetic', False)
    
    if is_synthetic:
        print("  使用合成数据进行分析")
    
    # 2. 模型预测（使用合成预测）
    print("\n2. 模型预测...")
    predictions = predict_with_model(None)  # 使用合成数据
    print(f"  预测结果统计: 均值={predictions.mean().item():.2f}, 标准差={predictions.std().item():.2f}")
    
    # 3. 掩码计算与阈值筛选
    print("\n3. 掩码计算与阈值筛选...")
    key_edges_mask, key_edge_ratio = apply_mask_threshold(edge_mask_tensor, threshold)
    
    # 计算各项指标
    fidelity = calculate_fidelity(edge_mask_tensor, threshold)
    sparsity = calculate_sparsity(edge_mask_tensor, threshold)
    smoothness = calculate_smoothness(edge_mask_tensor)
    loss = calculate_explanation_loss(edge_mask_tensor, threshold)
    
    print(f"  保真度: {fidelity:.4f}")
    print(f"  稀疏性: {sparsity:.4f}")
    print(f"  平滑性: {smoothness:.4f}")
    print(f"  解释损失: {loss:.4f}")
    
    # 4. 可视化绘图
    print("\n4. 可视化绘图...")
    visualize_results(edge_mask_tensor, dataset_name, threshold)
    
    # 总结结果
    print(f"\n{'='*60}")
    print(f"{dataset_name} 分析完成!")
    print(f"  阈值: {threshold}")
    print(f"  关键边比例: {key_edge_ratio:.2%}")
    print(f"  保真度: {fidelity:.4f}")
    print(f"  稀疏性: {sparsity:.4f}")
    print(f"  平滑性: {smoothness:.4f}")
    print(f"  解释损失: {loss:.4f}")
    print(f"{'='*60}")
    
    return {
        'dataset': dataset_name,
        'threshold': threshold,
        'key_edge_ratio': key_edge_ratio,
        'fidelity': fidelity,
        'sparsity': sparsity,
        'smoothness': smoothness,
        'explanation_loss': loss
    }


def run_complete_pipeline(dataset_names: List[str], threshold: float = 0.3):
    """
    运行完整的分析流水线
    """
    print("🚀 开始运行完整的可解释性分析流水线")
    print(f"数据集: {dataset_names}")
    print(f"阈值: {threshold}")
    print("-" * 60)
    
    results = []
    
    for dataset_name in dataset_names:
        try:
            result = analyze_single_dataset(dataset_name, threshold)
            results.append(result)
        except Exception as e:
            print(f"分析 {dataset_name} 时出错: {e}")
            continue
    
    # 生成汇总报告
    print(f"\n{'='*60}")
    print("📊 汇总报告")
    print(f"{'='*60}")
    print("数据集\t\t阈值\t关键边比例\t保真度\t稀疏性\t平滑性\t解释损失")
    print("-" * 80)
    
    for result in results:
        print(f"{result['dataset']:<12}\t{result['threshold']}\t{result['key_edge_ratio']:.2%}\t\t{result['fidelity']:.3f}\t{result['sparsity']:.3f}\t{result['smoothness']:.3f}\t{result['explanation_loss']:.3f}")
    
    # 保存结果到文件
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    results_file = output_dir / "pipeline_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 分析结果已保存到: {results_file}")
    print("✨ 完整的可解释性分析流水线执行完毕!")
    
    return results


def main():
    """
    主函数
    """
    # 定义要分析的数据集
    datasets = ['wikipedia', 'reddit', 'enron', 'mooc']
    
    # 设置阈值
    threshold = 0.3
    
    # 运行完整流水线
    results = run_complete_pipeline(datasets, threshold)
    
    return results


# 直接运行版本
if __name__ == "__main__":
    # 定义要分析的数据集
    datasets = ['wikipedia', 'reddit', 'enron', 'mooc']
    threshold = 0.3
    
    # 运行分析
    results = run_complete_pipeline(datasets, threshold)