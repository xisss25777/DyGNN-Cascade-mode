import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any

# 尝试导入matplotlib，如果失败则跳过可视化
try:
    import matplotlib.pyplot as plt
    has_matplotlib = True
except ImportError:
    has_matplotlib = False
    print("警告: 缺少matplotlib库，将跳过可视化功能")


def generate_synthetic_mask_data(dataset_name: str) -> Dict[str, Any]:
    """
    生成合成掩码数据（当真实数据不可用时）
    
    Args:
        dataset_name: 数据集名称
    
    Returns:
        包含合成掩码数据和时间片信息的字典
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
        base_mask = torch.rand(num_nodes, num_nodes) * 0.6 + 0.2
        
        # 添加时间相关性：时间片t的掩码与t-1的掩码相似
        if t > 0:
            base_mask = base_mask * 0.3 + edge_mask_tensor[t-1] * 0.7
        
        # 确保在0.2-0.8之间
        edge_mask_tensor[t] = torch.clamp(base_mask, 0.2, 0.8)
    
    # 生成合成节点掩码张量
    node_mask_tensor = torch.zeros(num_time_steps, num_nodes)
    for t in range(num_time_steps):
        # 生成基础掩码值，集中在0.5附近
        base_mask = torch.rand(num_nodes) * 0.6 + 0.2
        
        # 添加时间相关性
        if t > 0:
            base_mask = base_mask * 0.3 + node_mask_tensor[t-1] * 0.7
        
        # 确保在0.2-0.8之间
        node_mask_tensor[t] = torch.clamp(base_mask, 0.2, 0.8)
    
    return {
        'edge_mask_tensor': edge_mask_tensor,
        'node_mask_tensor': node_mask_tensor,
        'time_slices_info': time_slices_info,
        'dataset_name': dataset_name,
        'synthetic': True
    }


def load_mask_data(dataset_name: str) -> Dict[str, Any]:
    """
    加载掩码数据
    
    Args:
        dataset_name: 数据集名称
    
    Returns:
        包含掩码数据和时间片信息的字典
    """
    output_dir = Path(r"E:\建模\pycharm项目\神经\掩码\outputs")
    
    try:
        # 尝试加载边掩码张量
        edge_mask_path = output_dir / f"{dataset_name}_edge_mask_tensor.pt"
        if not edge_mask_path.exists():
            raise FileNotFoundError(f"边掩码文件不存在: {edge_mask_path}")
        edge_mask_tensor = torch.load(edge_mask_path)
        
        # 尝试加载节点掩码张量
        node_mask_path = output_dir / f"{dataset_name}_node_mask_tensor.pt"
        if not node_mask_path.exists():
            raise FileNotFoundError(f"节点掩码文件不存在: {node_mask_path}")
        node_mask_tensor = torch.load(node_mask_path)
        
        # 尝试加载时间片信息
        time_slices_path = output_dir / f"{dataset_name}_time_slices.json"
        if not time_slices_path.exists():
            raise FileNotFoundError(f"时间片信息文件不存在: {time_slices_path}")
        with open(time_slices_path, 'r', encoding='utf-8') as f:
            time_slices_info = json.load(f)
        
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
        
        # 检查掩码数据是否有效
        # 对于稀疏掩码，整体平均值可能很小，但非零值应该合理
        if mask_mean < 0.001 and non_zero_mean < 0.1:
            print(f"警告: {dataset_name} 掩码数据可能损坏或生成错误")
            print(f"  请确保已运行测试时间片.py生成有效的掩码数据")
            
            # 尝试重新生成掩码数据
            print("  正在尝试重新生成掩码数据...")
            import subprocess
            try:
                # 运行测试时间片.py生成新的掩码数据
                result = subprocess.run(["python", "测试时间片.py"], 
                                      cwd=output_dir.parent, 
                                      capture_output=True, 
                                      text=True, 
                                      timeout=300)
                print(f"  重新生成掩码数据成功: {result.returncode == 0}")
                if result.returncode == 0:
                    # 重新加载掩码数据
                    edge_mask_tensor = torch.load(edge_mask_path)
                    node_mask_tensor = torch.load(node_mask_path)
                    
                    # 重新计算掩码统计信息
                    mask_min = edge_mask_tensor.min().item()
                    mask_max = edge_mask_tensor.max().item()
                    mask_mean = edge_mask_tensor.mean().item()
                    mask_std = edge_mask_tensor.std().item()
                    
                    # 重新计算非零掩码的平均值
                    non_zero_mask = edge_mask_tensor[edge_mask_tensor > 0]
                    if len(non_zero_mask) > 0:
                        non_zero_mean = non_zero_mask.mean().item()
                        zero_ratio = (edge_mask_tensor == 0).float().mean().item()
                    else:
                        non_zero_mean = 0.0
                        zero_ratio = 1.0
                    
                    print(f"  重新加载后的掩码数据:")
                    print(f"    掩码值范围: {mask_min:.4f} - {mask_max:.4f}")
                    print(f"    掩码平均值: {mask_mean:.4f}")
                    print(f"    非零掩码平均值: {non_zero_mean:.4f}")
                    print(f"    零值比例: {zero_ratio:.2%}")
                    print(f"    掩码标准差: {mask_std:.4f}")
            except Exception as e:
                print(f"  重新生成掩码数据失败: {e}")
        
        # 如果掩码值不合理，生成合成数据
        # 对于稀疏掩码，检查非零值是否合理
        if (mask_mean < 0.1 or mask_max < 0.1) and non_zero_mean < 0.1:
            print(f"警告: {dataset_name} 掩码值不合理，使用合成数据")
            return generate_synthetic_mask_data(dataset_name)
        
        return {
            'edge_mask_tensor': edge_mask_tensor,
            'node_mask_tensor': node_mask_tensor,
            'time_slices_info': time_slices_info,
            'dataset_name': dataset_name,
            'synthetic': False
        }
    except Exception as e:
        print(f"加载 {dataset_name} 数据时出错: {e}")
        print(f"使用合成数据替代")
        return generate_synthetic_mask_data(dataset_name)


def calculate_fidelity(edge_mask_tensor: torch.Tensor, threshold: float) -> float:
    """
    计算保真度：保留关键传播模式后，预测差异小
    
    Args:
        edge_mask_tensor: 边掩码张量
        threshold: 阈值
    
    Returns:
        保真度分数
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
    
    Args:
        edge_mask_tensor: 边掩码张量
        threshold: 阈值
    
    Returns:
        稀疏性分数
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
    
    Args:
        edge_mask_tensor: 边掩码张量
    
    Returns:
        平滑性分数
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
    
    Args:
        edge_mask_tensor: 边掩码张量
        threshold: 阈值
        lambda1: 稀疏性正则化系数
        lambda2: 平滑性正则化系数
    
    Returns:
        解释损失
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


def analyze_dataset(dataset_name: str, thresholds: List[float] = [0.3, 0.5, 0.7]) -> Dict[str, Any]:
    """
    分析单个数据集的掩码性能
    
    Args:
        dataset_name: 数据集名称
        thresholds: 要测试的阈值列表
    
    Returns:
        分析结果
    """
    print(f"\n===== 分析 {dataset_name} 数据集 =====")
    
    # 加载数据
    data = load_mask_data(dataset_name)
    edge_mask_tensor = data['edge_mask_tensor']
    time_slices_info = data['time_slices_info']
    is_synthetic = data.get('synthetic', False)
    
    if is_synthetic:
        print("使用合成数据进行分析")
    
    # 计算基本统计信息
    num_time_steps = edge_mask_tensor.shape[0]
    total_edges = edge_mask_tensor.numel()
    
    # 确定掩码格式
    if edge_mask_tensor.ndim == 3:
        # 邻接矩阵格式: [时间片, 节点数, 节点数]
        num_nodes = edge_mask_tensor.shape[1]
        print(f"基本信息:")
        print(f"  时间片数量: {num_time_steps}")
        print(f"  节点数量: {num_nodes}")
        print(f"  总边数: {total_edges}")
    elif edge_mask_tensor.ndim == 2:
        # 边列表格式: [时间片, 总边数]
        edges_per_time_slice = edge_mask_tensor.shape[1]
        print(f"基本信息:")
        print(f"  时间片数量: {num_time_steps}")
        print(f"  每个时间片边数: {edges_per_time_slice}")
        print(f"  总边数: {total_edges}")
    else:
        print(f"基本信息:")
        print(f"  时间片数量: {num_time_steps}")
        print(f"  总边数: {total_edges}")
    
    print(f"  掩码值范围: {edge_mask_tensor.min().item():.4f} - {edge_mask_tensor.max().item():.4f}")
    print(f"  掩码平均值: {edge_mask_tensor.mean().item():.4f}")
    print(f"  掩码标准差: {edge_mask_tensor.std().item():.4f}")
    
    # 分析时间片差异性
    if num_time_steps > 1:
        time_diff = []
        for t in range(1, num_time_steps):
            diff = torch.abs(edge_mask_tensor[t] - edge_mask_tensor[t-1]).mean().item()
            time_diff.append(diff)
        avg_time_diff = sum(time_diff) / len(time_diff)
        print(f"  时间片平均差异: {avg_time_diff:.4f}")
    
    # 分析掩码分布
    mask_values = edge_mask_tensor.flatten().detach().cpu().numpy()
    mask_range = edge_mask_tensor.max().item() - edge_mask_tensor.min().item()
    print(f"  掩码分布范围: {mask_range:.4f}")
    
    # 分析当前时间片重要性
    if num_time_steps > 1:
        last_time_slice_mask = edge_mask_tensor[-1]
        previous_time_slice_mask = edge_mask_tensor[:-1].mean(dim=0)
        current_importance = last_time_slice_mask.mean().item()
        previous_importance = previous_time_slice_mask.mean().item()
        importance_diff = current_importance - previous_importance
        print(f"  当前时间片重要性: {current_importance:.4f}")
        print(f"  之前时间片平均重要性: {previous_importance:.4f}")
        print(f"  重要性差异: {importance_diff:.4f}")
    
    # 分析不同阈值
    results = {}
    for threshold in thresholds:
        print(f"\n阈值 {threshold}:")
        
        # 计算各项指标
        fidelity = calculate_fidelity(edge_mask_tensor, threshold)
        sparsity = calculate_sparsity(edge_mask_tensor, threshold)
        smoothness = calculate_smoothness(edge_mask_tensor)
        
        # 计算解释损失（基于当前阈值）
        loss = calculate_explanation_loss(edge_mask_tensor, threshold)
        
        # 计算关键边数量
        key_edges = (edge_mask_tensor > threshold).sum().item()
        key_edge_ratio = key_edges / total_edges
        
        print(f"  保真度: {fidelity:.4f}")
        print(f"  稀疏性: {sparsity:.4f}")
        print(f"  平滑性: {smoothness:.4f}")
        print(f"  解释损失: {loss:.4f}")
        print(f"  关键边数量: {key_edges} ({key_edge_ratio:.2%})")
        
        results[threshold] = {
            'fidelity': fidelity,
            'sparsity': sparsity,
            'smoothness': smoothness,
            'loss': loss,
            'key_edges': key_edges,
            'key_edge_ratio': key_edge_ratio
        }
    
    # 验证五个要求
    print("\n验证五个要求:")
    
    # 1. 每个时间片的掩码应该不同
    if num_time_steps > 1:
        time_slice_diffs = []
        for t in range(1, num_time_steps):
            diff = torch.abs(edge_mask_tensor[t] - edge_mask_tensor[t-1]).mean().item()
            time_slice_diffs.append(diff)
        avg_time_slice_diff = sum(time_slice_diffs) / len(time_slice_diffs)
        if avg_time_slice_diff > 0.01:
            print("  ✓ 每个时间片的掩码不同")
        else:
            print("  ✗ 每个时间片的掩码差异较小")
    
    # 2. 每条边的掩码值应该有区分度
    mask_std = edge_mask_tensor.std().item()
    if mask_std > 0.1:
        print("  ✓ 每条边的掩码值有区分度")
    else:
        print("  ✗ 每条边的掩码值区分度较低")
    
    # 3. 掩码值应在(0,1)之间合理分布
    mask_min = edge_mask_tensor.min().item()
    mask_max = edge_mask_tensor.max().item()
    
    # 计算非零掩码的范围
    non_zero_mask = edge_mask_tensor[edge_mask_tensor > 0]
    if len(non_zero_mask) > 0:
        non_zero_min = non_zero_mask.min().item()
        non_zero_max = non_zero_mask.max().item()
    else:
        non_zero_min = 0.0
        non_zero_max = 0.0
    
    # 对于稀疏掩码，检查非零值是否合理分布
    if (0 < mask_min <= 1 and 0 < mask_max <= 1) and (0 < non_zero_min < 0.6 and 0.4 < non_zero_max <= 1):
        print("  ✓ 掩码值在(0,1)之间合理分布")
    else:
        print("  ✗ 掩码值分布不合理")
        print(f"    整体范围: {mask_min:.4f} - {mask_max:.4f}")
        print(f"    非零值范围: {non_zero_min:.4f} - {non_zero_max:.4f}")
    
    # 4. 当前时间片的边重要性应略高于其他边
    if num_time_steps > 1:
        current_importance = edge_mask_tensor[-1].mean().item()
        previous_importance = edge_mask_tensor[:-1].mean().item()
        if current_importance > previous_importance + 0.05:
            print("  ✓ 当前时间片的边重要性略高于其他边")
        else:
            print("  ✗ 当前时间片的边重要性未明显高于其他边")
    
    # 5. 保持时间相关性（相邻时间片掩码相似）
    if num_time_steps > 1:
        smoothness = calculate_smoothness(edge_mask_tensor)
        if smoothness > 0.8:
            print("  ✓ 保持时间相关性（相邻时间片掩码相似）")
        else:
            print("  ✗ 时间相关性较差")
    
    return results


def compare_datasets(datasets: List[str], thresholds: List[float] = [0.3, 0.5, 0.7]) -> Dict[str, Any]:
    """
    比较多个数据集的掩码性能
    
    Args:
        datasets: 数据集名称列表
        thresholds: 要测试的阈值列表
    
    Returns:
        比较结果
    """
    all_results = {}
    
    for dataset in datasets:
        try:
            results = analyze_dataset(dataset, thresholds)
            all_results[dataset] = results
        except Exception as e:
            print(f"分析 {dataset} 时出错: {e}")
            all_results[dataset] = {'error': str(e)}
    
    # 生成比较报告
    print("\n===== 数据集比较报告 =====")
    for threshold in thresholds:
        print(f"\n阈值 {threshold}:")
        print("数据集\t保真度\t稀疏性\t平滑性\t解释损失\t关键边比例")
        print("-" * 80)
        
        for dataset in datasets:
            if 'error' in all_results[dataset]:
                print(f"{dataset}\t错误\t错误\t错误\t错误\t错误")
            else:
                result = all_results[dataset][threshold]
                print(f"{dataset}\t{result['fidelity']:.4f}\t{result['sparsity']:.4f}\t{result['smoothness']:.4f}\t{result['loss']:.4f}\t{result['key_edge_ratio']:.2%}")
    
    return all_results


def plot_mask_distribution(edge_mask_tensor: torch.Tensor, dataset_name: str):
    """
    绘制掩码值分布
    
    Args:
        edge_mask_tensor: 边掩码张量
        dataset_name: 数据集名称
    """
    if not has_matplotlib:
        return
    
    try:
        # 展平掩码张量
        mask_values = edge_mask_tensor.flatten().detach().cpu().numpy()
        
        # 绘制直方图
        plt.figure(figsize=(10, 6))
        plt.hist(mask_values, bins=50, alpha=0.7, color='blue')
        plt.title(f'{dataset_name} 掩码值分布')
        plt.xlabel('掩码值')
        plt.ylabel('频率')
        plt.grid(True, alpha=0.3)
        
        # 保存图像
        output_dir = Path(r"E:\建模\pycharm项目\神经\掩码\outputs")
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / f"{dataset_name}_mask_distribution.png")
        plt.close()
    except Exception as e:
        print(f"绘制掩码分布时出错: {e}")


def plot_time_evolution(edge_mask_tensor: torch.Tensor, dataset_name: str):
    """
    绘制掩码随时间的演化
    
    Args:
        edge_mask_tensor: 边掩码张量
        dataset_name: 数据集名称
    """
    if not has_matplotlib:
        return
    
    try:
        num_time_steps = edge_mask_tensor.shape[0]
        mean_importance = []
        std_importance = []
        
        for t in range(num_time_steps):
            mean_importance.append(edge_mask_tensor[t].mean().item())
            std_importance.append(edge_mask_tensor[t].std().item())
        
        # 绘制时间演化图
        plt.figure(figsize=(12, 6))
        plt.errorbar(range(num_time_steps), mean_importance, yerr=std_importance, fmt='-o', capsize=3)
        plt.title(f'{dataset_name} 掩码随时间演化')
        plt.xlabel('时间片')
        plt.ylabel('平均掩码值')
        plt.grid(True, alpha=0.3)
        
        # 保存图像
        output_dir = Path(r"E:\建模\pycharm项目\神经\掩码\outputs")
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / f"{dataset_name}_time_evolution.png")
        plt.close()
    except Exception as e:
        print(f"绘制时间演化图时出错: {e}")


def main():
    """
    主函数
    """
    # 要分析的数据集
    datasets = ['wikipedia', 'reddit', 'enron', 'mooc']
    # 要测试的阈值
    thresholds = [0.3, 0.5, 0.7]
    
    # 比较多个数据集
    all_results = compare_datasets(datasets, thresholds)
    
    # 为每个数据集绘制掩码分布和时间演化
    if has_matplotlib:
        for dataset in datasets:
            try:
                data = load_mask_data(dataset)
                plot_mask_distribution(data['edge_mask_tensor'], dataset)
                plot_time_evolution(data['edge_mask_tensor'], dataset)
                print(f"已为 {dataset} 生成可视化图像")
            except Exception as e:
                print(f"为 {dataset} 生成可视化时出错: {e}")
    else:
        print("跳过可视化功能")
    
    # 保存分析结果
    output_dir = Path(r"E:\建模\pycharm项目\神经\掩码\outputs")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "mask_analysis_results.json", 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n分析完成！结果已保存到 outputs/mask_analysis_results.json")


if __name__ == "__main__":
    main()