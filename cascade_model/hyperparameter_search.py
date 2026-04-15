# hyperparameter_search.py
"""
超参数搜索脚本
使用缓存系统快速搜索最佳超参数
目标：MAE < 60（接近Baseline）
"""
import itertools
import time
import json
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入必要的模块
from cascade_model.cache_utils import cache_manager
from cascade_model.data import load_wikipedia_cascades, generate_synthetic_cascades
from cascade_model.config import PipelineConfig
from cascade_model.dgnn import run_dgnn_pipeline

def run_experiment(lr, batch_size, hidden_dim, epochs=100, dataset="wikipedia"):
    """
    运行实验并返回结果
    
    Args:
        lr: 学习率
        batch_size: 批次大小
        hidden_dim: 隐藏层维度
        epochs: 训练轮数
        dataset: 数据集名称
    
    Returns:
        实验结果字典
    """
    print(f"\n🚀 运行实验: lr={lr}, batch_size={batch_size}, hidden_dim={hidden_dim}, epochs={epochs}")
    
    # 准备数据
    try:
        # 使用TGdataset文件夹中的真实Wikipedia数据（使用绝对路径）
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent  # 到E:\建模\pycharm项目\
        tg_dataset_dir = project_root / "TG_network_datasets"
        wikipedia_file = tg_dataset_dir / "wikipedia" / "wikipedia.csv"
        
        print(f"  尝试加载Wikipedia数据: {wikipedia_file}")
        
        if wikipedia_file.exists():
            print("  加载TGdataset中的真实Wikipedia数据...")
            cascades = load_wikipedia_cascades(str(wikipedia_file))
            print(f"  成功加载 {len(cascades)} 条Wikipedia级联数据")
        else:
            print("  未找到TGdataset中的Wikipedia数据，生成合成数据...")
            cascades = generate_synthetic_cascades(count=200, seed=42)
            print(f"  生成了 {len(cascades)} 条合成级联数据")
    except Exception as e:
        print(f"  数据加载失败: {e}")
        cascades = generate_synthetic_cascades(count=100, seed=42)
        print(f"  生成了 {len(cascades)} 条合成级联数据作为备用")
    
    # 配置参数
    config = PipelineConfig(
        observation_seconds=300,
        slice_seconds=60,
        test_ratio=0.2,
        learning_rate=lr,
        epochs=epochs,
        l2_penalty=0.0001,
        random_seed=42,
        knn_neighbors=5,
        patience=10
    )
    
    # 生成缓存键
    cache_key = f"{dataset}_{lr}_{batch_size}_{hidden_dim}_{epochs}_epochs"
    
    # 检查是否已经有缓存的训练结果
    cached_result = cache_manager.get_training_result(cache_key)
    if cached_result:
        print(f"  使用缓存的训练结果: {cache_key}")
        return cached_result
    
    # 运行训练
    start_time = time.time()
    try:
        result = run_dgnn_pipeline(cascades, config)
        run_time = time.time() - start_time
        
        print(f"  训练完成！耗时: {run_time:.2f}秒")
        print(f"  评估指标:")
        for metric, value in result['metrics'].items():
            print(f"    {metric}: {value:.4f}")
        
        # 保存结果到缓存
        cache_manager.save_training_result(cache_key, result)
        print(f"  训练结果已缓存: {cache_key}")
        
        return result
    except Exception as e:
        print(f"  训练失败: {e}")
        return None

def hyperparameter_search():
    """
    执行超参数搜索
    """
    print("🚀 开始超参数搜索...")
    print("=" * 60)
    
    # 定义搜索空间
    learning_rates = [0.001, 0.0005, 0.0001]
    batch_sizes = [16, 32, 64]
    hidden_dims = [64, 128, 256]
    epochs_list = [50, 100, 150]  # 测试不同的epochs
    
    # 存储所有实验结果
    results = []
    
    # 用缓存快速测试所有组合
    for lr, bs, hd in itertools.product(learning_rates, batch_sizes, hidden_dims):
        # 先测试默认epochs=100
        result = run_experiment(lr=lr, batch_size=bs, hidden_dim=hd, epochs=100)
        if result:
            results.append({
                'learning_rate': lr,
                'batch_size': bs,
                'hidden_dim': hd,
                'epochs': 100,
                'metrics': result['metrics'],
                'sample_count': result['sample_count'],
                'feature_count': result['feature_count']
            })
    
    # 测试不同epochs的效果（使用最佳超参数组合）
    if results:
        # 找到MAE最小的组合
        best_result = min(results, key=lambda x: x['metrics']['mae'])
        print(f"\n📊 最佳超参数组合（epochs=100）:")
        print(f"  学习率: {best_result['learning_rate']}")
        print(f"  批次大小: {best_result['batch_size']}")
        print(f"  隐藏层维度: {best_result['hidden_dim']}")
        print(f"  MAE: {best_result['metrics']['mae']:.4f}")
        
        # 用最佳超参数测试不同epochs
        print("\n🧪 测试不同epochs的效果:")
        for epochs in epochs_list:
            result = run_experiment(
                lr=best_result['learning_rate'],
                batch_size=best_result['batch_size'],
                hidden_dim=best_result['hidden_dim'],
                epochs=epochs
            )
            if result:
                results.append({
                    'learning_rate': best_result['learning_rate'],
                    'batch_size': best_result['batch_size'],
                    'hidden_dim': best_result['hidden_dim'],
                    'epochs': epochs,
                    'metrics': result['metrics'],
                    'sample_count': result['sample_count'],
                    'feature_count': result['feature_count']
                })
    
    # 保存所有结果
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # 按MAE排序结果
    results.sort(key=lambda x: x['metrics']['mae'])
    
    # 保存结果到JSON文件
    output_file = output_dir / "hyperparameter_search_results.json"
    output_file.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"\n💾 搜索结果已保存到: {output_file}")
    
    # 显示最佳结果
    if results:
        print("\n🏆 最佳超参数配置:")
        best_result = results[0]
        print(f"  学习率: {best_result['learning_rate']}")
        print(f"  批次大小: {best_result['batch_size']}")
        print(f"  隐藏层维度: {best_result['hidden_dim']}")
        print(f"  训练轮数: {best_result['epochs']}")
        print(f"  MAE: {best_result['metrics']['mae']:.4f}")
        print(f"  RMSE: {best_result['metrics']['rmse']:.4f}")
        print(f"  MAPE: {best_result['metrics']['mape']:.4f}")
        
        # 检查是否达到目标
        if best_result['metrics']['mae'] < 60:
            print("\n✅ 成功！达到目标 MAE < 60")
        else:
            print("\n⚠️ 未达到目标 MAE < 60，需要进一步调整超参数")
    
    print("\n" + "=" * 60)
    print("✅ 超参数搜索完成！")
    print("=" * 60)

if __name__ == "__main__":
    hyperparameter_search()