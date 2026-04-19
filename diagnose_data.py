# 数据诊断脚本
import numpy as np
import pandas as pd
import torch
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cascade_model.data import load_cascades_from_csv
from cascade_model.dataset_profiles import load_dataset_and_config
from cascade_model.dgnn import build_dgnn_dataset

def diagnose_data_pipeline():
    """诊断数据预处理是否正常"""
    print("\n" + "=" * 60)
    print("🧪 数据预处理诊断")
    print("=" * 60)
    
    try:
        # 1. 加载数据
        input_path = None
        possible_paths = [
            "sample_data/wikipedia.csv",
            "pp/sample_data/wikipedia.csv",
            "../pp/sample_data/wikipedia.csv"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                input_path = path
                break
        
        if input_path is None:
            print(f"❌ 找不到数据文件")
            return
        
        # 2. 加载原始数据
        print(f"加载数据文件: {input_path}")
        from cascade_model.data import load_wikipedia_cascades
        cascades = load_wikipedia_cascades(input_path)
        print(f"原始数据统计:")
        print(f"  样本数: {len(cascades)}")
        
        # 3. 检查级联大小
        cascade_sizes = [cascade.final_size for cascade in cascades]
        print(f"  级联大小均值: {np.mean(cascade_sizes):.1f}")
        print(f"  级联大小中位数: {np.median(cascade_sizes):.1f}")
        print(f"  级联大小范围: [{np.min(cascade_sizes)}, {np.max(cascade_sizes)}]")
        print(f"  级联大小标准差: {np.std(cascade_sizes):.1f}")
        
        # 4. 检查标签处理
        print(f"\n标签处理检查:")
        print(f"  原始标签均值: {np.mean(cascade_sizes):.2f}")
        
        # 检查log变换
        log_sizes = np.log1p(cascade_sizes)
        print(f"  log变换后均值: {np.mean(log_sizes):.2f}")
        print(f"  log变换后范围: [{np.min(log_sizes):.2f}, {np.max(log_sizes):.2f}]")
        
        # 5. 加载配置和构建数据集
        dataset_name, cascades, config = load_dataset_and_config(input_path, "wikipedia")
        dataset = build_dgnn_dataset(
            cascades,
            observation_seconds=config.observation_seconds,
            slice_seconds=config.slice_seconds,
        )
        
        # 6. 检查数据集样本
        sample = dataset[0]
        print(f"\n数据集样本检查:")
        print(f"  时间片数量: {len(sample.snapshots)}")
        print(f"  目标值: {sample.target:.2f}")
        
        # 7. 检查特征
        if sample.snapshots:
            node_features = sample.snapshots[0].node_features
            graph_features = sample.snapshots[0].graph_features
            print(f"  节点特征维度: {node_features.shape if node_features is not None else 'N/A'}")
            print(f"  图特征维度: {graph_features.shape if graph_features is not None else 'N/A'}")
            
            if node_features is not None:
                print(f"  节点特征均值: {node_features.mean().item():.4f}")
                print(f"  节点特征标准差: {node_features.std().item():.4f}")
            if graph_features is not None:
                print(f"  图特征均值: {graph_features.mean().item():.4f}")
                print(f"  图特征标准差: {graph_features.std().item():.4f}")
        
        # 8. 关键检查
        if np.mean(cascade_sizes) > 100 and (node_features is not None and node_features.mean().item() < 1.0):
            print("\n⚠️  问题：级联大小很大但特征值很小")
            print("  可能特征被过度归一化或缩放错误")
        
        # 9. 检查目标值分布
        targets = [sample.target for sample in dataset]
        print(f"\n目标值分布:")
        print(f"  目标均值: {np.mean(targets):.2f}")
        print(f"  目标范围: [{np.min(targets)}, {np.max(targets)}]")
        print(f"  目标标准差: {np.std(targets):.2f}")
        
        # 10. 对比原始大小和目标值
        print(f"\n原始大小 vs 目标值:")
        print(f"  原始大小均值: {np.mean(cascade_sizes):.2f}")
        print(f"  目标值均值: {np.mean(targets):.2f}")
        print(f"  比例: {np.mean(targets)/np.mean(cascade_sizes):.6f}")
        
        if np.mean(targets) < 1.0 and np.mean(cascade_sizes) > 100:
            print("\n❌ 严重问题：目标值被过度缩放！")
            print(f"  原始级联大小均值: {np.mean(cascade_sizes):.2f}")
            print(f"  目标值均值: {np.mean(targets):.2f}")
            print(f"  缩放比例: {np.mean(targets)/np.mean(cascade_sizes):.6f}")
        
    except Exception as e:
        print(f"❌ 诊断失败: {e}")
        import traceback
        traceback.print_exc()


def test_minimal_model():
    """用最简单的模型验证是否能学到东西"""
    print("\n" + "=" * 60)
    print("🧪 简化模型验证")
    print("=" * 60)
    
    try:
        # 1. 加载数据
        input_path = None
        possible_paths = [
            "sample_data/wikipedia.csv",
            "pp/sample_data/wikipedia.csv",
            "../pp/sample_data/wikipedia.csv"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                input_path = path
                break
        
        if input_path is None:
            print(f"❌ 找不到数据文件")
            return
        
        # 2. 构建数据集
        dataset_name, cascades, config = load_dataset_and_config(input_path, "wikipedia")
        dataset = build_dgnn_dataset(
            cascades,
            observation_seconds=config.observation_seconds,
            slice_seconds=config.slice_seconds,
        )
        
        # 3. 准备数据
        from cascade_model.dgnn import DynamicCascadeGNN
        
        # 提取特征和标签
        features = []
        labels = []
        
        for sample in dataset[:50]:  # 使用前50个样本
            # 简单特征：使用最后一个时间片的图特征
            if sample.snapshots:
                last_snapshot = sample.snapshots[-1]
                graph_features = last_snapshot.graph_features
                if graph_features is not None:
                    features.append(graph_features.numpy())
                    labels.append(sample.target)
        
        features = np.array(features)
        labels = np.array(labels)
        
        print(f"简化模型数据:")
        print(f"  特征维度: {features.shape}")
        print(f"  标签维度: {labels.shape}")
        print(f"  标签均值: {labels.mean():.2f}")
        print(f"  标签范围: [{labels.min()}, {labels.max()}]")
        
        # 4. 创建最简单的线性模型
        import torch.nn as nn
        import torch.optim as optim
        
        input_dim = features.shape[1]
        simple_model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # 5. 训练
        X_tensor = torch.tensor(features, dtype=torch.float32)
        y_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        
        optimizer = optim.Adam(simple_model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()
        
        print(f"\n训练简单模型...")
        for epoch in range(20):
            optimizer.zero_grad()
            outputs = simple_model(X_tensor)
            loss = loss_fn(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}, Loss: {loss.item():.4f}")
        
        # 6. 评估
        simple_model.eval()
        with torch.no_grad():
            predictions = simple_model(X_tensor).squeeze().numpy()
        
        mae = np.mean(np.abs(predictions - labels))
        print(f"\n简单线性模型评估:")
        print(f"  MAE: {mae:.4f}")
        print(f"  预测均值: {predictions.mean():.4f}")
        print(f"  预测范围: [{predictions.min():.4f}, {predictions.max():.4f}]")
        print(f"  标签均值: {labels.mean():.4f}")
        
        if mae < 0.5:
            print("\n✅ 简单模型能学到东西，问题在DGNN架构")
        else:
            print("\n❌ 简单模型也学不到，问题在数据或特征")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="数据诊断工具")
    parser.add_argument("--dataset", default="wikipedia", help="数据集名称")
    parser.add_argument("--check-data", action="store_true", help="检查数据流程")
    parser.add_argument("--test-minimal", action="store_true", help="测试简化模型")
    
    args = parser.parse_args()
    
    if args.check_data:
        diagnose_data_pipeline()
    elif args.test_minimal:
        test_minimal_model()
    else:
        diagnose_data_pipeline()
        test_minimal_model()