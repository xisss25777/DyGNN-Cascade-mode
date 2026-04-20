# 特征问题诊断脚本
import numpy as np
import torch
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cascade_model.data import load_wikipedia_cascades
from cascade_model.dataset_profiles import load_dataset_and_config
from cascade_model.dgnn import build_dgnn_dataset, snapshot_to_graph_data
from cascade_model.dynamic_graph import build_snapshots

def comprehensive_feature_diagnosis():
    """综合特征问题诊断"""
    print("=" * 60)
    print("🔍 综合特征问题诊断")
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
        cascades = load_wikipedia_cascades(input_path)
        print(f"原始数据统计:")
        print(f"  样本数: {len(cascades)}")
        
        # 3. 检查第一个级联
        cascade = cascades[0]
        print(f"\n测试级联: {cascade.cascade_id}")
        print(f"级联大小: {cascade.final_size}")
        
        # 4. 构建快照
        snapshots = build_snapshots(cascade, 6*3600, 1800)
        print(f"时间片数量: {len(snapshots)}")
        
        # 5. 检查每个时间片的特征
        feature_means = []
        feature_stds = []
        print("\n1. 检查时间片特征差异:")
        
        for t, snapshot in enumerate(snapshots):
            graph_data = snapshot_to_graph_data(cascade, snapshot)
            node_features = graph_data.node_features
            graph_features = graph_data.graph_features
            
            if node_features is not None:
                mean = node_features.mean().item()
                std = node_features.std().item()
                feature_means.append(mean)
                feature_stds.append(std)
                
                print(f"  时间片{t}: 均值={mean:.4f}, 标准差={std:.4f}, 形状={node_features.shape}")
        
        # 6. 检查特征是否随时间变化
        if feature_means:
            unique_means = len(set(round(m, 4) for m in feature_means))
            if unique_means == 1:
                print("\n❌ 问题确认：所有时间片特征均值相同")
                print("  这意味着特征提取没有考虑时间维度")
            else:
                print(f"\n✅ 特征随时间变化正常 (发现{unique_means}个不同的均值)")
        
        # 7. 检查特征-标签比例
        print("\n2. 检查特征-标签比例:")
        
        # 构建数据集获取标签
        dataset_name, cascades, config = load_dataset_and_config(input_path, "wikipedia")
        dataset = build_dgnn_dataset(
            cascades,
            observation_seconds=config.observation_seconds,
            slice_seconds=config.slice_seconds,
        )
        
        if dataset:
            sample = dataset[0]
            target = sample.target
            print(f"  标签值: {target:.2f}")
            
            if sample.snapshots:
                last_snapshot = sample.snapshots[-1]
                node_features = last_snapshot.node_features
                if node_features is not None:
                    feature_mean = node_features.mean().item()
                    ratio = feature_mean / target
                    print(f"  特征均值: {feature_mean:.4f}")
                    print(f"  特征-标签比例: {ratio:.4f}")
                    
                    if ratio < 0.1:
                        print("  ⚠️  特征-标签比例过小，需要重缩放")
                    elif 0.2 < ratio < 0.5:
                        print("  ✅ 特征-标签比例合理")
                    else:
                        print("  ⚠️  特征-标签比例过大")
        
        # 8. 检查特征重缩放函数
        print("\n3. 检查特征重缩放函数:")
        from cascade_model.dgnn import rescale_features
        
        # 测试重缩放函数
        test_features = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        scaled_features = rescale_features(test_features, target_mean=50, target_std=25)
        print(f"  测试特征: {test_features}")
        print(f"  重缩放后: {scaled_features}")
        print(f"  原始均值: {test_features.mean():.4f}")
        print(f"  重缩放后均值: {scaled_features.mean():.4f}")
        
        return feature_means
        
    except Exception as e:
        print(f"❌ 诊断失败: {e}")
        import traceback
        traceback.print_exc()
        return []

def validate_temporal_features():
    """验证特征是否随时间变化"""
    print("\n" + "=" * 60)
    print("🧪 时间特征变化验证")
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
        cascades = load_wikipedia_cascades(input_path)
        
        # 3. 检查前2个级联
        for cascade_idx, cascade in enumerate(cascades[:2]):
            print(f"\n验证级联 {cascade_idx} 的时间特征变化:")
            
            # 构建快照
            snapshots = build_snapshots(cascade, 6*3600, 1800)
            feature_means = []
            
            for t, snapshot in enumerate(snapshots):
                graph_data = snapshot_to_graph_data(cascade, snapshot)
                if graph_data.node_features is not None:
                    mean = graph_data.node_features.mean().item()
                    feature_means.append(mean)
                    
                    if t == 0 or t == len(snapshots) - 1:
                        print(f"  时间片{t}: 特征均值={mean:.4f}")
            
            # 检查是否所有时间片特征相同
            if feature_means:
                all_same = len(set(round(m, 4) for m in feature_means)) == 1
                if all_same:
                    print("  ❌ 所有时间片特征相同！需要修复特征提取")
                else:
                    print("  ✅ 特征随时间变化正常")
                    print(f"  特征均值范围: [{min(feature_means):.4f}, {max(feature_means):.4f}]")
    
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()


def test_feature_rescaling_effect():
    """测试特征重缩放效果"""
    print("\n" + "=" * 60)
    print("🧪 特征重缩放效果测试")
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
        
        # 3. 检查多个样本
        print("\n测试多个样本的特征重缩放效果:")
        feature_means = []
        targets = []
        
        for i, sample in enumerate(dataset[:5]):
            if sample.snapshots:
                last_snapshot = sample.snapshots[-1]
                if last_snapshot.node_features is not None:
                    feature_mean = last_snapshot.node_features.mean().item()
                    feature_means.append(feature_mean)
                    targets.append(sample.target)
                    
                    print(f"  样本{i+1}: 特征均值={feature_mean:.4f}, 目标值={sample.target:.2f}")
        
        # 4. 计算统计
        if feature_means and targets:
            avg_feature = np.mean(feature_means)
            avg_target = np.mean(targets)
            ratio = avg_feature / avg_target
            
            print(f"\n统计结果:")
            print(f"  平均特征均值: {avg_feature:.4f}")
            print(f"  平均目标值: {avg_target:.4f}")
            print(f"  特征-标签比例: {ratio:.4f}")
            
            if ratio < 0.1:
                print("  ⚠️  特征-标签比例过小")
            elif 0.2 < ratio < 0.5:
                print("  ✅ 特征-标签比例合理")
            else:
                print("  ⚠️  特征-标签比例过大")
                
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    comprehensive_feature_diagnosis()
    validate_temporal_features()
    test_feature_rescaling_effect()