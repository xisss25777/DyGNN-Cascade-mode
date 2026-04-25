# 时间片特征差异测试脚本
import numpy as np
import torch
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cascade_model.data import load_wikipedia_cascades
from cascade_model.dynamic_graph import build_snapshots
from cascade_model.dgnn import snapshot_to_graph_data
from cascade_model.dataset_profiles import load_dataset_and_config
from cascade_model.dgnn import build_dgnn_dataset

def test_temporal_feature_variation():
    """测试时间片特征是否不同"""
    print("=" * 60)
    print("🧪 时间片特征差异测试")
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

        # 2. 加载级联数据
        print(f"加载数据文件: {input_path}")
        cascades = load_wikipedia_cascades(input_path)
        print(f"级联数量: {len(cascades)}")

        # 3. 测试第一个级联
        cascade = cascades[0]
        print(f"\n测试级联: {cascade.cascade_id}")
        print(f"级联大小: {cascade.final_size}")
        print(f"事件数量: {len(cascade.events)}")

        # 4. 构建快照
        observation_seconds = 6 * 3600  # 6小时
        slice_seconds = 1800  # 30分钟
        snapshots = build_snapshots(cascade, observation_seconds, slice_seconds)
        print(f"时间片数量: {len(snapshots)}")

        # 5. 分析每个时间片的特征
        feature_means = []
        feature_stds = []
        node_counts = []
        edge_counts = []
        new_event_counts = []

        print("\n时间片特征分析:")
        print("-" * 80)
        print(f"{'时间片':<6} {'节点数':<8} {'边数':<8} {'新事件':<8} {'特征均值':<10} {'特征标准差':<12} {'特征维度':<10}")
        print("-" * 80)

        for t, snapshot in enumerate(snapshots):
            # 提取特征
            graph_data = snapshot_to_graph_data(cascade, snapshot)
            node_features = graph_data.node_features

            # 统计信息
            node_count = len(snapshot.seen_nodes)
            edge_count = len(snapshot.edges)
            new_event_count = len(snapshot.new_events)

            if node_features is not None and node_features.numel() > 0:
                feature_mean = node_features.mean().item()
                feature_std = node_features.std().item()
                feature_dim = node_features.shape[1]
            else:
                feature_mean = 0.0
                feature_std = 0.0
                feature_dim = 0

            feature_means.append(feature_mean)
            feature_stds.append(feature_std)
            node_counts.append(node_count)
            edge_counts.append(edge_count)
            new_event_counts.append(new_event_count)

            print(f"{t:<6} {node_count:<8} {edge_count:<8} {new_event_count:<8} {feature_mean:<10.4f} {feature_std:<12.4f} {feature_dim:<10}")

        # 6. 分析特征变化
        print("\n" + "=" * 80)
        print("特征变化分析:")
        print("-" * 80)

        # 计算特征均值的变化
        if feature_means:
            mean_std = np.std(feature_means)
            mean_range = max(feature_means) - min(feature_means)
            print(f"特征均值标准差: {mean_std:.4f}")
            print(f"特征均值范围: [{min(feature_means):.4f}, {max(feature_means):.4f}]")
            print(f"特征均值变化幅度: {mean_range:.4f}")

        # 计算节点数的变化
        if node_counts:
            node_std = np.std(node_counts)
            node_range = max(node_counts) - min(node_counts)
            print(f"节点数标准差: {node_std:.2f}")
            print(f"节点数范围: [{min(node_counts)}, {max(node_counts)}]")
            print(f"节点数变化幅度: {node_range}")

        # 计算边数的变化
        if edge_counts:
            edge_std = np.std(edge_counts)
            edge_range = max(edge_counts) - min(edge_counts)
            print(f"边数标准差: {edge_std:.2f}")
            print(f"边数范围: [{min(edge_counts)}, {max(edge_counts)}]")
            print(f"边数变化幅度: {edge_range}")

        # 7. 评估特征时间变化
        print("\n" + "=" * 80)
        print("时间特征变化评估:")
        print("-" * 80)

        if feature_means:
            # 检查特征是否随时间变化
            unique_means = len(set(round(m, 4) for m in feature_means))
            if unique_means == 1:
                print("❌ 严重问题：所有时间片特征均值相同")
                print("  特征提取没有考虑时间维度")
            elif unique_means < 3:
                print("⚠️  警告：时间片特征变化很小")
                print("  特征提取可能存在问题")
            else:
                print("✅ 良好：时间片特征有明显变化")
                print(f"  发现 {unique_means} 个不同的特征均值")

        # 检查节点数是否随时间增长
        if node_counts:
            is_growing = all(node_counts[i] <= node_counts[i+1] for i in range(len(node_counts)-1))
            if is_growing:
                print("✅ 节点数随时间增长正常")
            else:
                print("⚠️  节点数增长异常")

        # 8. 验证多个级联
        print("\n" + "=" * 80)
        print("验证多个级联:")
        print("-" * 80)

        max_cascades = min(3, len(cascades))
        for i in range(max_cascades):
            cascade = cascades[i]
            snapshots = build_snapshots(cascade, observation_seconds, slice_seconds)

            # 检查该级联的时间片特征
            cascade_feature_means = []
            for snapshot in snapshots:
                graph_data = snapshot_to_graph_data(cascade, snapshot)
                if graph_data.node_features is not None and graph_data.node_features.numel() > 0:
                    cascade_feature_means.append(graph_data.node_features.mean().item())

            if cascade_feature_means:
                cascade_unique_means = len(set(round(m, 4) for m in cascade_feature_means))
                print(f"级联 {i+1}: {cascade_unique_means} 个不同的特征均值")
                if cascade_unique_means == 1:
                    print("  ❌ 所有时间片特征相同")
                else:
                    print("  ✅ 特征随时间变化")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_feature_scaling_effect():
    """测试特征缩放效果"""
    print("\n" + "=" * 60)
    print("🧪 特征缩放效果测试")
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

        # 3. 检查特征缩放
        if dataset:
            sample = dataset[0]
            print(f"样本数量: {len(dataset)}")
            print(f"时间片数量: {len(sample.snapshots)}")

            # 检查特征范围
            print("\n特征范围检查:")
            print("-" * 80)

            for t, snapshot in enumerate(sample.snapshots[:3]):  # 只显示前3个
                node_features = snapshot.node_features
                graph_features = snapshot.graph_features

                if node_features is not None:
                    print(f"时间片 {t} 节点特征:")
                    print(f"  均值: {node_features.mean().item():.4f}")
                    print(f"  标准差: {node_features.std().item():.4f}")
                    print(f"  最小值: {node_features.min().item():.4f}")
                    print(f"  最大值: {node_features.max().item():.4f}")

                if graph_features is not None:
                    print(f"时间片 {t} 图特征:")
                    print(f"  均值: {graph_features.mean().item():.4f}")
                    print(f"  标准差: {graph_features.std().item():.4f}")
                    print(f"  最小值: {graph_features.min().item():.4f}")
                    print(f"  最大值: {graph_features.max().item():.4f}")

                print()

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_temporal_feature_variation()
    test_feature_scaling_effect()

    print("\n" + "=" * 60)
    print("🎯 测试完成")
    print("=" * 60)