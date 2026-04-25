# 特征重缩放测试脚本
import numpy as np
import torch
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cascade_model.data import load_wikipedia_cascades
from cascade_model.dataset_profiles import load_dataset_and_config
from cascade_model.dgnn import build_dgnn_dataset, snapshot_to_graph_data
from cascade_model.data import Cascade

def test_rescaling_effect():
    """测试特征重新缩放效果"""
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

        # 2. 加载原始数据
        print(f"加载数据文件: {input_path}")
        cascades = load_wikipedia_cascades(input_path)
        print(f"原始数据统计:")
        print(f"  样本数: {len(cascades)}")

        # 3. 检查级联大小
        cascade_sizes = [cascade.final_size for cascade in cascades]
        print(f"  级联大小均值: {np.mean(cascade_sizes):.1f}")
        print(f"  级联大小范围: [{np.min(cascade_sizes)}, {np.max(cascade_sizes)}]")

        # 4. 构建数据集
        dataset_name, cascades, config = load_dataset_and_config(input_path, "wikipedia")
        dataset = build_dgnn_dataset(
            cascades,
            observation_seconds=config.observation_seconds,
            slice_seconds=config.slice_seconds,
        )

        # 5. 检查目标值
        targets = [sample.target for sample in dataset]
        print(f"\n目标值分布:")
        print(f"  目标均值: {np.mean(targets):.2f}")
        print(f"  目标范围: [{np.min(targets)}, {np.max(targets)}]")
        print(f"  目标标准差: {np.std(targets):.2f}")

        # 6. 检查特征
        sample = dataset[0]
        if sample.snapshots:
            last_snapshot = sample.snapshots[-1]
            node_features = last_snapshot.node_features
            graph_features = last_snapshot.graph_features

            print(f"\n重缩放后特征统计:")
            print(f"  节点特征维度: {node_features.shape if node_features is not None else 'N/A'}")
            if node_features is not None:
                print(f"  节点特征均值: {node_features.mean().item():.4f}")
                print(f"  节点特征标准差: {node_features.std().item():.4f}")
                print(f"  节点特征范围: [{node_features.min().item():.4f}, {node_features.max().item():.4f}]")

            print(f"  图特征维度: {graph_features.shape if graph_features is not None else 'N/A'}")
            if graph_features is not None:
                print(f"  图特征均值: {graph_features.mean().item():.4f}")
                print(f"  图特征标准差: {graph_features.std().item():.4f}")
                print(f"  图特征范围: [{graph_features.min().item():.4f}, {graph_features.max().item():.4f}]")

        # 7. 计算特征-标签比例
        if node_features is not None:
            feature_mean = node_features.mean().item()
            label_mean = np.mean(targets)
            ratio = feature_mean / label_mean
            print(f"\n特征-标签比例:")
            print(f"  特征均值: {feature_mean:.4f}")
            print(f"  标签均值: {label_mean:.4f}")
            print(f"  比例: {ratio:.4f}")

            if 0.2 < ratio < 0.5:
                print("✅ 特征-标签比例合理")
            else:
                print("⚠️  比例仍需调整")

        # 8. 验证多个样本
        print(f"\n验证多个样本的特征一致性:")
        feature_means = []
        for i, sample in enumerate(dataset[:5]):
            if sample.snapshots:
                last_snapshot = sample.snapshots[-1]
                if last_snapshot.node_features is not None:
                    mean = last_snapshot.node_features.mean().item()
                    feature_means.append(mean)
                    print(f"  样本{i+1}特征均值: {mean:.4f}")

        if feature_means:
            mean_of_means = np.mean(feature_means)
            std_of_means = np.std(feature_means)
            print(f"  特征均值的均值: {mean_of_means:.4f}")
            print(f"  特征均值的标准差: {std_of_means:.4f}")

            if std_of_means < 10:
                print("✅ 特征重缩放一致性好")
            else:
                print("⚠️  特征重缩放一致性差")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_feature_range():
    """测试特征范围是否合理"""
    print("\n" + "=" * 60)
    print("🧪 特征范围测试")
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

        # 3. 检查第一个级联的所有时间片
        cascade = cascades[0]
        print(f"测试级联: {cascade.cascade_id}")
        print(f"级联大小: {cascade.final_size}")

        # 4. 构建快照
        from cascade_model.dgnn import build_snapshots
        snapshots = build_snapshots(cascade, 6*3600, 1800)

        # 5. 检查每个时间片的特征
        for i, snapshot in enumerate(snapshots):
            graph_data = snapshot_to_graph_data(cascade, snapshot)
            node_features = graph_data.node_features
            graph_features = graph_data.graph_features

            print(f"\n时间片{i+1}:")
            print(f"  节点数: {node_features.shape[0] if node_features is not None else 0}")
            if node_features is not None:
                print(f"  节点特征均值: {node_features.mean().item():.4f}")
                print(f"  节点特征标准差: {node_features.std().item():.4f}")
            if graph_features is not None:
                print(f"  图特征均值: {graph_features.mean().item():.4f}")
                print(f"  图特征标准差: {graph_features.std().item():.4f}")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="特征重缩放测试")
    parser.add_argument("--dataset", default="wikipedia", help="数据集名称")
    parser.add_argument("--sample-size", type=int, default=20, help="测试样本数")

    args = parser.parse_args()

    test_rescaling_effect()
    test_feature_range()