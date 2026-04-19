# 事件筛选逻辑诊断脚本
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cascade_model.data import load_wikipedia_cascades
from cascade_model.dynamic_graph import build_snapshots

def diagnose_event_selection():
    """诊断事件筛选逻辑"""
    print("=" * 60)
    print("🔍 事件筛选逻辑诊断")
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

        # 4. 检查事件时间
        timestamps = [event.timestamp for event in cascade.events]
        print(f"事件时间范围: [{min(timestamps)}, {max(timestamps)}]")

        # 5. 构建快照
        observation_seconds = 86400  # 24小时
        slice_seconds = 7200  # 2小时

        print(f"\n时间片配置:")
        print(f"观察窗口: {observation_seconds}秒")
        print(f"时间片长度: {slice_seconds}秒")
        print(f"时间片数量: {observation_seconds//slice_seconds}")

        # 6. 诊断时间片筛选
        print("\n时间片事件筛选诊断:")
        print("-" * 90)
        print(f"{'时间片':<6} {'截止时间':<12} {'筛选事件':<10} {'节点数':<8} {'期望节点':<10} {'状态':<10}")
        print("-" * 90)

        # 手动检查每个时间片
        event_index = 0
        seen_nodes = set()
        expected_node_counts = []

        for slice_idx in range(12):
            cutoff_time = (slice_idx + 1) * slice_seconds

            # 手动筛选事件
            selected_events = []
            while event_index < len(cascade.events) and cascade.events[event_index].timestamp <= cutoff_time:
                selected_events.append(cascade.events[event_index])
                seen_nodes.add(cascade.events[event_index].user_id)
                event_index += 1

            node_count = len(seen_nodes)
            expected_node_counts.append(node_count)

            status = "✅" if node_count > 0 else "❌"
            print(f"{slice_idx:<6} {cutoff_time:<12} {len(selected_events):<10} {node_count:<8} {node_count:<10} {status:<10}")

        # 7. 构建实际快照
        print("\n" + "=" * 90)
        print("实际快照构建结果:")
        print("-" * 90)

        snapshots = build_snapshots(cascade, observation_seconds, slice_seconds)
        actual_node_counts = [len(s.seen_nodes) for s in snapshots]

        print(f"{'时间片':<6} {'实际节点数':<10} {'期望节点数':<10} {'差异':<8} {'状态':<10}")
        print("-" * 90)

        for i, (actual, expected) in enumerate(zip(actual_node_counts, expected_node_counts)):
            diff = abs(actual - expected)
            status = "✅" if diff == 0 else "❌"
            print(f"{i:<6} {actual:<10} {expected:<10} {diff:<8} {status:<10}")

        # 8. 最终统计
        print("\n" + "=" * 90)
        print("最终统计:")
        print("-" * 90)

        final_actual = actual_node_counts[-1] if actual_node_counts else 0
        final_expected = expected_node_counts[-1] if expected_node_counts else 0
        total_nodes = cascade.final_size

        print(f"级联总节点数: {total_nodes}")
        print(f"实际最终节点数: {final_actual}")
        print(f"期望最终节点数: {final_expected}")
        print(f"覆盖率: {final_actual/total_nodes*100:.1f}%")

        if final_actual == total_nodes:
            print("✅ 动态图构建完整")
        elif final_actual >= total_nodes * 0.9:
            print("✅ 动态图构建基本完整")
        else:
            print("❌ 动态图构建不完整")

        # 9. 验证多个级联
        print("\n" + "=" * 90)
        print("验证多个级联:")
        print("-" * 90)

        max_cascades = min(3, len(cascades))
        for i in range(max_cascades):
            cascade = cascades[i]
            snapshots = build_snapshots(cascade, observation_seconds, slice_seconds)
            final_nodes = len(snapshots[-1].seen_nodes) if snapshots else 0
            coverage = final_nodes / cascade.final_size * 100

            print(f"级联 {i+1}:")
            print(f"  ID: {cascade.cascade_id}")
            print(f"  总节点数: {cascade.final_size}")
            print(f"  最终节点数: {final_nodes}")
            print(f"  覆盖率: {coverage:.1f}%")

            if coverage >= 90:
                print(f"  ✅ 构建完整")
            else:
                print(f"  ❌ 构建不完整")
            print()

    except Exception as e:
        print(f"❌ 诊断失败: {e}")
        import traceback
        traceback.print_exc()


def verify_consistency_between_tests():
    """验证测试结果一致性"""
    print("\n" + "=" * 60)
    print("🎯 测试结果一致性验证")
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
        cascades = load_wikipedia_cascades(input_path)
        cascade = cascades[0]

        # 3. 构建快照
        observation_seconds = 86400
        slice_seconds = 7200
        snapshots = build_snapshots(cascade, observation_seconds, slice_seconds)

        # 4. 分析特征变化（动态图验证方式）
        from cascade_model.dgnn import snapshot_to_graph_data
        feature_means = []
        for snapshot in snapshots:
            graph_data = snapshot_to_graph_data(cascade, snapshot)
            if graph_data.node_features is not None:
                feature_means.append(graph_data.node_features.mean().item())

        if feature_means:
            dynamic_variation = max(feature_means) - min(feature_means)
            print(f"动态图验证 - 特征变化幅度: {dynamic_variation:.1f}")

        # 5. 分析数据集特征（数据集测试方式）
        from cascade_model.dgnn import build_dgnn_dataset
        from cascade_model.dataset_profiles import load_dataset_and_config

        dataset_name, cascades, config = load_dataset_and_config(input_path, "wikipedia")
        dataset = build_dgnn_dataset(
            cascades,
            observation_seconds=config.observation_seconds,
            slice_seconds=config.slice_seconds,
        )

        if dataset:
            sample = dataset[0]
            dataset_feature_means = []
            for snapshot in sample.snapshots:
                if snapshot.node_features is not None:
                    dataset_feature_means.append(snapshot.node_features.mean().item())

            if dataset_feature_means:
                dataset_variation = max(dataset_feature_means) - min(dataset_feature_means)
                print(f"数据集测试 - 特征变化幅度: {dataset_variation:.1f}")

                if 'dynamic_variation' in locals():
                    ratio = dynamic_variation / dataset_variation if dataset_variation > 0 else 0
                    print(f"变化幅度比例: {ratio:.1f}倍")

                    if ratio > 10:
                        print("⚠️  严重不一致：两个测试结果相差巨大")
                    else:
                        print("✅ 测试结果一致")

    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    diagnose_event_selection()
    verify_consistency_between_tests()

    print("\n" + "=" * 60)
    print("🎯 诊断完成")
    print("=" * 60)