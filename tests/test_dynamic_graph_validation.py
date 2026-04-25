# 动态图增长验证脚本
import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cascade_model.data import load_wikipedia_cascades
from cascade_model.dynamic_graph import build_snapshots
from cascade_model.dgnn import build_dgnn_dataset, snapshot_to_graph_data
from cascade_model.dataset_profiles import load_dataset_and_config

def validate_dynamic_graph_growth_after_normalization():
    """验证归一化后的动态图增长"""
    print("=" * 60)
    print("🎯 动态图增长验证")
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
        observation_seconds = 86400  # 24小时
        slice_seconds = 7200  # 2小时
        
        print(f"\n时间片配置:")
        print(f"观察窗口: {observation_seconds}秒 ({observation_seconds/3600:.1f}小时)")
        print(f"时间片长度: {slice_seconds}秒 ({slice_seconds/60:.1f}分钟)")
        print(f"时间片数量: {observation_seconds//slice_seconds}")
        
        snapshots = build_snapshots(cascade, observation_seconds, slice_seconds)
        print(f"\n构建的快照数: {len(snapshots)}")
        
        # 5. 检查每个快照的增长
        node_counts = []
        edge_counts = []
        new_event_counts = []
        event_timestamps = []
        
        print("\n时间片增长分析:")
        print("-" * 90)
        print(f"{'时间片':<6} {'截止时间':<12} {'节点数':<8} {'边数':<8} {'新事件':<8} {'事件时间范围':<20} {'累计用户':<8}")
        print("-" * 90)
        
        for t, snapshot in enumerate(snapshots):
            node_count = len(snapshot.seen_nodes)
            edge_count = len(snapshot.edges)
            new_event_count = len(snapshot.new_events)
            
            # 收集事件时间
            if snapshot.new_events:
                event_times = [event.timestamp for event in snapshot.new_events]
                event_time_range = f"[{min(event_times)}, {max(event_times)}]"
            else:
                event_time_range = "[]"
            
            node_counts.append(node_count)
            edge_counts.append(edge_count)
            new_event_counts.append(new_event_count)
            
            print(f"{t:<6} {snapshot.cutoff_time:<12} {node_count:<8} {edge_count:<8} {new_event_count:<8} {event_time_range:<20} {node_count:<8}")
        
        # 6. 分析增长模式
        print("\n" + "=" * 90)
        print("增长模式分析:")
        print("-" * 90)
        
        if node_counts:
            print(f"  起始节点数: {node_counts[0]}")
            print(f"  最终节点数: {node_counts[-1]}")
            print(f"  总增长: {node_counts[-1] - node_counts[0]}")
            print(f"  节点数标准差: {np.std(node_counts):.2f}")
            print(f"  节点数变化幅度: {max(node_counts) - min(node_counts)}")
            
            if node_counts[-1] != cascade.final_size:
                print(f"  ⚠️  警告：最终节点数({node_counts[-1]})不等于级联大小({cascade.final_size})")
            else:
                print(f"  ✅ 最终节点数等于级联大小")
            
            # 检查是否单调增长
            is_monotonic = all(node_counts[i] <= node_counts[i+1] for i in range(len(node_counts)-1))
            if is_monotonic:
                print(f"  ✅ 节点数单调增长")
            else:
                print(f"  ❌ 节点数非单调增长")
        
        if edge_counts:
            print(f"  起始边数: {edge_counts[0]}")
            print(f"  最终边数: {edge_counts[-1]}")
            print(f"  总增长: {edge_counts[-1] - edge_counts[0]}")
        
        if new_event_counts:
            print(f"  新事件总数: {sum(new_event_counts)}")
            print(f"  平均每时间片新事件: {np.mean(new_event_counts):.1f}")
            print(f"  新事件分布标准差: {np.std(new_event_counts):.2f}")
        
        # 7. 检查事件分布
        print("\n" + "=" * 90)
        print("事件分布检查:")
        print("-" * 90)
        
        if new_event_counts:
            # 计算事件分布的均匀性
            event_std = np.std(new_event_counts)
            event_mean = np.mean(new_event_counts)
            
            print(f"  新事件分布: {new_event_counts}")
            print(f"  标准差: {event_std:.2f}, 均值: {event_mean:.2f}")
            
            if event_std < event_mean * 0.5:
                print(f"  ✅ 事件分布相对均匀")
            else:
                print(f"  ⚠️  事件分布不均匀")
        
        # 8. 验证特征提取
        print("\n" + "=" * 90)
        print("特征提取验证:")
        print("-" * 90)
        
        feature_means = []
        for t, snapshot in enumerate(snapshots):
            graph_data = snapshot_to_graph_data(cascade, snapshot)
            if graph_data.node_features is not None:
                feature_mean = graph_data.node_features.mean().item()
                feature_means.append(feature_mean)
                print(f"  时间片 {t}: 特征均值={feature_mean:.4f}, 形状={graph_data.node_features.shape}")
        
        if feature_means:
            mean_std = np.std(feature_means)
            mean_range = max(feature_means) - min(feature_means)
            print(f"\n  特征均值标准差: {mean_std:.4f}")
            print(f"  特征均值范围: [{min(feature_means):.4f}, {max(feature_means):.4f}]")
            print(f"  特征均值变化幅度: {mean_range:.4f}")
            
            if mean_range > 5:
                print(f"  ✅ 特征随时间有显著变化")
            elif mean_range > 1:
                print(f"  ✅ 特征随时间有中等变化")
            else:
                print(f"  ⚠️  特征变化较小")
        
        # 9. 验证多个级联
        print("\n" + "=" * 90)
        print("验证多个级联:")
        print("-" * 90)
        
        max_cascades = min(3, len(cascades))
        for i in range(max_cascades):
            cascade = cascades[i]
            
            # 构建快照
            snapshots = build_snapshots(cascade, observation_seconds, slice_seconds)
            node_counts = [len(s.seen_nodes) for s in snapshots]
            
            print(f"级联 {i+1}:")
            print(f"  ID: {cascade.cascade_id}")
            print(f"  大小: {cascade.final_size}")
            print(f"  时间片数: {len(snapshots)}")
            print(f"  节点数范围: [{min(node_counts)}, {max(node_counts)}]")
            print(f"  最终节点数: {node_counts[-1]}")
            
            if node_counts[-1] < cascade.final_size * 0.9:
                print(f"  ⚠️  最终节点数小于级联大小的90%")
            else:
                print(f"  ✅ 最终节点数接近级联大小")
            print()
        
        return node_counts
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return []

def test_dataset_features():
    """测试数据集特征"""
    print("\n" + "=" * 60)
    print("🧪 数据集特征测试")
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
        
        print(f"\n数据集创建成功:")
        print(f"  样本数量: {len(dataset)}")
        
        if dataset:
            sample = dataset[0]
            print(f"  时间片数量: {len(sample.snapshots)}")
            
            # 检查特征变化
            feature_means = []
            for t, snapshot in enumerate(sample.snapshots):
                if snapshot.node_features is not None:
                    feature_mean = snapshot.node_features.mean().item()
                    feature_means.append(feature_mean)
                    print(f"  时间片 {t}: 特征均值={feature_mean:.4f}")
            
            if feature_means:
                mean_std = np.std(feature_means)
                mean_range = max(feature_means) - min(feature_means)
                print(f"\n  特征均值标准差: {mean_std:.4f}")
                print(f"  特征均值范围: [{min(feature_means):.4f}, {max(feature_means):.4f}]")
                print(f"  特征均值变化幅度: {mean_range:.4f}")
                
                if mean_range > 5:
                    print(f"  ✅ 特征随时间有显著变化")
                elif mean_range > 1:
                    print(f"  ✅ 特征随时间有中等变化")
                else:
                    print(f"  ⚠️  特征变化较小")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    validate_dynamic_graph_growth_after_normalization()
    test_dataset_features()
    
    print("\n" + "=" * 60)
    print("🎯 验证完成")
    print("=" * 60)