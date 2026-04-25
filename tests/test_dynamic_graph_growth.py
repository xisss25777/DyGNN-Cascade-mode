# 动态图增长测试脚本
import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cascade_model.data import load_wikipedia_cascades
from cascade_model.dynamic_graph import build_snapshots

def test_dynamic_graph_growth():
    """测试动态图是否随时间增长"""
    print("=" * 60)
    print("🎯 动态图增长测试")
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
        
        # 4. 分析事件时间
        if cascade.events:
            timestamps = [event.timestamp for event in cascade.events]
            start_time = min(timestamps)
            end_time = max(timestamps)
            print(f"事件时间范围: [{start_time}, {end_time}]")
            print(f"时间跨度: {end_time - start_time}秒")
        
        # 5. 构建快照
        # 测试不同的观察窗口
        observation_seconds = 24 * 3600  # 24小时
        slice_seconds = observation_seconds // 12  # 保持12个时间片
        
        print(f"\n时间片配置:")
        print(f"观察窗口: {observation_seconds}秒 ({observation_seconds/3600:.1f}小时)")
        print(f"时间片长度: {slice_seconds}秒 ({slice_seconds/60:.1f}分钟)")
        print(f"时间片数量: {observation_seconds//slice_seconds}")
        
        snapshots = build_snapshots(cascade, observation_seconds, slice_seconds)
        print(f"\n构建的快照数: {len(snapshots)}")
        
        # 6. 检查每个快照的增长
        node_counts = []
        edge_counts = []
        new_event_counts = []
        cutoff_times = []
        
        print("\n时间片增长分析:")
        print("-" * 80)
        print(f"{'时间片':<6} {'截止时间':<12} {'节点数':<8} {'边数':<8} {'新事件':<8} {'累计用户':<8}")
        print("-" * 80)
        
        for t, snapshot in enumerate(snapshots):
            node_count = len(snapshot.seen_nodes)
            edge_count = len(snapshot.edges)
            new_event_count = len(snapshot.new_events)
            
            node_counts.append(node_count)
            edge_counts.append(edge_count)
            new_event_counts.append(new_event_count)
            cutoff_times.append(snapshot.cutoff_time)
            
            print(f"{t:<6} {snapshot.cutoff_time:<12} {node_count:<8} {edge_count:<8} {new_event_count:<8} {node_count:<8}")
        
        # 7. 分析增长模式
        print("\n" + "=" * 80)
        print("增长模式分析:")
        print("-" * 80)
        
        if node_counts:
            print(f"  起始节点数: {node_counts[0]}")
            print(f"  最终节点数: {node_counts[-1]}")
            print(f"  总增长: {node_counts[-1] - node_counts[0]}")
            print(f"  节点数标准差: {np.std(node_counts):.2f}")
            
            if node_counts[-1] != cascade.final_size:
                print(f"  ⚠️  警告：最终节点数({node_counts[-1]})不等于级联大小({cascade.final_size})")
            
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
        
        # 8. 检查时间片划分
        print("\n" + "=" * 80)
        print("时间片划分检查:")
        print("-" * 80)
        
        if cutoff_times:
            # 检查时间片间隔
            intervals = [cutoff_times[i+1] - cutoff_times[i] for i in range(len(cutoff_times)-1)]
            print(f"  时间片间隔: {intervals}")
            print(f"  平均间隔: {np.mean(intervals):.0f}秒")
            
            # 检查事件分布
            event_distribution = []
            for i, cutoff in enumerate(cutoff_times):
                if i == 0:
                    prev_cutoff = cascade.events[0].timestamp if cascade.events else 0
                else:
                    prev_cutoff = cutoff_times[i-1]
                
                event_count = sum(1 for event in cascade.events if prev_cutoff < event.timestamp <= cutoff)
                event_distribution.append(event_count)
            
            print(f"  事件分布: {event_distribution}")
            print(f"  事件分布标准差: {np.std(event_distribution):.2f}")
        
        # 9. 验证多个级联
        print("\n" + "=" * 80)
        print("验证多个级联:")
        print("-" * 80)
        
        max_cascades = min(3, len(cascades))
        for i in range(max_cascades):
            cascade = cascades[i]
            snapshots = build_snapshots(cascade, observation_seconds, slice_seconds)
            node_counts = [len(s.seen_nodes) for s in snapshots]
            
            print(f"级联 {i+1}:")
            print(f"  总大小: {cascade.final_size}")
            print(f"  时间片数: {len(snapshots)}")
            print(f"  节点数范围: [{min(node_counts)}, {max(node_counts)}]")
            print(f"  最终节点数: {node_counts[-1]}")
            
            if node_counts[-1] < cascade.final_size * 0.5:
                print(f"  ⚠️  警告：最终节点数远小于级联大小")
            elif node_counts[-1] == cascade.final_size:
                print(f"  ✅ 最终节点数等于级联大小")
            print()
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


def diagnose_time_slice_division():
    """诊断时间片划分是否正确"""
    print("\n" + "=" * 60)
    print("🔍 时间片划分诊断")
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
        
        # 3. 分析时间片
        observation_seconds = 6 * 3600
        slice_seconds = 1800
        num_slices = observation_seconds // slice_seconds
        
        print(f"时间片划分诊断:")
        print(f"观察窗口: {observation_seconds}秒 ({observation_seconds/3600:.1f}小时)")
        print(f"时间片长度: {slice_seconds}秒 ({slice_seconds/60:.1f}分钟)")
        print(f"时间片数量: {num_slices}")
        
        # 4. 统计每个时间片的节点数
        if cascade.events:
            start_time = cascade.events[0].timestamp
            
            for t in range(num_slices):
                cutoff_time = start_time + (t + 1) * slice_seconds
                
                # 获取该时间片内的事件
                events_in_slice = [event for event in cascade.events 
                                if event.timestamp <= cutoff_time]
                
                # 提取用户
                users_in_slice = set(event.user_id for event in events_in_slice)
                
                print(f"时间片 {t}:")
                print(f"  截止时间: {cutoff_time}")
                print(f"  事件数: {len(events_in_slice)}")
                print(f"  用户数: {len(users_in_slice)}")
                
                if t == 0 and len(users_in_slice) < 2:
                    print(f"  ⚠️  第一个时间片用户太少")
                if t == num_slices - 1 and len(users_in_slice) != cascade.final_size:
                    print(f"  ⚠️  最后一个时间片未包含所有用户")
                print()
        
    except Exception as e:
        print(f"❌ 诊断失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_dynamic_graph_growth()
    diagnose_time_slice_division()
    
    print("\n" + "=" * 60)
    print("🎯 测试完成")
    print("=" * 60)