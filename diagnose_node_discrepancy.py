# 节点数不一致分析脚本
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cascade_model.data import load_wikipedia_cascades
from cascade_model.dynamic_graph import build_snapshots

def analyze_node_discrepancy():
    """分析节点数不一致的根本原因"""
    print("=" * 60)
    print("🔍 节点数不一致分析")
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

        # 3. 分析第一个级联
        cascade = cascades[0]
        print(f"\n分析级联: {cascade.cascade_id}")
        print(f"级联目标大小: {cascade.final_size}")

        # 4. 原始事件分析
        print("\n1. 原始事件分析:")
        events = cascade.events
        print(f"  总事件数: {len(events)}")

        # 统计唯一用户
        unique_users = set()
        for event in events:
            unique_users.add(event.user_id)

        print(f"  唯一用户数: {len(unique_users)}")
        print(f"  平均每个用户事件数: {len(events)/len(unique_users):.2f}")

        # 5. 动态图分析
        print("\n2. 动态图分析:")
        observation_seconds = 86400
        slice_seconds = 7200
        snapshots = build_snapshots(cascade, observation_seconds, slice_seconds)

        # 动态图中的唯一用户
        dynamic_users = set()
        for snapshot in snapshots:
            if hasattr(snapshot, 'seen_nodes'):
                for user_id in snapshot.seen_nodes:
                    dynamic_users.add(user_id)

        print(f"  动态图唯一用户数: {len(dynamic_users)}")

        # 6. 比较差异
        print("\n3. 差异分析:")
        print(f"  原始唯一用户: {len(unique_users)}")
        print(f"  动态图用户: {len(dynamic_users)}")
        print(f"  差异: {len(unique_users) - len(dynamic_users)}")
        print(f"  覆盖率: {len(dynamic_users)/len(unique_users)*100:.1f}%")
        print(f"  级联目标大小/原始唯一用户: {cascade.final_size/len(unique_users):.2f}倍")

        if len(dynamic_users) < len(unique_users):
            missing_users = unique_users - dynamic_users
            print(f"  缺失用户数: {len(missing_users)}")

            # 分析缺失用户
            print(f"\n4. 缺失用户分析:")
            print(f"  前5个缺失用户的事件信息:")
            for user_id in list(missing_users)[:5]:  # 查看前5个
                # 查找该用户的所有事件
                user_events = [e for e in events if e.user_id == user_id]
                event_times = [e.timestamp for e in user_events]
                print(f"  用户 {user_id}: {len(user_events)}个事件, "
                      f"时间范围[{min(event_times)}, {max(event_times)}]")

        # 7. 时间分析
        print("\n5. 时间分析:")
        timestamps = [event.timestamp for event in events]
        print(f"  事件时间范围: [{min(timestamps)}, {max(timestamps)}]")
        print(f"  观察窗口: {observation_seconds}秒")

        # 分析时间超出观察窗口的事件
        late_events = [e for e in events if e.timestamp > observation_seconds]
        print(f"  超出观察窗口的事件数: {len(late_events)}")
        if late_events:
            late_users = set(e.user_id for e in late_events)
            print(f"  超出观察窗口的用户数: {len(late_users)}")

        # 8. 验证多个级联
        print("\n" + "=" * 90)
        print("6. 验证多个级联:")
        print("-" * 90)

        max_cascades = min(3, len(cascades))
        for i in range(max_cascades):
            cascade = cascades[i]
            events = cascade.events
            unique_users = set(e.user_id for e in events)

            snapshots = build_snapshots(cascade, observation_seconds, slice_seconds)
            dynamic_users = set()
            for snapshot in snapshots:
                if hasattr(snapshot, 'seen_nodes'):
                    dynamic_users.update(snapshot.seen_nodes)

            print(f"级联 {i+1}:")
            print(f"  ID: {cascade.cascade_id}")
            print(f"  级联目标大小: {cascade.final_size}")
            print(f"  事件数: {len(events)}")
            print(f"  唯一用户数: {len(unique_users)}")
            print(f"  动态图用户数: {len(dynamic_users)}")
            print(f"  覆盖率: {len(dynamic_users)/len(unique_users)*100:.1f}%")
            print()

        return len(unique_users), len(dynamic_users)

    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()

def check_cascade_size_definition():
    """检查级联大小的定义"""
    print("\n" + "=" * 60)
    print("🔍 级联大小定义检查")
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

        # 3. 分析级联大小定义
        print("级联大小定义分析:")
        print("-" * 90)

        for i, cascade in enumerate(cascades[:3]):
            events = cascade.events
            unique_users = set(e.user_id for e in events)

            print(f"级联 {i+1}:")
            print(f"  ID: {cascade.cascade_id}")
            print(f"  级联目标大小: {cascade.final_size}")
            print(f"  事件数: {len(events)}")
            print(f"  唯一用户数: {len(unique_users)}")
            print(f"  目标大小/事件数: {cascade.final_size/len(events):.2f}")
            print(f"  目标大小/唯一用户: {cascade.final_size/len(unique_users):.2f}")

            # 判断级联大小的定义
            if cascade.final_size == len(events):
                print(f"  ✅ 级联大小 = 事件数")
            elif cascade.final_size == len(unique_users):
                print(f"  ✅ 级联大小 = 唯一用户数")
            else:
                print(f"  ⚠️  级联大小定义未知")
            print()

    except Exception as e:
        print(f"❌ 检查失败: {e}")
        import traceback
        traceback.print_exc()

def validate_training_targets():
    """验证训练目标"""
    print("\n" + "=" * 60)
    print("🎯 训练目标验证")
    print("=" * 60)
    
    try:
        # 1. 加载数据集
        from cascade_model.dgnn import build_dgnn_dataset
        from cascade_model.dataset_profiles import load_dataset_and_config
        
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
        
        # 2. 加载数据集和配置
        dataset_name, cascades, config = load_dataset_and_config(input_path, "wikipedia")
        dataset = build_dgnn_dataset(
            cascades,
            observation_seconds=config.observation_seconds,
            slice_seconds=config.slice_seconds,
        )
        
        # 3. 分析训练目标
        print("训练目标分析:")
        print("-" * 90)
        
        for i, sample in enumerate(dataset[:3]):
            print(f"样本 {i+1}:")
            print(f"  类型: {type(sample).__name__}")
            
            # 检查可用属性
            if hasattr(sample, 'cascade_id'):
                print(f"  cascade_id: {sample.cascade_id}")
            
            # 尝试获取目标值
            target_size = None
            if hasattr(sample, 'target_size'):
                target_size = sample.target_size
            elif hasattr(sample, 'y'):
                # 可能是直接的目标张量
                target_size = sample.y.item() if hasattr(sample.y, 'item') else sample.y
            elif hasattr(sample, 'target'):
                target_size = sample.target
            
            if target_size is not None:
                print(f"  目标大小: {target_size}")
            else:
                print(f"  目标大小: 未知")
            
            # 动态图的实际大小
            if hasattr(sample, 'snapshots') and sample.snapshots:
                last_snapshot = sample.snapshots[-1]
                if hasattr(last_snapshot, 'seen_nodes'):
                    dynamic_size = len(last_snapshot.seen_nodes)
                else:
                    dynamic_size = "未知"
                
                print(f"  动态图大小: {dynamic_size}")
                if dynamic_size != "未知" and target_size is not None:
                    try:
                        print(f"  比例: {dynamic_size/target_size:.1%}")
                    except:
                        print(f"  比例: 计算失败")
            print()
            
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    analyze_node_discrepancy()
    check_cascade_size_definition()
    validate_training_targets()

    print("\n" + "=" * 60)
    print("🎯 分析完成")
    print("=" * 60)