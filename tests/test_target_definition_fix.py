# 目标定义修复验证脚本
import sys
import os
import numpy as np

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cascade_model.data import load_wikipedia_cascades

def test_target_fix_effect():
    """测试目标定义修复效果"""
    print("目标定义修复测试:")
    print("=" * 60)

    try:
        # 1. 加载原始数据
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

        raw_cascades = load_wikipedia_cascades(input_path)
        print(f"加载级联数量: {len(raw_cascades)}")

        # 2. 分析原始目标定义
        print("\n原始目标定义分析:")
        event_counts = []
        user_counts = []
        target_sizes = []
        ratios = []

        for cascade in raw_cascades[:10]:
            events = len(cascade.events)
            users = len(set([e.user_id for e in cascade.events]))
            target = cascade.target_size
            ratio = events / users if users > 0 else 0

            event_counts.append(events)
            user_counts.append(users)
            target_sizes.append(target)
            ratios.append(ratio)

            print(f"级联 {cascade.cascade_id}:")
            print(f"  事件数: {events}")
            print(f"  用户数: {users}")
            print(f"  目标大小: {target}")
            print(f"  事件/用户比: {ratio:.2f}")
            print()

        print("统计结果:")
        print(f"  平均事件数: {np.mean(event_counts):.1f}")
        print(f"  平均用户数: {np.mean(user_counts):.1f}")
        print(f"  平均目标大小: {np.mean(target_sizes):.1f}")
        print(f"  平均事件/用户比: {np.mean(ratios):.2f}")
        print(f"  事件/用户比范围: [{min(ratios):.2f}, {max(ratios):.2f}]")
        print(f"  事件/用户比标准差: {np.std(ratios):.2f}")

        if np.std(ratios) > 10:
            print("  ⚠️  事件/用户比差异巨大，预测困难")

        # 3. 测试修复方案
        print("\n修复方案测试:")

        # 方案A：使用用户数作为目标
        print("方案A - 用户数作为目标:")
        print(f"  目标范围: [{min(user_counts)}, {max(user_counts)}]")
        print(f"  目标均值: {np.mean(user_counts):.1f}")
        print(f"  目标标准差: {np.std(user_counts):.2f}")

        # 方案B：保持事件数，但增强特征
        print("方案B - 保持事件数，预测事件/用户比:")
        print(f"  事件/用户比标准差: {np.std(ratios):.2f}")

        # 4. 验证修复效果
        print("\n修复效果验证:")
        correct_targets = 0
        for i, cascade in enumerate(raw_cascades[:10]):
            users = len(set([e.user_id for e in cascade.events]))
            if cascade.target_size == users:
                correct_targets += 1
                print(f"级联 {cascade.cascade_id}: ✅ 目标大小正确 ({cascade.target_size})")
            else:
                print(f"级联 {cascade.cascade_id}: ❌ 目标大小错误 (应该是 {users}，实际是 {cascade.target_size})")

        print(f"\n修复准确率: {correct_targets/10*100:.1f}%")

        if correct_targets == 10:
            print("✅ 所有级联的目标大小都已正确修复")
        else:
            print("❌ 部分级联的目标大小仍需修复")

        return event_counts, user_counts, target_sizes, ratios

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


def validate_cascade_consistency():
    """验证级联数据一致性"""
    print("\n" + "=" * 60)
    print("🎯 级联数据一致性验证")
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

        cascades = load_wikipedia_cascades(input_path)

        # 2. 分析一致性
        print("级联一致性分析:")
        print("-" * 90)

        consistency_issues = 0
        for i, cascade in enumerate(cascades[:5]):
            events = len(cascade.events)
            users = len(set([e.user_id for e in cascade.events]))
            target = cascade.target_size

            print(f"级联 {i+1} ({cascade.cascade_id}):")
            print(f"  事件数: {events}")
            print(f"  唯一用户数: {users}")
            print(f"  目标大小: {target}")
            print(f"  动态图输入-目标一致性: {'✅' if target == users else '❌'}")

            if target != users:
                consistency_issues += 1
            print()

        if consistency_issues == 0:
            print("✅ 所有级联数据一致")
        else:
            print(f"❌ 发现 {consistency_issues} 个一致性问题")

    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    event_counts, user_counts, target_sizes, ratios = test_target_fix_effect()
    validate_cascade_consistency()

    print("\n" + "=" * 60)
    print("🎯 测试完成")
    print("=" * 60)

    # 总结修复效果
    if event_counts and user_counts and target_sizes:
        avg_event = np.mean(event_counts)
        avg_user = np.mean(user_counts)
        avg_target = np.mean(target_sizes)

        print(f"\n修复前后对比:")
        print(f"  平均事件数: {avg_event:.1f}")
        print(f"  平均用户数: {avg_user:.1f}")
        print(f"  平均目标大小: {avg_target:.1f}")

        if abs(avg_target - avg_user) < 0.1:
            print("✅ 目标大小已成功修复为用户数")
        else:
            print("❌ 目标大小修复失败")
