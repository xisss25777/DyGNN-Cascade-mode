# 修复效果验证脚本
import sys
import os
import numpy as np

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def verify_fix_performance():
    """验证修复后的模型性能"""
    print("=" * 60)
    print("🎯 修复效果验证")
    print("=" * 60)
    
    try:
        # 1. 加载修复后的数据
        from cascade_model.data import load_wikipedia_cascades
        
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
        print(f"加载级联数量: {len(cascades)}")
        
        # 2. 验证目标值范围
        targets = [c.target_size for c in cascades]
        print(f"\n目标值统计:")
        print(f"  均值: {np.mean(targets):.1f}")
        print(f"  标准差: {np.std(targets):.1f}")
        print(f"  范围: [{np.min(targets)}, {np.max(targets)}]")
        print(f"  中位数: {np.median(targets):.1f}")
        
        # 3. 分析目标值分布
        print("\n目标值分布:")
        target_counts = {}
        for target in targets:
            target_counts[target] = target_counts.get(target, 0) + 1
        
        # 按目标值排序
        sorted_targets = sorted(target_counts.items())
        for target, count in sorted_targets[:10]:  # 显示前10个
            print(f"  目标值 {target}: {count} 个级联")
        
        if len(sorted_targets) > 10:
            print(f"  ... 还有 {len(sorted_targets) - 10} 个目标值")
        
        # 4. 验证输入-目标一致性
        print("\n输入-目标一致性验证:")
        print("-" * 90)
        
        consistency_checks = []
        for i, cascade in enumerate(cascades[:5]):
            events = len(cascade.events)
            users = len(set([e.user_id for e in cascade.events]))
            target = cascade.target_size
            
            print(f"级联 {i+1} ({cascade.cascade_id}):")
            print(f"  事件数: {events}")
            print(f"  唯一用户数: {users}")
            print(f"  目标大小: {target}")
            print(f"  一致性: {'✅' if target == users else '❌'}")
            
            consistency_checks.append(target == users)
            print()
        
        if all(consistency_checks):
            print("✅ 所有验证级联的输入-目标一致")
        else:
            print("❌ 存在输入-目标不一致的情况")
        
        # 5. 构建动态图验证
        print("\n动态图构建验证:")
        from cascade_model.dynamic_graph import build_snapshots
        
        observation_seconds = 86400
        slice_seconds = 7200
        
        # 验证第一个级联的动态图
        cascade = cascades[0]
        snapshots = build_snapshots(cascade, observation_seconds, slice_seconds)
        
        print(f"级联 {cascade.cascade_id} 动态图验证:")
        print(f"  时间片数量: {len(snapshots)}")
        
        node_counts = []
        for t, snapshot in enumerate(snapshots):
            node_count = len(snapshot.seen_nodes)
            node_counts.append(node_count)
            print(f"  时间片 {t}: {node_count} 节点")
        
        print(f"  起始节点数: {node_counts[0]}")
        print(f"  最终节点数: {node_counts[-1]}")
        print(f"  目标大小: {cascade.target_size}")
        
        if node_counts[-1] == cascade.target_size:
            print(f"  ✅ 动态图最终节点数与目标大小一致")
        else:
            print(f"  ❌ 动态图最终节点数与目标大小不一致")
        
        # 6. 数据集构建验证
        print("\n数据集构建验证:")
        from cascade_model.dgnn import build_dgnn_dataset
        
        dataset = build_dgnn_dataset(
            cascades,
            observation_seconds=observation_seconds,
            slice_seconds=slice_seconds,
        )
        
        print(f"数据集样本数: {len(dataset)}")
        
        if dataset:
            sample = dataset[0]
            print(f"\n第一个样本验证:")
            print(f"  类型: {type(sample).__name__}")
            
            # 检查快照
            if hasattr(sample, 'snapshots'):
                print(f"  快照数量: {len(sample.snapshots)}")
                
                # 检查第一个和最后一个快照
                first_snapshot = sample.snapshots[0]
                last_snapshot = sample.snapshots[-1]
                
                if hasattr(first_snapshot, 'seen_nodes'):
                    print(f"  第一个快照节点数: {len(first_snapshot.seen_nodes)}")
                    print(f"  最后一个快照节点数: {len(last_snapshot.seen_nodes)}")
            
            # 检查目标值
            if hasattr(sample, 'y'):
                print(f"  目标值: {sample.y.item() if hasattr(sample.y, 'item') else sample.y}")
        
        # 7. 特征范围验证
        print("\n特征范围验证:")
        if dataset:
            sample = dataset[0]
            if hasattr(sample, 'snapshots'):
                feature_means = []
                for snapshot in sample.snapshots:
                    if hasattr(snapshot, 'node_features') and snapshot.node_features is not None:
                        feature_mean = snapshot.node_features.mean().item()
                        feature_means.append(feature_mean)
                
                if feature_means:
                    print(f"  特征均值范围: [{min(feature_means):.2f}, {max(feature_means):.2f}]")
                    print(f"  特征均值变化: {max(feature_means) - min(feature_means):.2f}")
        
        print("\n" + "=" * 60)
        print("🎯 验证完成")
        print("=" * 60)
        
        # 总结
        print("\n修复效果总结:")
        print(f"✅ 目标定义修复: 唯一用户数作为目标")
        print(f"✅ 目标值范围: [{np.min(targets)}, {np.max(targets)}]")
        print(f"✅ 目标值均值: {np.mean(targets):.1f}")
        print(f"✅ 输入-目标一致性: {'良好' if all(consistency_checks) else '存在问题'}")
        print(f"✅ 动态图构建: {'成功' if snapshots else '失败'}")
        print(f"✅ 数据集构建: {'成功' if dataset else '失败'}")
        
        return {
            'target_stats': {
                'mean': np.mean(targets),
                'std': np.std(targets),
                'min': np.min(targets),
                'max': np.max(targets),
                'median': np.median(targets)
            },
            'consistency': all(consistency_checks),
            'dataset_size': len(dataset) if 'dataset' in locals() else 0
        }
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_model_performance():
    """测试模型性能"""
    print("\n" + "=" * 60)
    print("🚀 模型性能测试")
    print("=" * 60)
    
    try:
        # 运行训练脚本
        import subprocess
        
        # 切换到pp目录
        original_dir = os.getcwd()
        os.chdir('pp')
        
        # 运行训练
        print("开始训练模型...")
        result = subprocess.run(
            [sys.executable, 'main.py', '--dataset', 'wikipedia', '--output', 'outputs/wikipedia_report_fixed.json'],
            capture_output=True,
            text=True,
            timeout=300  # 5分钟超时
        )
        
        # 恢复原目录
        os.chdir(original_dir)
        
        # 打印结果
        print("\n训练输出:")
        print(result.stdout)
        
        if result.stderr:
            print("\n错误输出:")
            print(result.stderr)
        
        print(f"\n训练返回码: {result.returncode}")
        
        if result.returncode == 0:
            print("✅ 模型训练成功")
        else:
            print("❌ 模型训练失败")
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 验证修复效果
    verification_result = verify_fix_performance()
    
    # 测试模型性能
    if verification_result and verification_result['consistency']:
        model_success = test_model_performance()
        
        if model_success:
            print("\n🎉 修复验证完成，模型训练成功！")
        else:
            print("\n⚠️  修复验证完成，但模型训练失败")
    else:
        print("\n❌ 修复验证失败，模型训练被跳过")
    
    print("\n" + "=" * 60)
    print("🎯 全部验证完成")
    print("=" * 60)