"""
test_speed.py - 模型速度诊断脚本
用于检查DGNN模型的计算性能和参数规模
"""

import sys
import os
import time
import torch
import torch.nn as nn

# 添加项目路径
sys.path.append('.')

# 1. 测试模型创建速度
print("=" * 60)
print("🚀 开始DGNN模型速度诊断")
print("=" * 60)

# 导入配置和模型
from cascade_model.config import PipelineConfig
from cascade_model.dgnn import DynamicCascadeGNN
from cascade_model.data import Cascade, Event
from cascade_model.dynamic_graph import Snapshot

# 2. 创建最小化配置
config = PipelineConfig(
    observation_seconds=3600,  # 1小时观测窗口
    slice_seconds=1800,  # 30分钟一个时间片
    epochs=3,  # 只跑3轮测试
    learning_rate=0.001
)

print("📊 测试配置:")
print(f"  - 观测窗口: {config.observation_seconds / 3600}小时")
print(f"  - 时间片长度: {config.slice_seconds / 60}分钟")
print(f"  - 时间片数量: {config.observation_seconds // config.slice_seconds}")
print(f"  - 训练轮数: {config.epochs}")
print(f"  - 学习率: {config.learning_rate}")

# 3. 检查计算设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"💻 使用设备: {device}")
if torch.cuda.is_available():
    print(f"  - GPU型号: {torch.cuda.get_device_name(0)}")
    print(f"  - GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# 4. 创建模型
print("\n" + "=" * 60)
print("🧠 创建DynamicCascadeGNN模型...")
start_time = time.time()

try:
    # 模拟输入维度
    input_dim = 17  # 节点特征维度
    graph_dim = 10  # 图特征维度

    model = DynamicCascadeGNN(input_dim=input_dim, graph_dim=graph_dim)
    creation_time = time.time() - start_time
    print(f"✅ 模型创建成功，耗时: {creation_time:.2f}秒")

    # 5. 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"📈 模型参数量分析:")
    print(f"  - 总参数: {total_params:,}")
    print(f"  - 可训练参数: {trainable_params:,}")

    if total_params > 1_000_000:
        print("⚠️  警告：模型参数量超过100万，可能导致训练缓慢")
        print("💡 建议：考虑减小 hidden_dim")
    elif total_params > 500_000:
        print("⚠️  注意：模型参数量较大")
    else:
        print("✅ 模型规模适中")

    # 6. 检查模型各层参数
    print("\n" + "=" * 60)
    print("🔍 模型各层参数详情:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  - {name}: {list(param.shape)} | 参数量: {param.numel():,}")

    # 7. 测试前向传播速度
    print("\n" + "=" * 60)
    print("⚡ 测试前向传播速度...")

    # 导入GraphSnapshotData类
    from cascade_model.dgnn import GraphSnapshotData

    # 创建模拟数据
    batch_size = 32
    num_snapshots = config.observation_seconds // config.slice_seconds

    # 模拟一个批次的数据
    test_input = []
    for _ in range(batch_size):
        # 模拟一个级联的时间片
        snapshots = []
        for _ in range(num_snapshots):
            # 随机创建节点特征和边
            num_nodes = torch.randint(5, 20, (1,)).item()
            node_features = torch.randn(num_nodes, input_dim)
            num_edges = torch.randint(5, 30, (1,)).item()
            edge_index = torch.randint(0, num_nodes, (2, num_edges))
            graph_features = torch.randn(graph_dim)  # 宏观特征

            # 创建GraphSnapshotData对象
            snapshot = GraphSnapshotData(
                node_features=node_features,
                edge_index=edge_index,
                graph_features=graph_features
            )
            snapshots.append(snapshot)
        test_input.append(snapshots)

    # 测试前向传播
    model.eval()
    with torch.no_grad():
        forward_times = []
        for i in range(3):  # 跑3次取平均
            start = time.time()
            for snapshots in test_input[:2]:  # 只测2个样本
                try:
                    output = model(snapshots)
                except Exception as e:
                    print(f"❌ 前向传播失败: {e}")
                    break
            forward_times.append(time.time() - start)

    if forward_times:
        avg_time = sum(forward_times) / len(forward_times)
        print(f"✅ 前向传播测试结果:")
        print(f"  - 平均耗时: {avg_time:.3f}秒/样本")
        print(f"  - 预估50轮训练时间: {avg_time * 1000 * config.epochs / 3600:.1f}小时")

    # 8. 内存使用检查
    print("\n" + "=" * 60)
    print("💾 内存使用检查:")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1e6
        reserved = torch.cuda.memory_reserved() / 1e6
        print(f"  - GPU已分配内存: {allocated:.1f} MB")
        print(f"  - GPU保留内存: {reserved:.1f} MB")

    # 9. 优化建议
    print("\n" + "=" * 60)
    print("💡 优化建议:")

    if total_params > 1_000_000:
        print("1. 减小模型规模:")
        print("   - 修改 dgnn.py 中的 hidden_dim")
        print("   - 减小 GCN 的层数")
        print("   - 减小 GRU 的隐藏层维度")

    if not torch.cuda.is_available():
        print("2. 启用GPU加速:")
        print("   - 检查CUDA和cuDNN安装")
        print("   - 确保PyTorch支持CUDA")

    if avg_time > 1.0:  # 如果单个样本前向传播超过1秒
        print("3. 优化计算效率:")
        print("   - 减小批次大小 (batch_size)")
        print("   - 减少时间片数量")
        print("   - 使用混合精度训练")

    print("\n4. 训练策略调整:")
    print("   - 先跑5-10轮看收敛趋势")
    print("   - 添加学习率调度器")
    print("   - 使用梯度裁剪防止梯度爆炸")
    print("   - 使用早停机制避免过拟合")

except Exception as e:
    print(f"❌ 模型测试失败: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 60)
print("✅ 诊断完成")
print("=" * 60)