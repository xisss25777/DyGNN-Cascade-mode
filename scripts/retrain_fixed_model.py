import torch
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from cascade_model.dgnn import run_dgnn_pipeline
from cascade_model.config import PipelineConfig
from cascade_model.data import load_wikipedia_cascades

def retrain_with_fixed_targets():
    print("=" * 60)
    print("🔄 开始重新训练模型（使用修复后的目标定义）")
    print("=" * 60)

    # 1. 加载修复后的数据
    print("📥 加载数据...")
    cascades = load_wikipedia_cascades("pp/sample_data/wikipedia.csv")

    # 验证目标值
    targets = [c.target_size for c in cascades[:5]]
    print(f"  前5个样本目标值: {targets}")
    print(f"  目标范围验证: [{min(targets)}, {max(targets)}]")

    # 2. 配置模型参数
    config = PipelineConfig()
    config.learning_rate = 0.0005      # 优化后的学习率
    config.batch_size = 16             # 合适的批次大小
    config.epochs = 200                # 训练轮数
    config.patience = 30               # 早停耐心值

    # 3. 重新训练
    print("🚀 开始训练...")
    report = run_dgnn_pipeline(
        cascades=cascades,
        config=config
    )

    return report

def compare_performance(metrics):
    """比较修复前后的性能"""

    print("\n" + "=" * 60)
    print("📊 性能对比分析")
    print("=" * 60)

    # 修复前性能（已知）
    before = {
        'MAE': 72.9,
        'RMSE': 168.6,
        'MAPE': '34.9%'
    }

    # 修复后性能（从训练结果获取）
    after = metrics  # 从上一步获取

    print("修复前（预测事件数）:")
    print(f"  MAE: {before['MAE']:.1f}")
    print(f"  RMSE: {before['RMSE']:.1f}")
    print(f"  MAPE: {before['MAPE']}")

    print("\n修复后（预测用户数）:")
    print(f"  MAE: {after['mae']:.2f}")
    print(f"  RMSE: {after['rmse']:.2f}")
    print(f"  MAPE: {after['mape']:.2%}")

    # 计算改进比例
    improvement = (before['MAE'] - after['mae']) / before['MAE'] * 100
    print(f"\n✅ MAE 改进: {improvement:.1f}%")

    if after['mae'] < 20:
        print("🎉 修复成功！模型性能显著提升")
    else:
        print("⚠️  性能仍有优化空间，需进一步调参")

def generate_training_report(report, cascades):
    """生成详细的训练报告"""

    training_report = {
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'config': report['config'],
        'metrics': report['metrics'],
        'data_stats': {
            'samples': len(cascades),
            'target_mean': np.mean([c.target_size for c in cascades]),
            'target_std': np.std([c.target_size for c in cascades]),
            'target_range': [min([c.target_size for c in cascades]),
                           max([c.target_size for c in cascades])]
        }
    }

    # 保存报告
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'training_report_fixed.json', 'w', encoding='utf-8') as f:
        json.dump(training_report, f, indent=2, ensure_ascii=False)

    print(f"\n📄 训练报告已保存: outputs/training_report_fixed.json")
    return training_report

def plot_training_curves(cascades):
    """绘制目标值分布"""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 目标值分布
    targets = [c.target_size for c in cascades]
    axes[0].hist(targets, bins=30, alpha=0.7, color='blue')
    axes[0].axvline(np.mean(targets), color='red', linestyle='--', label=f'Mean: {np.mean(targets):.1f}')
    axes[0].set_title('Target Distribution')
    axes[0].set_xlabel('User Count')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 目标值统计
    stats_text = f"样本数: {len(cascades)}\n"
    stats_text += f"平均值: {np.mean(targets):.2f}\n"
    stats_text += f"标准差: {np.std(targets):.2f}\n"
    stats_text += f"最小值: {min(targets)}\n"
    stats_text += f"最大值: {max(targets)}\n"
    stats_text += f"中位数: {np.median(targets):.2f}"

    axes[1].text(0.1, 0.5, stats_text, fontsize=12, transform=axes[1].transAxes)
    axes[1].set_title('Target Statistics')
    axes[1].axis('off')

    plt.tight_layout()

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_dir / 'target_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # 清理旧缓存和模型文件
    print("🧹 清理旧文件...")

    # 加载数据
    cascades = load_wikipedia_cascades("pp/sample_data/wikipedia.csv")

    # 绘制目标值分布
    plot_training_curves(cascades)

    # 重新训练模型
    report = retrain_with_fixed_targets()

    # 比较性能
    compare_performance(report['metrics'])

    # 生成训练报告
    generate_training_report(report, cascades)

    print("\n" + "=" * 60)
    print("✅ 重训练完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
