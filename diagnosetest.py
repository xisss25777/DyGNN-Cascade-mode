# D:\Users\HP\Documents\GitHub\DyGNN-Cascade-mode\diagnosetest.py
import numpy as np
import pandas as pd
import json
import sys
import os
import random
import math
import torch
import torch.nn as nn
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cascade_model.data import load_cascades_from_csv
from cascade_model.dataset_profiles import load_dataset_and_config
from cascade_model.dgnn import build_dgnn_dataset, DynamicCascadeGNN


def check_data_leakage(train_data, test_data):
    """检查数据泄露"""
    print("=" * 60)
    print("🔍 1. 数据泄露检查")
    print("=" * 60)

    train_ids = set([sample.cascade_id for sample in train_data])
    test_ids = set([sample.cascade_id for sample in test_data])

    overlap = train_ids.intersection(test_ids)
    print(f"训练集样本数: {len(train_ids)}")
    print(f"测试集样本数: {len(test_ids)}")
    print(f"重叠样本数: {len(overlap)}")

    if len(overlap) > 0:
        print("❌ 数据泄露！训练集和测试集有重叠样本")
        return False
    else:
        print("✅ 数据分割正确，无数据泄露")
        return True


def check_feature_correlation(dataset):
    """检查特征与标签的相关性"""
    print("\n" + "=" * 60)
    print("🔍 2. 特征质量检查")
    print("=" * 60)
    
    # 提取特征和标签
    features = []
    labels = []
    
    for sample in dataset:
        # 只使用第一个时间片的特征，避免不同时间片节点数量不一致的问题
        if len(sample.snapshots) > 0:
            first_snapshot = sample.snapshots[0]
            features.append(first_snapshot.node_features.numpy().flatten())
            labels.append(sample.target)
    
    # 处理不同长度的特征向量
    max_len = max(len(f) for f in features)
    features_padded = []
    for f in features:
        if len(f) < max_len:
            # 用0填充到最大长度
            padded = np.pad(f, (0, max_len - len(f)), mode='constant')
            features_padded.append(padded)
        else:
            features_padded.append(f)
    
    features = np.array(features_padded)
    labels = np.array(labels)
    
    print(f"特征矩阵形状: {features.shape}")
    print(f"标签数组形状: {labels.shape}")
    
    # 计算特征与标签的相关性
    correlations = []
    for i in range(features.shape[1]):
        try:
            corr = np.corrcoef(features[:, i], labels)[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))
        except:
            pass
    
    if correlations:
        avg_corr = np.mean(correlations)
        max_corr = np.max(correlations)
        print(f"特征-标签平均相关性: {avg_corr:.4f}")
        print(f"特征-标签最大相关性: {max_corr:.4f}")
        
        if avg_corr < 0.1:
            print("⚠️ 特征与标签相关性过低，可能需要改进特征工程")
            return False
        elif avg_corr < 0.2:
            print("⚠️ 特征与标签相关性一般，建议增强特征")
            return True
        else:
            print("✅ 特征与标签相关性良好")
            return True
    else:
        print("❌ 无法计算特征相关性")
        return False


def analyze_label_distribution(dataset):
    """分析标签分布"""
    print("\n" + "=" * 60)
    print("🔍 3. 标签分布分析")
    print("=" * 60)

    labels = [sample.target for sample in dataset]
    labels_array = np.array(labels)

    print(f"标签统计:")
    print(f"  均值: {np.mean(labels_array):.1f}")
    print(f"  中位数: {np.median(labels_array):.1f}")
    print(f"  标准差: {np.std(labels_array):.1f}")
    print(f"  最小值/最大值: {np.min(labels_array):.1f}/{np.max(labels_array):.1f}")
    # 使用numpy计算偏度和峰度
    skewness = np.mean(((labels_array - np.mean(labels_array)) / np.std(labels_array)) ** 3)
    kurtosis = np.mean(((labels_array - np.mean(labels_array)) / np.std(labels_array)) ** 4) - 3
    print(f"  偏度: {skewness:.3f}")
    print(f"  峰度: {kurtosis:.3f}")

    # 检查长尾分布
    mean_median_ratio = np.mean(labels_array) / np.median(labels_array)
    print(f"  均值/中位数比: {mean_median_ratio:.2f}")

    if mean_median_ratio > 2.0:
        print("⚠️ 强长尾分布，建议使用对数变换或分位数损失")
        return False
    elif mean_median_ratio > 1.5:
        print("⚠️ 中等长尾分布，建议调整损失函数")
        return True
    else:
        print("✅ 标签分布相对均衡")
        return True


def analyze_prediction_patterns(results_file):
    """分析预测模式"""
    print("\n" + "=" * 60)
    print("🔍 4. 预测模式分析")
    print("=" * 60)

    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except:
        print(f"❌ 无法读取结果文件: {results_file}")
        return False

    # 提取预测值和真实值
    test_reports = data.get('test_reports', [])
    if not test_reports:
        print("❌ 没有测试报告数据")
        return False

    predictions = []
    actuals = []

    for report in test_reports:
        predictions.append(report['prediction'])
        # 尝试获取真实值
        if 'actual' in report:
            actuals.append(report['actual'])
        else:
            # 如果没有真实值，使用预测值作为占位
            actuals.append(report['prediction'])

    pred_array = np.array(predictions)
    actual_array = np.array(actuals)

    print(f"预测值统计:")
    print(f"  均值: {pred_array.mean():.2f}")
    print(f"  中位数: {np.median(pred_array):.2f}")
    print(f"  标准差: {pred_array.std():.2f}")
    print(f"  最小值/最大值: {pred_array.min():.2f}/{pred_array.max():.2f}")

    # 计算变异系数
    pred_cv = pred_array.std() / pred_array.mean() * 100
    print(f"  变异系数: {pred_cv:.1f}%")

    # 检查预测值是否过于集中
    if pred_cv < 5.0:
        print("❌ 预测值变异不足，模型输出过于集中")
        print("   这表明模型可能没有学到样本间的差异")
        return False
    elif pred_cv < 10.0:
        print("⚠️ 预测值变异较低，模型区分能力有限")
        return True
    else:
        print("✅ 预测值变异合理")
        return True


def simple_baseline_test(dataset):
    """简单基准测试"""
    print("\n" + "=" * 60)
    print("🔍 5. 简单基准测试")
    print("=" * 60)
    
    # 提取特征和标签
    features = []
    labels = []
    
    for sample in dataset:
        # 只使用第一个时间片的特征，避免不同时间片节点数量不一致的问题
        if len(sample.snapshots) > 0:
            first_snapshot = sample.snapshots[0]
            features.append(first_snapshot.node_features.numpy().flatten())
            labels.append(sample.target)
    
    # 处理不同长度的特征向量
    max_len = max(len(f) for f in features)
    features_padded = []
    for f in features:
        if len(f) < max_len:
            # 用0填充到最大长度
            padded = np.pad(f, (0, max_len - len(f)), mode='constant')
            features_padded.append(padded)
        else:
            features_padded.append(f)
    
    features = np.array(features_padded)
    labels = np.array(labels)
    
    print(f"特征矩阵形状: {features.shape}")
    print(f"标签数组形状: {labels.shape}")
    
    # 确保完全分离
    indices = list(range(len(features)))
    random.seed(42)
    random.shuffle(indices)
    split_at = int(len(indices) * 0.8)
    train_indices = indices[:split_at]
    test_indices = indices[split_at:]
    
    X_train = features[train_indices]
    X_test = features[test_indices]
    y_train = labels[train_indices]
    y_test = labels[test_indices]
    
    # 训练简单线性回归模型
    # 添加偏置项
    X_train_with_bias = np.column_stack([X_train, np.ones(X_train.shape[0])])
    X_test_with_bias = np.column_stack([X_test, np.ones(X_test.shape[0])])
    
    # 使用最小二乘法求解
    try:
        weights = np.linalg.lstsq(X_train_with_bias, y_train, rcond=None)[0]
        y_pred = X_test_with_bias @ weights
        
        # 计算MAE
        mae = np.mean(np.abs(y_test - y_pred))
        print(f"简单线性模型MAE: {mae:.2f}")
        
        if mae < 70:
            print("✅ 特征有效，问题可能在DGNN模型")
            return True
        elif mae < 100:
            print("⚠️ 特征有一定效果，但需要改进")
            return True
        else:
            print("❌ 特征效果很差，需要改进特征工程")
            return False
    except Exception as e:
        print(f"❌ 线性回归训练失败: {e}")
        return False


def comprehensive_diagnosis(dataset_name="wikipedia", results_file="outputs/wikipedia_report_optimized_v5.json"):
    """全面诊断"""
    print("\n" + "=" * 60)
    print("🧪 模型性能瓶颈全面诊断")
    print("=" * 60)

    # 加载数据
    try:
        # 尝试不同的数据文件路径
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
            print(f"❌ 找不到数据文件，尝试的路径: {possible_paths}")
            return
        
        dataset_name, cascades, config = load_dataset_and_config(input_path, dataset_name)
        dataset = build_dgnn_dataset(
            cascades,
            observation_seconds=config.observation_seconds,
            slice_seconds=config.slice_seconds,
        )

        # 划分训练和测试集
        indices = list(range(len(dataset)))
        random.seed(config.random_seed)
        random.shuffle(indices)
        split_at = max(1, int(len(indices) * (1 - config.test_ratio)))
        train_ids = indices[:split_at]
        test_ids = indices[split_at:]

        train_data = [dataset[idx] for idx in train_ids]
        test_data = [dataset[idx] for idx in test_ids]

        print(f"数据集: {dataset_name}")
        print(f"总样本数: {len(dataset)}")
        print(f"训练集样本数: {len(train_data)}")
        print(f"测试集样本数: {len(test_data)}")

    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    # 运行各项检查
    results = {}

    # 1. 数据泄露检查
    results['data_leakage'] = check_data_leakage(train_data, test_data)

    # 2. 特征质量检查
    results['feature_quality'] = check_feature_correlation(dataset)

    # 3. 标签分布分析
    results['label_distribution'] = analyze_label_distribution(dataset)

    # 4. 预测模式分析
    results['prediction_patterns'] = analyze_prediction_patterns(results_file)

    # 5. 简单基准测试
    results['baseline_test'] = simple_baseline_test(dataset)

    # 总结诊断结果
    print("\n" + "=" * 60)
    print("📋 诊断总结")
    print("=" * 60)

    passed = sum(results.values())
    total = len(results)

    for check_name, result in results.items():
        status = "✅" if result else "❌"
        print(f"{status} {check_name}: {'通过' if result else '失败'}")

    print(f"\n通过检查: {passed}/{total}")

    if passed == total:
        print("🎉 所有检查通过！问题可能在模型架构或训练策略")
    elif passed >= total * 0.7:
        print("⚠️ 大部分检查通过，需要针对性改进")
    else:
        print("❌ 多项检查失败，需要全面改进")

    # 提供改进建议
    print("\n" + "=" * 60)
    print("💡 改进建议")
    print("=" * 60)

    if not results['data_leakage']:
        print("1. 修复数据泄露问题，确保训练/测试集完全分离")

    if not results['feature_quality']:
        print("2. 改进特征工程，添加更有意义的特征")

    if not results['label_distribution']:
        print("3. 调整损失函数，使用对数变换或分位数损失")

    if not results['prediction_patterns']:
        print("4. 改进模型架构，增强表达能力")

    if not results['baseline_test']:
        print("5. 重新设计特征提取流程")


def check_attention_mechanism(dataset_name="wikipedia"):
    """检查注意力机制是否正常工作"""
    print("\n" + "=" * 60)
    print("🔍 注意力机制诊断")
    print("=" * 60)
    
    try:
        # 加载数据
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
            print(f"❌ 找不到数据文件，尝试的路径: {possible_paths}")
            return False
        
        dataset_name, cascades, config = load_dataset_and_config(input_path, dataset_name)
        dataset = build_dgnn_dataset(
            cascades,
            observation_seconds=config.observation_seconds,
            slice_seconds=config.slice_seconds,
        )
        
        # 使用第一个样本测试
        sample = dataset[0]
        
        # 创建模型
        input_dim = sample.snapshots[0].node_features.shape[1]
        graph_dim = sample.snapshots[0].graph_features.shape[0]
        model = DynamicCascadeGNN(input_dim=input_dim, graph_dim=graph_dim)
        
        # 前向传播
        model.eval()
        with torch.no_grad():
            size_pred, growth_pred, attention_weights, channel_scores, spatial_masks, temporal_mask, edge_masks, node_masks = model(sample.snapshots)
        
        print(f"\n注意力权重分析:")
        print(f"  时间片数量: {len(attention_weights)}")
        print(f"  注意力权重: {attention_weights.tolist()}")
        print(f"  注意力权重统计:")
        print(f"    均值: {attention_weights.mean().item():.4f}")
        print(f"    标准差: {attention_weights.std().item():.4f}")
        print(f"    最小值: {attention_weights.min().item():.4f}")
        print(f"    最大值: {attention_weights.max().item():.4f}")
        print(f"    熵: {-torch.sum(attention_weights * torch.log(attention_weights + 1e-10)).item():.4f}")
        
        # 检查是否均匀分布
        expected_uniform = 1.0 / len(attention_weights)
        is_uniform = torch.allclose(attention_weights, torch.tensor([expected_uniform] * len(attention_weights)), atol=1e-3)
        
        if is_uniform:
            print(f"  ⚠️ 注意力权重过于均匀，可能失效！")
            print(f"     期望均匀值: {expected_uniform:.4f}")
            print(f"     实际值: {attention_weights.tolist()}")
            return False
        else:
            print(f"  ✅ 注意力权重分布正常")
            return True
            
    except Exception as e:
        print(f"❌ 注意力机制诊断失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_gradient_flow(dataset_name="wikipedia"):
    """检查梯度流向"""
    print("\n" + "=" * 60)
    print("🔍 梯度流向诊断")
    print("=" * 60)
    
    try:
        # 加载数据
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
            return False
        
        dataset_name, cascades, config = load_dataset_and_config(input_path, dataset_name)
        dataset = build_dgnn_dataset(
            cascades,
            observation_seconds=config.observation_seconds,
            slice_seconds=config.slice_seconds,
        )
        
        # 使用第一个样本测试
        sample = dataset[0]
        
        # 创建模型
        input_dim = sample.snapshots[0].node_features.shape[1]
        graph_dim = sample.snapshots[0].graph_features.shape[0]
        model = DynamicCascadeGNN(input_dim=input_dim, graph_dim=graph_dim)
        
        # 前向传播和反向传播
        model.train()
        size_pred, growth_pred, attention_weights, channel_scores, spatial_masks, temporal_mask, edge_masks, node_masks = model(sample.snapshots)
        
        # 计算损失
        target = torch.tensor([math.log1p(sample.target)], dtype=torch.float32)
        loss = nn.MSELoss()(size_pred.view(1), target)
        
        # 反向传播
        loss.backward()
        
        # 检查梯度
        print(f"\n梯度分析:")
        has_zero_grad = False
        has_small_grad = False
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if grad_norm < 1e-6:
                    has_zero_grad = True
                    print(f"  ⚠️ {name}: 梯度过小 ({grad_norm:.8f})")
                elif grad_norm < 1e-3:
                    has_small_grad = True
                    print(f"  ℹ️ {name}: 梯度较小 ({grad_norm:.6f})")
                else:
                    print(f"  ✅ {name}: 梯度正常 ({grad_norm:.4f})")
            else:
                print(f"  ❌ {name}: 无梯度")
                has_zero_grad = True
        
        if has_zero_grad:
            print(f"\n  ⚠️ 发现梯度消失问题！")
            return False
        elif has_small_grad:
            print(f"\n  ⚠️ 部分层梯度较小，可能存在梯度消失风险")
            return True
        else:
            print(f"\n  ✅ 梯度流向正常")
            return True
            
    except Exception as e:
        print(f"❌ 梯度流向诊断失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def analyze_attention_distribution(attention_weights):
    """分析注意力分布是否合理"""
    print("\n" + "=" * 60)
    print("🔍 注意力分布分析")
    print("=" * 60)
    
    weights_array = np.array(attention_weights)
    
    print(f"注意力权重统计:")
    print(f"  均值: {weights_array.mean():.4f}")
    print(f"  中位数: {np.median(weights_array):.4f}")
    print(f"  标准差: {weights_array.std():.4f}")
    print(f"  最小值: {weights_array.min():.4f}")
    print(f"  最大值: {weights_array.max():.4f}")
    print(f"  熵: {-np.sum(weights_array * np.log(weights_array + 1e-10)):.4f}")
    
    # 检查分布是否合理
    max_weight = weights_array.max()
    min_weight = weights_array.min()
    
    if max_weight > 0.9:
        print(f"  ⚠️ 注意力过于集中！最大权重={max_weight:.4f} > 0.9")
        print(f"     这意味着模型过度依赖单个时间片")
        return False
    elif min_weight < 0.01:
        print(f"  ⚠️ 注意力分布不均！最小权重={min_weight:.4f} < 0.01")
        print(f"     这意味着模型忽略了大部分时间片")
        return False
    elif weights_array.std() < 0.05:
        print(f"  ⚠️ 注意力分布过于平均！标准差={weights_array.std():.4f} < 0.05")
        print(f"     这意味着模型没有学到时间片间的差异")
        return False
    else:
        print(f"  ✅ 注意力分布合理")
        return True


def analyze_prediction_diversity(predictions):
    """分析预测值的多样性"""
    print("\n" + "=" * 60)
    print("🔍 预测值多样性分析")
    print("=" * 60)
    
    pred_array = np.array(predictions)
    
    print(f"预测值统计:")
    print(f"  均值: {pred_array.mean():.2f}")
    print(f"  中位数: {np.median(pred_array):.2f}")
    print(f"  标准差: {pred_array.std():.2f}")
    print(f"  最小值: {pred_array.min():.2f}")
    print(f"  最大值: {pred_array.max():.2f}")
    print(f"  变异系数: {pred_array.std() / pred_array.mean() * 100:.2f}%")
    
    # 检查预测值是否过于集中
    cv = pred_array.std() / pred_array.mean() * 100
    
    if cv < 5.0:
        print(f"  ⚠️ 预测值过于集中！变异系数={cv:.2f}% < 5%")
        print(f"     这意味着模型对不同级联的预测几乎相同")
        return False
    elif cv < 10.0:
        print(f"  ⚠️ 预测值多样性不足！变异系数={cv:.2f}% < 10%")
        print(f"     这意味着模型区分能力有限")
        return False
    else:
        print(f"  ✅ 预测值多样性合理")
        return True


def analyze_temporal_learning_pattern(attention_weights, predictions, targets):
    """分析时间学习模式"""
    print("\n" + "=" * 60)
    print("🔍 时间学习模式分析")
    print("=" * 60)
    
    # 分析注意力权重的时间分布
    weights_array = np.array(attention_weights)
    
    print(f"时间片重要性分析:")
    for i, weight in enumerate(attention_weights):
        importance = "高" if weight > 0.5 else "中" if weight > 0.1 else "低"
        print(f"  时间片{i+1}: 权重={weight:.6f}, 重要性={importance}")
    
    # 检查是否存在"偷懒"模式
    if weights_array[0] > 0.9:
        print(f"  ⚠️ 发现'偷懒'模式！模型过度依赖第一个时间片")
        print(f"     第一个时间片权重={weights_array[0]:.6f}，后续时间片被忽略")
        return False
    else:
        print(f"  ✅ 没有发现'偷懒'模式")
        return True


def comprehensive_attention_analysis(dataset_name="wikipedia", results_file="outputs/wikipedia_report_attention_final.json"):
    """全面的注意力机制分析"""
    print("\n" + "=" * 60)
    print("🧪 注意力机制全面分析")
    print("=" * 60)
    
    try:
        # 加载数据
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
        
        # 加载结果文件
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
        except:
            print(f"❌ 无法读取结果文件: {results_file}")
            return
        
        # 提取注意力权重
        test_reports = results.get('test_reports', [])
        if not test_reports:
            print("❌ 没有测试报告数据")
            return
        
        # 收集所有注意力权重
        all_attention_weights = []
        all_predictions = []
        all_targets = []
        
        for report in test_reports:
            all_attention_weights.extend([
                item['weight'] for item in report.get('top_attention_slices', [])
            ])
            all_predictions.append(report['prediction'])
            all_targets.append(report.get('actual', report['prediction']))  # 使用预测值作为占位
        
        if not all_attention_weights:
            print("❌ 没有注意力权重数据")
            return
        
        # 分析注意力分布
        attention_ok = analyze_attention_distribution(all_attention_weights)
        
        # 分析预测多样性
        prediction_ok = analyze_prediction_diversity(all_predictions)
        
        # 分析时间学习模式
        temporal_ok = analyze_temporal_learning_pattern(all_attention_weights[:12], all_predictions, all_targets)
        
        # 总结
        print("\n" + "=" * 60)
        print("📋 分析总结")
        print("=" * 60)
        
        passed = sum([attention_ok, prediction_ok, temporal_ok])
        total = 3
        
        print(f"通过检查: {passed}/{total}")
        
        if passed == total:
            print("🎉 所有检查通过！")
        elif passed >= total * 0.7:
            print("⚠️ 大部分检查通过，需要针对性改进")
        else:
            print("❌ 多项检查失败，需要全面改进")
        
        # 提供改进建议
        print("\n" + "=" * 60)
        print("💡 改进建议")
        print("=" * 60)
        
        if not attention_ok:
            print("1. 修复注意力机制计算")
            print("   - 添加时间衰减权重")
            print("   - 鼓励更均匀的注意力分布")
            print("   - 添加注意力多样性正则化")
        
        if not prediction_ok:
            print("2. 增强特征工程")
            print("   - 添加时间演化特征")
            print("   - 添加传播速度特征")
            print("   - 添加网络结构特征")
        
        if not temporal_ok:
            print("3. 调整损失函数")
            print("   - 添加注意力分布正则化")
            print("   - 添加预测多样性正则化")
            print("   - 防止模型'偷懒'")
        
        return attention_ok and prediction_ok and temporal_ok
        
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_attention_fix(dataset_name="wikipedia"):
    """验证注意力修复效果"""
    print("\n" + "=" * 60)
    print("🧪 注意力修复验证")
    print("=" * 60)
    
    try:
        # 加载数据
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
        
        dataset_name, cascades, config = load_dataset_and_config(input_path, dataset_name)
        dataset = build_dgnn_dataset(
            cascades,
            observation_seconds=config.observation_seconds,
            slice_seconds=config.slice_seconds,
        )
        
        # 使用前10个样本测试
        test_samples = dataset[:10]
        
        # 创建模型
        input_dim = test_samples[0].snapshots[0].node_features.shape[1]
        graph_dim = test_samples[0].snapshots[0].graph_features.shape[0]
        model = DynamicCascadeGNN(input_dim=input_dim, graph_dim=graph_dim)
        
        # 收集注意力权重和预测值
        all_attention_weights = []
        all_predictions = []
        
        model.eval()
        with torch.no_grad():
            for sample in test_samples:
                size_pred, _, attention_weights, _, _, _, _, _ = model(sample.snapshots)
                all_attention_weights.append(attention_weights.tolist())
                all_predictions.append(size_pred.item())
        
        # 计算统计量
        attention_array = np.array(all_attention_weights)
        pred_array = np.array(all_predictions)
        
        # 1. 检查注意力分布
        attn_std = attention_array.std(axis=0).mean()
        attn_entropy = -np.sum(attention_array * np.log(attention_array + 1e-10), axis=1).mean()
        attn_max = attention_array.max()
        attn_min = attention_array.min()
        
        print(f"\n注意力分布分析:")
        print(f"  标准差: {attn_std:.4f}")
        print(f"  熵: {attn_entropy:.4f}")
        print(f"  最大值: {attn_max:.4f}")
        print(f"  最小值: {attn_min:.4f}")
        
        if attn_std < 0.4 and attn_std > 0.1:
            print(f"  ✅ 注意力分布合理")
        else:
            print(f"  ⚠️  注意力分布仍需调整 (标准差={attn_std:.4f})")
        
        if attn_max < 0.9:
            print(f"  ✅ 没有过度集中")
        else:
            print(f"  ⚠️  仍然过度集中 (最大值={attn_max:.4f})")
        
        # 2. 检查预测值多样性
        pred_std = pred_array.std()
        pred_mean = pred_array.mean()
        cv = (pred_std / pred_mean) * 100 if pred_mean > 0 else 0
        
        print(f"\n预测值多样性分析:")
        print(f"  标准差: {pred_std:.4f}")
        print(f"  变异系数: {cv:.2f}%")
        print(f"  预测值范围: [{pred_array.min():.4f}, {pred_array.max():.4f}]")
        
        if cv > 5.0:
            print(f"  ✅ 预测值有足够多样性")
        else:
            print(f"  ⚠️  预测值过于集中 (变异系数={cv:.2f}%)")
        
        # 3. 综合评估
        print(f"\n" + "=" * 60)
        print("📋 修复效果评估")
        print("=" * 60)
        
        attention_ok = (attn_std < 0.4 and attn_std > 0.1) and (attn_max < 0.9)
        prediction_ok = cv > 5.0
        
        if attention_ok and prediction_ok:
            print("🎉 修复成功！")
        elif attention_ok or prediction_ok:
            print("⚠️  部分修复成功，需要进一步调整")
        else:
            print("❌ 修复效果不理想，需要重新设计")
        
        return attention_ok and prediction_ok
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_fix_effect(dataset_name="wikipedia"):
    """验证修复效果"""
    print("\n" + "=" * 60)
    print("🧪 修复效果验证")
    print("=" * 60)
    
    try:
        # 加载数据
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
        
        dataset_name, cascades, config = load_dataset_and_config(input_path, dataset_name)
        dataset = build_dgnn_dataset(
            cascades,
            observation_seconds=config.observation_seconds,
            slice_seconds=config.slice_seconds,
        )
        
        # 使用前20个样本测试
        test_samples = dataset[:20]
        
        # 创建模型
        input_dim = test_samples[0].snapshots[0].node_features.shape[1]
        graph_dim = test_samples[0].snapshots[0].graph_features.shape[0]
        model = DynamicCascadeGNN(input_dim=input_dim, graph_dim=graph_dim)
        
        # 收集注意力权重和预测值
        all_attention_weights = []
        all_predictions = []
        
        model.eval()
        with torch.no_grad():
            for sample in test_samples:
                size_pred, _, attention_weights, _, _, _, _, _ = model(sample.snapshots)
                all_attention_weights.append(attention_weights.tolist())
                all_predictions.append(size_pred.item())
        
        # 计算统计量
        attention_array = np.array(all_attention_weights)
        pred_array = np.array(all_predictions)
        
        # 1. 预测值检查
        pred_mean = pred_array.mean()
        pred_std = pred_array.std()
        pred_min = pred_array.min()
        pred_max = pred_array.max()
        cv = (pred_std / pred_mean) * 100 if pred_mean > 0 else 0
        
        print(f"\n预测值统计:")
        print(f"  均值: {pred_mean:.4f}")
        print(f"  标准差: {pred_std:.4f}")
        print(f"  变异系数: {cv:.2f}%")
        print(f"  范围: [{pred_min:.4f}, {pred_max:.4f}]")
        
        pred_ok = True
        if pred_mean < 1.0:
            print(f"  ⚠️  预测均值过小，可能输出层仍需调整")
            pred_ok = False
        if pred_std < 5.0:
            print(f"  ⚠️  预测值多样性不足")
            pred_ok = False
        if pred_min < 0:
            print(f"  ⚠️  预测值仍有负数")
            pred_ok = False
        else:
            print(f"  ✅ 预测值为正数")
        
        # 2. 注意力检查
        attn_std = attention_array.std(axis=0).mean()
        attn_max = attention_array.max()
        attn_min = attention_array.min()
        max_min_ratio = attn_max / attn_min if attn_min > 0 else 0
        
        print(f"\n注意力统计:")
        print(f"  标准差: {attn_std:.4f}")
        print(f"  范围: [{attn_min:.4f}, {attn_max:.4f}]")
        print(f"  最大/最小比: {max_min_ratio:.2f}")
        
        attn_ok = True
        if attn_std < 0.05:
            print(f"  ⚠️  注意力分布仍然太平均")
            attn_ok = False
        if max_min_ratio < 2.0:
            print(f"  ⚠️  时间片区分度不足")
            attn_ok = False
        else:
            print(f"  ✅ 注意力分布有合理区分度")
        
        # 3. 综合评估
        print(f"\n" + "=" * 60)
        print("📋 修复效果评估")
        print("=" * 60)
        
        if pred_ok and attn_ok:
            print("🎉 修复成功！")
        elif pred_ok or attn_ok:
            print("⚠️  部分修复成功，需要进一步调整")
        else:
            print("❌ 修复效果不理想，需要重新设计")
        
        return pred_ok and attn_ok
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="模型性能诊断工具")
    parser.add_argument("--dataset", default="wikipedia", help="数据集名称")
    parser.add_argument("--results", default="outputs/wikipedia_report_optimized_v5.json", help="结果文件路径")
    parser.add_argument("--check-attention", action="store_true", help="检查注意力机制")
    parser.add_argument("--check-gradient", action="store_true", help="检查梯度流向")
    parser.add_argument("--comprehensive-attention", action="store_true", help="全面的注意力分析")
    parser.add_argument("--validate-fix", action="store_true", help="验证修复效果")
    parser.add_argument("--validate-fix-effect", action="store_true", help="验证新修复效果")

    args = parser.parse_args()

    if args.check_attention:
        check_attention_mechanism(args.dataset)
    elif args.check_gradient:
        check_gradient_flow(args.dataset)
    elif args.comprehensive_attention:
        comprehensive_attention_analysis(args.dataset, args.results)
    elif args.validate_fix:
        validate_attention_fix(args.dataset)
    elif args.validate_fix_effect:
        validate_fix_effect(args.dataset)
    else:
        comprehensive_diagnosis(args.dataset, args.results)