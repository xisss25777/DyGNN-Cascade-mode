import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path

def load_model():
    """
    加载训练好的模型
    
    Returns:
        torch.nn.Module: 加载的模型
    """
    print("🔍 加载模型")
    print("=" * 60)
    
    # 查找模型文件
    model_dir = Path('../pp/cache')
    model_files = list(model_dir.glob('*.pth'))
    
    if not model_files:
        print("❌ 未找到模型文件")
        return None
    
    # 加载最新的模型
    model_file = sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    print(f"加载模型文件: {model_file}")
    
    try:
        # 尝试加载模型
        model = torch.load(model_file)
        print("✅ 模型加载成功")
        return model
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None

def analyze_model_output(model):
    """
    分析模型的输出层
    
    Args:
        model: 加载的模型
    """
    print("\n🔍 分析模型输出层")
    print("=" * 60)
    
    if not model:
        return
    
    # 打印模型结构
    print("模型结构:")
    print(model)
    
    # 检查输出层
    print("\n输出层分析:")
    if hasattr(model, 'output_layer'):
        print(f"输出层: {model.output_layer}")
    elif hasattr(model, 'fc'):
        print(f"输出层: {model.fc}")
    elif hasattr(model, 'final_layer'):
        print(f"输出层: {model.final_layer}")
    else:
        # 尝试获取模型的最后一层
        layers = list(model.children())
        if layers:
            last_layer = layers[-1]
            print(f"最后一层: {last_layer}")
        else:
            print("❌ 无法识别输出层")

def load_original_cascades():
    """
    加载原始级联数据
    
    Returns:
        dict: 级联 ID 到真实值的映射
    """
    print("\n🔍 加载原始级联数据")
    print("=" * 60)
    
    # 使用相对路径
    data_path = '../pp/sample_data/wikipedia.csv'
    
    try:
        print(f"读取数据文件: {data_path}")
        df = pd.read_csv(data_path)
        print(f"数据形状: {df.shape}")
        print(f"数据列: {list(df.columns)}")
    except Exception as e:
        print(f"读取数据失败: {e}")
        return {}
    
    # 转换 item_id 为整数
    df['item_id'] = df['item_id'].astype(int)
    
    # 按 item_id 分组，计算每个级联的唯一用户数量
    print("\n计算每个级联的唯一用户数量...")
    cascades = df.groupby('item_id')['user_id'].nunique().reset_index()
    cascades.columns = ['cascade_id', 'unique_users']
    
    # 计算真实值 Y=log(1+N)
    cascades['true_value'] = np.log(1 + cascades['unique_users'])
    cascades['integer_true_value'] = cascades['true_value'].round().astype(int)
    
    print(f"共找到 {len(cascades)} 个原始级联")
    print(f"级联 ID: {list(cascades['cascade_id'])}")
    print(f"唯一用户数: {list(cascades['unique_users'])}")
    print(f"真实值 (log): {[round(v, 4) for v in cascades['true_value']]}")
    print(f"真实值 (整数): {list(cascades['integer_true_value'])}")
    
    # 转换为字典
    original_cascades = {}
    for _, row in cascades.iterrows():
        cascade_id = str(int(row['cascade_id']))
        original_cascades[cascade_id] = {
            'unique_users': row['unique_users'],
            'true_value': row['true_value'],
            'integer_true_value': row['integer_true_value']
        }
    
    print("\n" + "=" * 60)
    return original_cascades

def load_predictions():
    """
    加载模型预测结果
    
    Returns:
        dict: 预测结果字典
    """
    print("\n🔍 加载模型预测结果")
    print("=" * 60)
    
    # 使用相对路径
    result_file = '../pp/outputs/wikipedia_report_final.json'
    
    try:
        print(f"读取预测文件: {result_file}")
        with open(result_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
    except Exception as e:
        print(f"读取预测文件失败: {e}")
        return {}
    
    # 提取预测结果
    predictions = {}
    if 'test_reports' in results:
        for item in results['test_reports']:
            cascade_id = item.get('cascade_id')
            prediction = item.get('prediction')
            if cascade_id and prediction is not None:
                predictions[cascade_id] = prediction
    
    print(f"共找到 {len(predictions)} 个预测结果")
    print(f"预测值范围: [{min(predictions.values()):.2f}, {max(predictions.values()):.2f}]")
    
    # 显示前几个预测值
    sample_preds = list(predictions.items())[:5]
    print(f"预测值示例: {sample_preds}")
    
    print("\n" + "=" * 60)
    return predictions

def analyze_prediction_format(original_cascades, predictions):
    """
    分析预测值格式与真实值的匹配情况
    
    Args:
        original_cascades: 原始级联字典
        predictions: 预测结果字典
    """
    print("\n🔍 分析预测值格式")
    print("=" * 60)
    
    # 分析真实值分布
    true_values = [data['true_value'] for data in original_cascades.values()]
    print(f"真实值分布:")
    print(f"  平均值: {np.mean(true_values):.2f}")
    print(f"  范围: [{min(true_values):.2f}, {max(true_values):.2f}]")
    
    # 分析预测值分布
    pred_values = list(predictions.values())
    print(f"\n预测值分布:")
    print(f"  平均值: {np.mean(pred_values):.2f}")
    print(f"  范围: [{min(pred_values):.2f}, {max(pred_values):.2f}]")
    
    # 检查预测值是否需要转换
    print("\n检查预测值转换:")
    print("  原始预测值:")
    print(f"    平均值: {np.mean(pred_values):.2f}")
    print(f"    范围: [{min(pred_values):.2f}, {max(pred_values):.2f}]")
    
    # 尝试 log(1+pred) 转换
    log_preds = [np.log(1 + p) for p in pred_values]
    print("  log(1+pred) 转换后:")
    print(f"    平均值: {np.mean(log_preds):.2f}")
    print(f"    范围: [{min(log_preds):.2f}, {max(log_preds):.2f}]")
    
    # 尝试直接取对数转换
    log_only_preds = [np.log(p) if p > 0 else 0 for p in pred_values]
    print("  log(pred) 转换后:")
    print(f"    平均值: {np.mean(log_only_preds):.2f}")
    print(f"    范围: [{min(log_only_preds):.2f}, {max(log_only_preds):.2f}]")
    
    # 比较与真实值的接近程度
    print("\n与真实值的接近程度:")
    true_mean = np.mean(true_values)
    original_diff = abs(np.mean(pred_values) - true_mean)
    log1p_diff = abs(np.mean(log_preds) - true_mean)
    log_diff = abs(np.mean(log_only_preds) - true_mean)
    
    print(f"  原始预测值与真实值差异: {original_diff:.2f}")
    print(f"  log(1+pred) 与真实值差异: {log1p_diff:.2f}")
    print(f"  log(pred) 与真实值差异: {log_diff:.2f}")
    
    # 确定最佳转换
    min_diff = min(original_diff, log1p_diff, log_diff)
    if min_diff == original_diff:
        print("\n✅ 原始预测值与真实值最接近，无需转换")
    elif min_diff == log1p_diff:
        print("\n✅ log(1+pred) 转换后与真实值最接近")
    else:
        print("\n✅ log(pred) 转换后与真实值最接近")
    
    print("\n" + "=" * 60)

def main():
    """
    主函数
    """
    print("🚀 开始分析模型输出")
    print("=" * 80)
    
    # 加载模型
    model = load_model()
    
    # 分析模型输出层
    analyze_model_output(model)
    
    # 加载数据
    original_cascades = load_original_cascades()
    predictions = load_predictions()
    
    if original_cascades and predictions:
        # 分析预测值格式
        analyze_prediction_format(original_cascades, predictions)
    
    print("=" * 80)
    print("分析完成!")

if __name__ == "__main__":
    main()
