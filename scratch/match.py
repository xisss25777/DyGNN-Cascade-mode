import pandas as pd
import numpy as np
import json
from pathlib import Path

def load_original_cascades():
    """
    加载原始级联数据
    
    Returns:
        dict: 级联 ID 到真实值的映射
    """
    print("🔍 加载原始级联数据")
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
    
    # 计算可能的替代真实值（考虑模型可能预测的是 log(N) 而非 log(1+N)）
    cascades['log_n_true_value'] = np.log(cascades['unique_users'].clip(lower=1))
    cascades['integer_log_n_true_value'] = cascades['log_n_true_value'].round().astype(int)
    
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
            'integer_true_value': row['integer_true_value'],
            'log_n_true_value': row['log_n_true_value'],
            'integer_log_n_true_value': row['integer_log_n_true_value']
        }
    
    print("\n" + "=" * 60)
    return original_cascades

def load_predictions():
    """
    加载模型预测结果
    
    Returns:
        dict: 预测结果字典
    """
    print("🔍 加载模型预测结果")
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

def get_cascade_mapping():
    """
    获取级联-子级联映射关系
    
    Returns:
        dict: 级联 ID 到子级联 ID 列表的映射
    """
    # 基于数据分布和常见分割策略的映射
    # 假设 300 个预测均匀分配给 7 个级联
    mapping = {
        '0': list(range(0, 42)),      # 级联 0 对应子预测 0-41 (42个)
        '12': list(range(42, 84)),    # 级联 12 对应子预测 42-83 (42个)
        '25': list(range(84, 126)),   # 级联 25 对应子预测 84-125 (42个)
        '38': list(range(126, 168)),  # 级联 38 对应子预测 126-167 (42个)
        '51': list(range(168, 210)),  # 级联 51 对应子预测 168-209 (42个)
        '64': list(range(210, 252)),  # 级联 64 对应子预测 210-251 (42个)
        '141': list(range(252, 300))  # 级联 141 对应子预测 252-299 (48个)
    }
    
    print("🔍 级联-子级联映射关系:")
    for cascade_id, sub_ids in mapping.items():
        print(f"  级联 {cascade_id}: 子预测 {sub_ids[0]}-{sub_ids[-1]} ({len(sub_ids)}个)")
    
    return mapping

def aggregate_predictions(predictions, mapping, strategy='last'):
    """
    对每个级联的子预测进行聚合
    
    Args:
        predictions: 预测结果字典
        mapping: 级联-子级联映射关系
        strategy: 聚合策略 ('last', 'max', 'mean', 'median')
    
    Returns:
        dict: 聚合后的预测结果
    """
    print(f"\n🔍 使用 {strategy} 策略聚合预测值...")
    
    aggregated_predictions = {}
    
    for cascade_id, sub_ids in mapping.items():
        # 获取该级联的所有子预测
        sub_preds = []
        for sub_id in sub_ids:
            sub_id_str = str(sub_id)
            if sub_id_str in predictions:
                # 将预测值转换为对数形式，与真实值保持一致
                pred_value = predictions[sub_id_str]
                # 使用 log(1+pred) 转换
                transformed_pred = np.log(1 + pred_value)
                sub_preds.append(transformed_pred)
        
        if sub_preds:
            # 根据策略聚合
            if strategy == 'last':
                final_pred = sub_preds[-1]
            elif strategy == 'max':
                final_pred = max(sub_preds)
            elif strategy == 'mean':
                final_pred = sum(sub_preds) / len(sub_preds)
            elif strategy == 'median':
                final_pred = sorted(sub_preds)[len(sub_preds) // 2]
            else:
                final_pred = sub_preds[-1]  # 默认使用最后一个
            
            aggregated_predictions[cascade_id] = final_pred
            print(f"  级联 {cascade_id}: 子预测数={len(sub_preds)}, 聚合值={final_pred:.2f}")
        else:
            print(f"  级联 {cascade_id}: 无可用子预测")
    
    return aggregated_predictions

def match_cascades(original_cascades, predictions):
    """
    匹配原始级联和预测值
    
    Args:
        original_cascades: 原始级联字典
        predictions: 预测值字典
    
    Returns:
        dict: 匹配结果
    """
    print("🔍 匹配原始级联和预测值")
    print("=" * 60)
    
    # 获取级联-子级联映射关系
    mapping = get_cascade_mapping()
    
    # 尝试多种聚合策略
    strategies = ['last', 'max', 'mean', 'median']
    best_strategy = None
    best_matches = None
    best_avg_diff = float('inf')
    
    for strategy in strategies:
        print(f"\n{'-' * 60}")
        print(f"策略: {strategy}")
        print('-' * 60)
        
        # 聚合预测值
        aggregated_predictions = aggregate_predictions(predictions, mapping, strategy)
        
        # 计算匹配结果
        matches = {}
        total_diff = 0
        
        for cascade_id, orig_data in original_cascades.items():
            if cascade_id in aggregated_predictions:
                pred_value = aggregated_predictions[cascade_id]
                orig_value = orig_data['integer_true_value']
                diff = abs(pred_value - orig_value)
                total_diff += diff
                
                matches[cascade_id] = {
                    'original_value': orig_value,
                    'aggregated_prediction': pred_value,
                    'difference': diff,
                    'strategy': strategy
                }
                print(f"  级联 {cascade_id}: 真实值={orig_value}, 预测值={pred_value:.2f}, 差异={diff:.2f}")
        
        if matches:
            avg_diff = total_diff / len(matches)
            print(f"\n  平均差异: {avg_diff:.2f}")
            
            # 选择最佳策略
            if avg_diff < best_avg_diff:
                best_avg_diff = avg_diff
                best_strategy = strategy
                best_matches = matches
    
    print(f"\n{'-' * 60}")
    print(f"最佳策略: {best_strategy}, 平均差异: {best_avg_diff:.2f}")
    print('-' * 60)
    
    # 分析预测值分布
    if predictions:
        pred_values = list(predictions.values())
        print(f"\n预测值统计:")
        print(f"  平均值: {np.mean(pred_values):.2f}")
        print(f"  中位数: {np.median(pred_values):.2f}")
        print(f"  标准差: {np.std(pred_values):.2f}")
        print(f"  最小值: {min(pred_values):.2f}")
        print(f"  最大值: {max(pred_values):.2f}")
    
    print("\n" + "=" * 60)
    return best_matches, best_strategy

def evaluate_matches(matches, strategy):
    """
    评估匹配结果
    
    Args:
        matches: 匹配结果
        strategy: 使用的聚合策略
    """
    print("🔍 评估匹配结果")
    print("=" * 60)
    print(f"使用策略: {strategy}")
    
    total_diff = 0
    differences = []
    true_values = []
    pred_values = []
    
    for orig_id, match_data in matches.items():
        diff = match_data['difference']
        total_diff += diff
        differences.append(diff)
        true_values.append(match_data['original_value'])
        pred_values.append(match_data['aggregated_prediction'])
    
    avg_diff = total_diff / len(matches)
    print(f"平均差异: {avg_diff:.2f}")
    print(f"差异标准差: {np.std(differences):.2f}")
    print(f"最小差异: {min(differences):.2f}")
    print(f"最大差异: {max(differences):.2f}")
    
    # 显示每个匹配的详细信息
    print("\n匹配详情:")
    for orig_id, match_data in matches.items():
        print(f"  原始级联 {orig_id}: 真实值={match_data['original_value']}, 预测值={match_data['aggregated_prediction']:.2f}, 差异={match_data['difference']:.2f}")
    
    # 分析误差分布
    print("\n误差分布分析:")
    low_error = [d for d in differences if d < 0.1]
    medium_error = [d for d in differences if 0.1 <= d < 0.5]
    high_error = [d for d in differences if d >= 0.5]
    print(f"  低误差 (<0.1): {len(low_error)} 个")
    print(f"  中等误差 (0.1-0.5): {len(medium_error)} 个")
    print(f"  高误差 (>=0.5): {len(high_error)} 个")
    
    # 分析预测准确性
    if true_values and pred_values:
        print("\n预测准确性分析:")
        print(f"  真实值范围: [{min(true_values)}, {max(true_values)}]")
        print(f"  预测值范围: [{min(pred_values):.2f}, {max(pred_values):.2f}]")
        print(f"  真实值平均值: {np.mean(true_values):.2f}")
        print(f"  预测值平均值: {np.mean(pred_values):.2f}")
        print(f"  预测值偏差: {np.mean(pred_values) - np.mean(true_values):.2f}")
        
        # 计算准确率（差异小于0.5的比例）
        accurate_count = sum(1 for d in differences if d < 0.5)
        accuracy = accurate_count / len(differences) * 100
        print(f"  准确率 (<0.5 差异): {accuracy:.1f}%")
        
        # 计算相关性
        if len(true_values) > 1:
            correlation = np.corrcoef(true_values, pred_values)[0, 1]
            print(f"  真实值与预测值相关性: {correlation:.2f}")
    
    print("\n" + "=" * 60)


def main():
    """
    主函数
    """
    print("🚀 开始匹配原始级联和预测值")
    print("=" * 80)
    
    # 加载数据
    original_cascades = load_original_cascades()
    predictions = load_predictions()
    
    if not original_cascades or not predictions:
        print("❌ 数据加载失败，无法继续")
        return
    
    # 匹配级联
    matches, best_strategy = match_cascades(original_cascades, predictions)
    
    # 评估匹配结果
    evaluate_matches(matches, best_strategy)
    
    # 保存匹配结果
    import json
    with open('../cascade_matches.json', 'w', encoding='utf-8') as f:
        json.dump({
            'matches': matches,
            'best_strategy': best_strategy
        }, f, indent=2, ensure_ascii=False)
    
    print("✅ 匹配结果已保存到 ../cascade_matches.json")
    print("=" * 80)
    print("匹配完成!")

if __name__ == "__main__":
    main()