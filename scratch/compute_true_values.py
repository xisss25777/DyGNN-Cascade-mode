import pandas as pd
import numpy as np

def compute_true_values(use_integer=False):
    """
    根据 Cascade 数据计算每个级联的真实值 Y=log(1+N)
    
    Args:
        use_integer: 是否返回整数真实值
    
    Returns:
        dict: 级联 ID 到真实值的映射，格式与预测结果相同
    """
    print("🔍 开始计算级联真实值")
    print("=" * 60)
    
    # 读取原始数据
    try:
        # 使用绝对路径，确保从任何位置调用都能正确找到文件
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(project_root, 'pp', 'sample_data', 'wikipedia.csv')
        print(f"读取数据文件: {data_path}")
        df = pd.read_csv(data_path)
        print(f"数据形状: {df.shape}")
        print(f"数据列: {list(df.columns)}")
    except Exception as e:
        print(f"读取数据失败: {e}")
        return {}
    
    # 按 item_id 分组，计算每个级联的唯一用户数量
    print("\n计算每个级联的唯一用户数量...")
    # 转换 item_id 为整数，确保与预测值的 ID 格式匹配
    df['item_id'] = df['item_id'].astype(int)
    cascades = df.groupby('item_id')['user_id'].nunique().reset_index()
    cascades.columns = ['cascade_id', 'unique_users']
    
    print(f"共找到 {len(cascades)} 个级联")
    print(f"唯一用户数范围: [{cascades['unique_users'].min()}, {cascades['unique_users'].max()}]")
    print(f"级联 ID 范围: [{cascades['cascade_id'].min()}, {cascades['cascade_id'].max()}]")
    print(f"级联 ID 类型: {type(cascades['cascade_id'].iloc[0])}")
    
    # 计算真实值 Y=log(1+N)
    print("\n计算真实值 Y=log(1+N)...")
    cascades['true_value'] = np.log(1 + cascades['unique_users'])
    
    # 如果需要整数真实值
    if use_integer:
        print("\n转换真实值为整数...")
        cascades['true_value'] = cascades['true_value'].round().astype(int)
    
    # 转换为字典格式，与预测结果格式一致
    true_values = {}
    for _, row in cascades.iterrows():
        # 将级联 ID 转换为字符串，与预测值的 ID 格式一致
        cascade_id = str(int(row['cascade_id']))
        true_value = row['true_value']
        true_values[cascade_id] = true_value
    
    # 显示前几个结果
    print("\n前5个级联的真实值:")
    sample = list(true_values.items())[:5]
    for cascade_id, value in sample:
        if use_integer:
            print(f"  级联 {cascade_id}: Y={value}")
        else:
            print(f"  级联 {cascade_id}: Y={value:.4f}")
    
    print("\n" + "=" * 60)
    print("计算完成!")
    return true_values

def compute_integer_true_values():
    """
    计算整数真实值，与预测值格式匹配
    
    Returns:
        dict: 级联 ID 到整数真实值的映射
    """
    return compute_true_values(use_integer=True)

if __name__ == "__main__":
    true_values = compute_true_values()
    print(f"\n总级联数: {len(true_values)}")