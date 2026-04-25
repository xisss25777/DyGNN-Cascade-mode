import os
import json
import pandas as pd

def diagnose():
    """诊断预测文件问题"""
    
    print("🔍 预测文件诊断")
    print("="*60)
    
    # 1. 检查文件存在
    files_to_check = [
        'outputs/prediction.txt',
        'outputs/wikipedia_predictions.json',
        'pp/outputs/wikipedia_report_final.json',
        'outputs/wikipedia_report_final.json'
    ]
    
    print("1. 检查预测文件:")
    for file in files_to_check:
        exists = os.path.exists(file)
        print(f"   {file}: {'✅ 存在' if exists else '❌ 不存在'}")
        if exists:
            try:
                size = os.path.getsize(file)
                print(f"     大小: {size} 字节")
            except:
                pass
    
    # 2. 检查 JSON 文件内容
    json_file = 'pp/outputs/wikipedia_report_final.json'
    if os.path.exists(json_file):
        print(f"\n2. 检查 {json_file} 内容:")
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 查找预测结果
        prediction_keys = []
        for key in data.keys():
            if 'pred' in key.lower() or 'predict' in key.lower():
                prediction_keys.append(key)
        
        if prediction_keys:
            print(f"   找到预测相关键: {prediction_keys}")
            for key in prediction_keys[:3]:
                value = data[key]
                if isinstance(value, list):
                    print(f"     {key}: 列表长度={len(value)}")
                else:
                    print(f"     {key}: 类型={type(value)}")
        else:
            print("   未找到预测相关的键")
        
        # 检查 test_reports
        if 'test_reports' in data:
            print(f"\n3. 检查 test_reports 内容:")
            test_reports = data['test_reports']
            print(f"   test_reports 长度: {len(test_reports)}")
            
            if test_reports:
                # 显示前3个报告
                for i, report in enumerate(test_reports[:3]):
                    print(f"   报告 {i+1}:")
                    print(f"     cascade_id: {report.get('cascade_id')}")
                    print(f"     prediction: {report.get('prediction')}")
                    print(f"     包含的键: {list(report.keys())}")
    
    # 4. 检查原始数据
    print(f"\n4. 检查原始数据格式:")
    try:
        df = pd.read_csv('pp/sample_data/wikipedia.csv')
        print(f"   行数: {len(df)}")
        print(f"   列: {list(df.columns)}")
        
        # 查看级联ID
        if 'item_id' in df.columns:
            unique_ids = df['item_id'].unique()[:5]
            print(f"   前5个 item_id: {unique_ids}")
            print(f"   item_id 类型: {type(unique_ids[0])}")
        
        if 'user_id' in df.columns:
            unique_user_ids = df['user_id'].unique()[:5]
            print(f"   前5个 user_id: {unique_user_ids}")
            print(f"   user_id 类型: {type(unique_user_ids[0])}")
        
        # 检查数据分布
        if 'item_id' in df.columns:
            item_counts = df['item_id'].value_counts()
            print(f"\n5. 数据分布:")
            print(f"   唯一 item_id 数量: {len(item_counts)}")
            print(f"   前5个 item_id 的事件数: {item_counts.head().to_dict()}")
    except Exception as e:
        print(f"   读取数据失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    diagnose()
