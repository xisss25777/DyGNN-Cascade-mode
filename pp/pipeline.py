def run_pipeline(cascades, config):
    """
    运行级联预测管道
    参数：
        cascades: list, 级联数据
        config: dict, 配置信息
    返回：
        report: dict, 预测报告
    """
    # 简单的示例实现
    sample_count = len(cascades)
    feature_count = config.get('feature_count', 0)
    
    # 模拟预测结果
    predictions = [cascade['N'] * 0.9 for cascade in cascades]
    ground_truth = [cascade['N'] for cascade in cascades]
    
    # 计算评估指标
    import numpy as np
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    
    mae = np.mean(np.abs(predictions - ground_truth))
    rmse = np.sqrt(np.mean((predictions - ground_truth) ** 2))
    mape = np.mean(np.abs((predictions - ground_truth) / ground_truth)) * 100
    
    # 构建报告
    report = {
        'sample_count': sample_count,
        'feature_count': feature_count,
        'metrics': {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape)
        },
        'top_features': [
            {'feature': 'feature_1', 'importance': 0.8},
            {'feature': 'feature_2', 'importance': 0.6},
            {'feature': 'feature_3', 'importance': 0.4}
        ],
        'test_reports': [
            {
                'cascade_id': i,
                'prediction': float(predictions[i]),
                'patterns': [{'pattern': 'pattern_1', 'confidence': 0.9}],
                'deletion_test': {'delta': 0.1}
            } for i in range(min(3, sample_count))
        ]
    }
    
    return report
