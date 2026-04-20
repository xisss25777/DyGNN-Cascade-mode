# predict_explain_pipeline.py
import torch
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any

from cascade_model.dgnn import DynamicCascadeGNN, run_dgnn_pipeline
from cascade_model.data import load_wikipedia_cascades, Cascade
from cascade_model.config import PipelineConfig
from cascade_model.dynamic_graph import build_snapshots, Snapshot
from 掩码.spatio_temporal_mask import SpatioTemporalMask, extract_key_propagation_patterns_from_mask

def load_model_from_cache(cache_path: str) -> Dict[str, Any]:
    """从缓存加载模型信息"""
    with open(cache_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def predict_and_explain(cascade: Cascade, model: DynamicCascadeGNN, config: PipelineConfig) -> Dict[str, Any]:
    """预测并生成解释"""
    # 构建快照
    raw_snapshots = build_snapshots(cascade, config)
    if not raw_snapshots:
        return {"error": "无法构建快照"}
    
    # 转换为图数据
    from cascade_model.dgnn import snapshot_to_graph_data, GraphSnapshotData
    graph_snapshots = [snapshot_to_graph_data(cascade, snapshot) for snapshot in raw_snapshots]
    
    # 预测
    with torch.no_grad():
        pred_log, attention_weights, channel_scores, spatial_masks, temporal_mask, edge_masks, node_masks = model(graph_snapshots)
    prediction = max(1.0, torch.exp(pred_log).item() - 1)
    
    # 生成解释
    explanation = model.generate_explanation(graph_snapshots)
    
    # 提取关键模式
    key_edge_patterns = []
    for t, pattern in enumerate(explanation["edge_patterns"]):
        if pattern["num_selected_elements"] > 0:
            key_edge_patterns.append({
                "time_step": t + 1,
                "edges": pattern["selected_edges_indices"].tolist() if pattern["selected_edges_indices"].numel() > 0 else [],
                "active_nodes": torch.where(pattern["active_nodes_mask"])[0].tolist(),
                "importance": pattern["num_selected_elements"]
            })
    
    key_node_patterns = []
    for t, pattern in enumerate(explanation["node_patterns"]):
        if pattern["num_selected_elements"] > 0:
            key_node_patterns.append({
                "time_step": t + 1,
                "active_nodes": torch.where(pattern["active_nodes_mask"])[0].tolist(),
                "importance": pattern["num_selected_elements"]
            })
    
    # 时间注意力分布
    temporal_weights = attention_weights.tolist()
    top_time_slices = sorted(
        enumerate(temporal_weights), 
        key=lambda x: x[1], 
        reverse=True
    )[:3]
    
    # 通道重要性
    top_channels = channel_scores[:3]
    
    return {
        "cascade_id": cascade.cascade_id,
        "actual_size": cascade.target_size,
        "predicted_size": round(prediction, 4),
        "error": round(abs(prediction - cascade.target_size), 4),
        "temporal_attention": [
            {"slice": idx + 1, "weight": round(weight, 4)}
            for idx, weight in top_time_slices
        ],
        "channel_importance": [
            {"channel": item["feature"], "importance": round(item["importance"], 4)}
            for item in top_channels
        ],
        "key_edge_patterns": key_edge_patterns,
        "key_node_patterns": key_node_patterns,
        "raw_snapshots": raw_snapshots
    }

def visualize_key_paths(result: Dict[str, Any], output_dir: Path):
    """可视化关键传播路径"""
    cascade_id = result["cascade_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 时间注意力可视化
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    time_steps = [item["slice"] for item in result["temporal_attention"]]
    weights = [item["weight"] for item in result["temporal_attention"]]
    plt.bar(time_steps, weights)
    plt.title(f"Time Attention - Cascade {cascade_id}")
    plt.xlabel("Time Slice")
    plt.ylabel("Attention Weight")
    
    # 关键边模式可视化
    plt.subplot(1, 2, 2)
    if result["key_edge_patterns"]:
        time_steps = [p["time_step"] for p in result["key_edge_patterns"]]
        edge_counts = [p["importance"] for p in result["key_edge_patterns"]]
        plt.bar(time_steps, edge_counts)
        plt.title(f"Key Edge Patterns - Cascade {cascade_id}")
        plt.xlabel("Time Step")
        plt.ylabel("Number of Key Edges")
    else:
        plt.text(0.5, 0.5, "No significant edge patterns", ha="center", va="center")
        plt.title(f"Key Edge Patterns - Cascade {cascade_id}")
    
    plt.tight_layout()
    plt.savefig(output_dir / f"key_paths_{cascade_id}.png", dpi=300, bbox_inches="tight")
    plt.close()

def main():
    print("=" * 60)
    print("🔄 预测-解释端到端流水线")
    print("=" * 60)
    
    # 配置
    config = PipelineConfig()
    config.observation_seconds = 21600  # 6小时
    config.slice_seconds = 1800  # 30分钟
    config.learning_rate = 0.0005      # 优化后的学习率
    config.batch_size = 16             # 合适的批次大小
    config.epochs = 200                # 训练轮数
    config.patience = 30               # 早停耐心值
    
    # 加载数据
    print("📥 加载数据...")
    cascades = load_wikipedia_cascades("pp/sample_data/wikipedia.csv")
    print(f"  加载了 {len(cascades)} 个级联")
    
    # 准备模型
    print("🤖 准备模型...")
    
    # 检查缓存文件
    cache_path = "pp/cache/training_results.json"
    if Path(cache_path).exists():
        print("📦 从缓存加载训练结果...")
        # 从缓存加载配置
        cache_data = load_model_from_cache(cache_path)
        if 'config' in cache_data:
            # 更新配置
            for key, value in cache_data['config'].items():
                if hasattr(config, key):
                    setattr(config, key, value)
            print("  从缓存加载配置成功")
    
    # 运行训练获取模型
    print("🚀 运行模型训练...")
    report = run_dgnn_pipeline(
        cascades=cascades,
        config=config
    )
    
    # 从报告中获取模型
    if 'model' in report:
        model = report['model']
        model.eval()
        print("  从训练结果获取模型成功")
    else:
        # 如果报告中没有模型，使用随机初始化模型
        print("⚠️  报告中未找到模型，使用随机初始化模型")
        # 构建示例数据来确定维度
        sample_cascade = cascades[0]
        sample_snapshots = build_snapshots(sample_cascade, config)
        from cascade_model.dgnn import snapshot_to_graph_data
        sample_graph_data = [snapshot_to_graph_data(sample_cascade, s) for s in sample_snapshots]
        input_dim = sample_graph_data[0].node_features.shape[1]
        graph_dim = sample_graph_data[0].graph_features.shape[0]
        model = DynamicCascadeGNN(input_dim=input_dim, graph_dim=graph_dim)
    
    # 测试3-5个不同大小的级联
    print("🚀 开始预测和解释...")
    test_cascades = cascades[:5]  # 选择前5个级联
    results = []
    
    for i, cascade in enumerate(test_cascades):
        print(f"\n--- 处理级联 {i+1}/{len(test_cascades)} ---")
        print(f"  级联ID: {cascade.cascade_id}")
        print(f"  实际用户数: {cascade.target_size}")
        
        # 预测和解释
        result = predict_and_explain(cascade, model, config)
        results.append(result)
        
        # 打印结果
        print(f"  预测用户数: {result['predicted_size']}")
        print(f"  预测误差: {result['error']}")
        
        # 可视化
        visualize_key_paths(result, Path("outputs/visualizations"))
    
    # 保存结果
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "predict_explain_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print("✅ 预测-解释流水线完成！")
    print("📄 结果已保存到: outputs/predict_explain_results.json")
    print("📊 可视化已保存到: outputs/visualizations/")
    print("=" * 60)

if __name__ == "__main__":
    main()