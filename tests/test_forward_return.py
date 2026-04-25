# 测试forward方法返回值
import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cascade_model.dataset_profiles import load_dataset_and_config
from cascade_model.dgnn import build_dgnn_dataset, DynamicCascadeGNN

def test_forward_return():
    """测试forward方法返回值"""
    print("\n" + "=" * 60)
    print("🧪 测试forward方法返回值")
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
        
        dataset_name, cascades, config = load_dataset_and_config(input_path, "wikipedia")
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
            result = model(sample.snapshots)
        
        print(f"\nForward方法返回值:")
        print(f"  返回值数量: {len(result)}")
        print(f"  返回值类型: {[type(r).__name__ for r in result]}")
        
        if len(result) == 8:
            print(f"  ✅ 返回值数量正确 (8个)")
            size_pred, growth_pred, weights, channel_scores, spatial_masks, temporal_mask, edge_masks, node_masks = result
            print(f"  size_pred: {size_pred.item():.4f}")
            print(f"  growth_pred: {growth_pred.item():.4f}")
            print(f"  weights shape: {weights.shape}")
            print(f"  channel_scores: {len(channel_scores)} items")
            print(f"  spatial_masks: {len(spatial_masks)} items")
            print(f"  temporal_mask: {temporal_mask.shape}")
            print(f"  edge_masks: {len(edge_masks)} items")
            print(f"  node_masks: {len(node_masks)} items")
        else:
            print(f"  ❌ 返回值数量错误 (期望8个，实际{len(result)}个)")
            
            return False
            
        return True
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_forward_return()