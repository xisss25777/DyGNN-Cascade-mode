# diagnose_current_state.py
import torch
import os
import json


def diagnose_wikipedia_state():
    """诊断当前wikipedia保存状态"""
    print("=== 诊断wikipedia当前状态 ===")

    # 检查所有相关文件
    files = [
        "outputs/wikipedia_edge_mask_tensor.pt",
        "outputs/wikipedia_edge_mask_list.pt",
        "outputs/wikipedia_node_mask_tensor.pt",
        "outputs/wikipedia_time_slices.json"
    ]

    file_info = {}

    for file_path in files:
        if os.path.exists(file_path):
            print(f"\n✅ 文件存在: {file_path}")
            file_size = os.path.getsize(file_path) / 1024
            print(f"   大小: {file_size:.1f} KB")

            try:
                if file_path.endswith('.pt'):
                    data = torch.load(file_path)
                    data_type = type(data)
                    print(f"   类型: {data_type}")

                    if isinstance(data, torch.Tensor):
                        print(f"   形状: {data.shape}")
                        print(f"   值范围: [{data.min():.4f}, {data.max():.4f}]")
                        print(f"   平均值: {data.mean():.4f}")
                    elif isinstance(data, list):
                        print(f"   列表长度: {len(data)}")
                        if len(data) > 0 and isinstance(data[0], torch.Tensor):
                            print(f"   第一个元素形状: {data[0].shape}")
                            # 检查所有元素形状
                            shapes = [item.shape for item in data if hasattr(item, 'shape')]
                            if shapes:
                                print(f"   所有元素形状: {shapes[:3]}...")

                    file_info[file_path] = {
                        'type': str(data_type),
                        'exists': True
                    }

                elif file_path.endswith('.json'):
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        print(f"   JSON列表长度: {len(data)}")
                        if len(data) > 0:
                            print(f"   第一个元素键: {list(data[0].keys())}")
            except Exception as e:
                print(f"   ❌ 加载失败: {e}")
        else:
            print(f"\n❌ 文件不存在: {file_path}")
            file_info[file_path] = {'exists': False}

    return file_info


if __name__ == "__main__":
    diagnose_wikipedia_state()