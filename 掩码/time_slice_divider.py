import torch
import numpy as np
from typing import List, Dict, Any, Optional
import math


class TimeSliceDivider:
    def __init__(self, strategy='uniform_time', device=None):
        """
        参数: 
            strategy: 划分策略，可选: 
                - 'uniform_time': 均匀时间窗口 
                - 'uniform_events': 均匀事件数量 
                - 'adaptive': 自适应划分 
                - 'quantile': 分位数划分 
            device: 计算设备 
        """
        self.strategy = strategy
        self.device = device if device is not None else torch.device('cpu')
        
    def generate_time_slices_from_timestamps(
        self,
        timestamps: torch.Tensor,
        num_slices: Optional[int] = None,
        time_window_size: Optional[float] = None,
        min_events_per_slice: int = 50
    ) -> List[Dict[str, Any]]:
        """
        从时间戳生成时间片划分
        
        参数: 
            timestamps: 时间戳张量，形状 (N_events,)
            num_slices: 时间片数量（可选）
            time_window_size: 时间窗口大小（可选）
            min_events_per_slice: 每个时间片最小事件数（用于自适应划分）
            
        返回: 
            time_slices_info: 时间片信息列表，每个元素包含: 
                {
                    'slice_index': int,           # 时间片索引 (0-based)
                    'start_time': float,          # 开始时间
                    'end_time': float,            # 结束时间
                    'event_indices': torch.Tensor, # 属于这个时间片的事件索引
                    'num_events': int,            # 事件数量
                    'time_span': float,           # 时间跨度
                    'event_density': float        # 事件密度
                }
        """
        # 验证输入
        if not isinstance(timestamps, torch.Tensor):
            raise TypeError("timestamps 必须是 torch.Tensor 类型")
        
        if timestamps.dim() != 1:
            raise ValueError("timestamps 必须是一维张量")
        
        if len(timestamps) == 0:
            return []
        
        # 处理负数时间戳
        timestamps = timestamps[timestamps >= 0]
        if len(timestamps) == 0:
            # 所有时间戳都是负数，使用默认时间戳
            timestamps = torch.linspace(0, 10000, 1000)
        
        # 排序时间戳
        sorted_indices = torch.argsort(timestamps)
        sorted_timestamps = timestamps[sorted_indices]
        
        # 确定时间片数量
        if num_slices is None:
            num_slices = self.suggest_optimal_num_slices(sorted_timestamps)
        
        # 限制时间片数量
        max_slices = 20
        num_slices = min(num_slices, max_slices)
        num_slices = max(num_slices, 2)  # 确保至少有2个时间片
        
        # 根据策略生成时间片
        if self.strategy == 'uniform_time':
            time_slices = self.generate_uniform_time_slices(sorted_timestamps, num_slices)
        elif self.strategy == 'uniform_events':
            time_slices = self.generate_uniform_event_slices(sorted_timestamps, num_slices)
        elif self.strategy == 'adaptive':
            time_slices = self.generate_adaptive_time_slices(sorted_timestamps, min_events_per_slice)
        elif self.strategy == 'quantile':
            time_slices = self.generate_quantile_time_slices(sorted_timestamps, num_slices)
        else:
            raise ValueError(f"不支持的划分策略: {self.strategy}")
        
        # 为每个时间片添加事件索引
        valid_time_slices = []
        for i, time_slice in enumerate(time_slices):
            start_time = time_slice['start_time']
            end_time = time_slice['end_time']
            
            # 找到属于当前时间片的事件索引
            if i == len(time_slices) - 1:
                # 最后一个时间片包含到最后一个事件
                mask = (sorted_timestamps >= start_time) & (sorted_timestamps <= end_time)
            else:
                # 其他时间片不包含结束时间
                mask = (sorted_timestamps >= start_time) & (sorted_timestamps < end_time)
            
            event_indices = sorted_indices[mask]
            num_events = len(event_indices)
            
            # 只保留有数据的时间片
            if num_events > 0:
                time_span = end_time - start_time
                event_density = num_events / time_span if time_span > 0 else 0
                
                time_slice.update({
                    'event_indices': event_indices,
                    'num_events': num_events,
                    'time_span': time_span,
                    'event_density': event_density
                })
                valid_time_slices.append(time_slice)
        
        # 重新分配时间片索引
        for i, time_slice in enumerate(valid_time_slices):
            time_slice['slice_index'] = i
        
        # 验证时间片划分
        self.validate_time_slices(valid_time_slices)
        
        return valid_time_slices
    
    def generate_uniform_time_slices(self, timestamps, num_slices):
        """
        均匀时间划分：每个时间片的时间跨度相同
        """
        min_time = timestamps.min().item()
        max_time = timestamps.max().item()
        time_range = max_time - min_time
        
        if time_range == 0:
            # 所有时间戳相同
            return [{
                'slice_index': 0,
                'start_time': min_time,
                'end_time': max_time
            }]
        
        time_window = time_range / num_slices
        time_slices = []
        
        for i in range(num_slices):
            start_time = min_time + i * time_window
            end_time = min_time + (i + 1) * time_window
            if i == num_slices - 1:
                end_time = max_time  # 确保最后一个时间片包含最大时间
            
            time_slices.append({
                'slice_index': i,
                'start_time': start_time,
                'end_time': end_time
            })
        
        return time_slices
    
    def generate_uniform_event_slices(self, timestamps, num_slices):
        """
        均匀事件划分：每个时间片包含相同数量的事件
        """
        num_events = len(timestamps)
        
        # 确保时间片数量合理
        num_slices = min(num_slices, num_events)
        if num_slices < 1:
            num_slices = 1
        
        events_per_slice = num_events // num_slices
        
        time_slices = []
        for i in range(num_slices):
            start_idx = i * events_per_slice
            if i == num_slices - 1:
                end_idx = num_events
            else:
                end_idx = (i + 1) * events_per_slice
            
            # 确保时间片有数据
            if start_idx < end_idx:
                start_time = timestamps[start_idx].item()
                end_time = timestamps[end_idx - 1].item()
                
                time_slices.append({
                    'slice_index': i,
                    'start_time': start_time,
                    'end_time': end_time
                })
        
        return time_slices
    
    def generate_adaptive_time_slices(self, timestamps, min_events_per_slice=50):
        """
        自适应划分：确保每个时间片至少有 min_events_per_slice 个事件
        在事件间隔大的地方切分
        """
        num_events = len(timestamps)
        if num_events <= min_events_per_slice:
            # 事件数量不足，返回单个时间片
            return [{
                'slice_index': 0,
                'start_time': timestamps.min().item(),
                'end_time': timestamps.max().item()
            }]
        
        # 限制最大时间片数量
        max_slices = 20
        events_per_slice = max(min_events_per_slice, num_events // max_slices)
        
        # 计算时间间隔
        time_diffs = timestamps[1:] - timestamps[:-1]
        time_diffs = time_diffs.cpu().numpy()
        
        # 找到大的时间间隔作为切分点
        sorted_diff_indices = np.argsort(time_diffs)[::-1]  # 降序排序
        
        # 选择切分点
        cut_points = []
        current_events = num_events
        
        for idx in sorted_diff_indices:
            if current_events <= events_per_slice or len(cut_points) >= max_slices - 1:
                break
            
            # 检查切分后两边的事件数量
            left_events = idx + 1
            right_events = num_events - left_events
            
            if left_events >= events_per_slice and right_events >= events_per_slice:
                cut_points.append(idx + 1)  # 切分点是事件索引
                current_events = right_events
        
        # 按顺序排序切分点
        cut_points.sort()
        
        # 生成时间片
        time_slices = []
        start_idx = 0
        
        for i, cut_point in enumerate(cut_points):
            start_time = timestamps[start_idx].item()
            end_time = timestamps[cut_point - 1].item()
            
            time_slices.append({
                'slice_index': i,
                'start_time': start_time,
                'end_time': end_time
            })
            
            start_idx = cut_point
        
        # 添加最后一个时间片
        if start_idx < num_events:
            start_time = timestamps[start_idx].item()
            end_time = timestamps[-1].item()
            
            time_slices.append({
                'slice_index': len(cut_points),
                'start_time': start_time,
                'end_time': end_time
            })
        
        return time_slices
    
    def load_real_timestamps(self, dataset_name='wikipedia'):
        """
        从真实数据集中加载时间戳
        
        Args:
            dataset_name: 数据集名称
        
        Returns:
            timestamps: 时间戳张量
        """
        # 使用绝对路径确保正确读取
        data_dir = Path("E:\建模\pycharm项目\TG_network_datasets") / dataset_name
        edge_features_path = data_dir / f"ml_{dataset_name}.npy"
        
        if edge_features_path.exists():
            edge_features = np.load(edge_features_path)
            # 使用边特征的和作为时间戳
            time_values = edge_features.sum(axis=1)
            timestamps = torch.tensor(time_values, dtype=torch.float32)
            print(f"从 {dataset_name} 数据集加载了 {len(timestamps)} 个时间戳数据")
            return timestamps
        else:
            raise FileNotFoundError(f"数据集文件不存在: {dataset_name}")
    
    def generate_quantile_time_slices(self, timestamps, num_slices):
        """
        分位数划分：按时间分位数划分，确保每个时间片的事件分布均匀
        """
        num_events = len(timestamps)
        quantiles = torch.linspace(0, 1, num_slices + 1)
        time_bounds = torch.quantile(timestamps, quantiles)
        
        time_slices = []
        for i in range(num_slices):
            start_time = time_bounds[i].item()
            end_time = time_bounds[i + 1].item()
            
            time_slices.append({
                'slice_index': i,
                'start_time': start_time,
                'end_time': end_time
            })
        
        return time_slices
    
    def suggest_optimal_num_slices(
        self,
        timestamps: torch.Tensor,
        strategy: str = 'empirical'
    ) -> int:
        """
        根据数据特征建议最优时间片数量
        
        策略: 
            - 'empirical': 经验公式 K = min(20, max(2, N_events / 1000))
            - 'sqrt': K = sqrt(N_events)
            - 'sturges': K = ceil(log2(N_events)) + 1
            - 'rice': K = ceil(2 * N_events^(1/3))
            
        返回: 
            suggested_K: 建议的时间片数量
        """
        num_events = len(timestamps)
        
        if num_events == 0:
            return 1
        
        if strategy == 'empirical':
            k = max(2, min(20, num_events // 1000))
        elif strategy == 'sqrt':
            k = max(2, min(20, int(math.sqrt(num_events))))
        elif strategy == 'sturges':
            k = max(2, min(20, int(math.ceil(math.log2(num_events))) + 1))
        elif strategy == 'rice':
            k = max(2, min(20, int(math.ceil(2 * num_events ** (1/3)))))
        else:
            raise ValueError(f"不支持的策略: {strategy}")
        
        return k
    
    def validate_time_slices(self, time_slices_info):
        """
        验证时间片划分的合理性
        """
        if not time_slices_info:
            return
        
        # 检查时间片是否覆盖所有时间范围
        start_times = [slice_info['start_time'] for slice_info in time_slices_info]
        end_times = [slice_info['end_time'] for slice_info in time_slices_info]
        
        # 检查时间片是否按顺序排列
        for i in range(1, len(time_slices_info)):
            if start_times[i] < end_times[i-1]:
                raise ValueError(f"时间片 {i} 的开始时间小于时间片 {i-1} 的结束时间")
        
        # 检查时间片是否有重叠
        for i in range(len(time_slices_info)):
            for j in range(i+1, len(time_slices_info)):
                if not (time_slices_info[i]['end_time'] <= time_slices_info[j]['start_time'] or 
                        time_slices_info[j]['end_time'] <= time_slices_info[i]['start_time']):
                    raise ValueError(f"时间片 {i} 和时间片 {j} 有重叠")
        
        # 检查时间片数量是否合理
        num_slices = len(time_slices_info)
        if num_slices > 100:
            print(f"警告: 时间片数量 {num_slices} 可能过多")
        elif num_slices < 2:
            print(f"警告: 时间片数量 {num_slices} 可能过少")


# 示例用法
if __name__ == "__main__":
    # 从超参数搜索结果中提取时间戳数据
    import json
    import os
    from pathlib import Path
    
    # 尝试从JSON文件中加载数据
    timestamps = None
    json_path = Path(r"E:\建模\pycharm项目\神经\cascade_model\outputs\hyperparameter_search_results.json")
    
    if json_path.exists():
        print("从 hyperparameter_search_results.json 加载数据...")
        with open(json_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # 从结果中提取时间相关数据（使用学习率、批次大小等作为时间戳）
        # 这里使用学习率的倒数作为时间戳，模拟时间流逝
        time_values = []
        for i, result in enumerate(results):
            # 使用学习率的倒数作为时间戳
            lr = result['learning_rate']
            time_val = 1.0 / lr
            time_values.append(time_val)
        
        # 转换为张量
        timestamps = torch.tensor(time_values, dtype=torch.float32)
        print(f"加载了 {len(timestamps)} 个时间戳数据")
        print(f"时间范围: {timestamps.min().item():.2f} - {timestamps.max().item():.2f}")
    else:
        # 如果JSON文件不存在，从npy文件中提取数据
        print("从TG_network_datasets加载数据...")
        dataset_name = 'wikipedia'
        data_dir = Path(r"E:\建模\pycharm项目\TG_network_datasets") / dataset_name
        edge_features_path = data_dir / f"ml_{dataset_name}.npy"
        
        if edge_features_path.exists():
            edge_features = np.load(edge_features_path)
            # 使用边特征的和作为时间戳
            time_values = edge_features.sum(axis=1)[:1000]  # 只使用前1000个
            timestamps = torch.tensor(time_values, dtype=torch.float32)
            print(f"从 {dataset_name} 数据集加载了 {len(timestamps)} 个时间戳数据")
            print(f"时间范围: {timestamps.min().item():.2f} - {timestamps.max().item():.2f}")
        else:
            print("无法加载真实数据，使用默认时间戳")
            # 作为后备，使用固定的时间戳
            timestamps = torch.linspace(0, 10000, 1000)
    
    # 测试不同策略
    strategies = ['uniform_time', 'uniform_events', 'adaptive', 'quantile']
    
    for strategy in strategies:
        print(f"\n===== 测试 {strategy} 策略 =====")
        divider = TimeSliceDivider(strategy=strategy)
        
        # 生成时间片
        time_slices = divider.generate_time_slices_from_timestamps(timestamps)
        
        # 打印结果
        print(f"生成了 {len(time_slices)} 个时间片")
        for i, time_slice in enumerate(time_slices[:3]):  # 只打印前3个时间片
            print(f"时间片 {i}:")
            print(f"  时间范围: {time_slice['start_time']:.2f} - {time_slice['end_time']:.2f}")
            print(f"  事件数量: {time_slice['num_events']}")
            print(f"  时间跨度: {time_slice['time_span']:.2f}")
            print(f"  事件密度: {time_slice['event_density']:.4f}")
        if len(time_slices) > 3:
            print(f"... 共 {len(time_slices)} 个时间片")
    
    # 测试最优时间片数量建议
    print("\n===== 测试最优时间片数量建议 =====")
    divider = TimeSliceDivider()
    
    strategies = ['empirical', 'sqrt', 'sturges', 'rice']
    for strategy in strategies:
        k = divider.suggest_optimal_num_slices(timestamps, strategy=strategy)
        print(f"{strategy} 策略建议的时间片数量: {k}")
    
    # 测试验证功能
    print("\n===== 测试时间片验证 =====")
    divider = TimeSliceDivider(strategy='uniform_time')
    time_slices = divider.generate_time_slices_from_timestamps(timestamps)
    divider.validate_time_slices(time_slices)
    print("时间片验证通过！")