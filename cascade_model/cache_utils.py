"""
缓存工具模块
用于加速数据加载和模型训练
"""

import os
import json
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from .data import Cascade
from .dynamic_graph import Snapshot, build_snapshots
from .config import PipelineConfig


class CacheManager:
    """
    缓存管理器
    """
    def __init__(self, cache_dir: str = "cache"):
        """
        初始化缓存管理器
        
        Args:
            cache_dir: 缓存目录
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # 缓存文件路径
        self.snapshots_cache = self.cache_dir / "snapshots.pkl"
        self.model_cache = self.cache_dir / "models"
        self.model_cache.mkdir(exist_ok=True)
        self.training_cache = self.cache_dir / "training_results.json"
        
        # 内存缓存
        self._snapshots_cache: Dict[str, List[Snapshot]] = {}
        self._training_results: Dict[str, Dict] = {}
        
        # 加载缓存
        self._load_snapshots_cache()
        self._load_training_results()
    
    def _load_snapshots_cache(self):
        """
        加载快照缓存
        """
        if self.snapshots_cache.exists():
            try:
                with open(self.snapshots_cache, 'rb') as f:
                    self._snapshots_cache = pickle.load(f)
                print(f"加载快照缓存成功，共 {len(self._snapshots_cache)} 个缓存项")
            except Exception as e:
                print(f"加载快照缓存失败: {e}")
                self._snapshots_cache = {}
    
    def _save_snapshots_cache(self):
        """
        保存快照缓存
        """
        try:
            with open(self.snapshots_cache, 'wb') as f:
                pickle.dump(self._snapshots_cache, f)
            print(f"保存快照缓存成功，共 {len(self._snapshots_cache)} 个缓存项")
        except Exception as e:
            print(f"保存快照缓存失败: {e}")
    
    def _load_training_results(self):
        """
        加载训练结果缓存
        """
        if self.training_cache.exists():
            try:
                with open(self.training_cache, 'r', encoding='utf-8') as f:
                    self._training_results = json.load(f)
                print(f"加载训练结果缓存成功，共 {len(self._training_results)} 个缓存项")
            except Exception as e:
                print(f"加载训练结果缓存失败: {e}")
                self._training_results = {}
    
    def _save_training_results(self):
        """
        保存训练结果缓存
        """
        try:
            with open(self.training_cache, 'w', encoding='utf-8') as f:
                json.dump(self._training_results, f, ensure_ascii=False, indent=2)
            print(f"保存训练结果缓存成功，共 {len(self._training_results)} 个缓存项")
        except Exception as e:
            print(f"保存训练结果缓存失败: {e}")
    
    def get_snapshots(self, cascade: Cascade, config: PipelineConfig) -> List[Snapshot]:
        """
        获取级联的快照，如果缓存中存在则直接返回，否则计算并缓存
        
        Args:
            cascade: 级联对象
            config: 配置对象
        
        Returns:
            快照列表
        """
        # 生成缓存键
        cache_key = f"{cascade.cascade_id}_{config.observation_seconds}_{config.slice_seconds}"
        
        # 检查内存缓存
        if cache_key in self._snapshots_cache:
            return self._snapshots_cache[cache_key]
        
        # 计算快照
        snapshots = build_snapshots(
            cascade, 
            observation_seconds=config.observation_seconds, 
            slice_seconds=config.slice_seconds
        )
        
        # 缓存结果
        self._snapshots_cache[cache_key] = snapshots
        
        # 每100个缓存项保存一次
        if len(self._snapshots_cache) % 100 == 0:
            self._save_snapshots_cache()
        
        return snapshots
    
    def save_model(self, model: Any, model_name: str):
        """
        保存模型
        
        Args:
            model: 模型对象
            model_name: 模型名称
        """
        model_path = self.model_cache / f"{model_name}.pkl"
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"模型保存成功: {model_name}")
        except Exception as e:
            print(f"模型保存失败: {e}")
    
    def load_model(self, model_name: str) -> Optional[Any]:
        """
        加载模型
        
        Args:
            model_name: 模型名称
        
        Returns:
            模型对象，如果不存在则返回None
        """
        model_path = self.model_cache / f"{model_name}.pkl"
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                print(f"模型加载成功: {model_name}")
                return model
            except Exception as e:
                print(f"模型加载失败: {e}")
                return None
        return None
    
    def save_training_result(self, key: str, result: Dict):
        """
        保存训练结果
        
        Args:
            key: 缓存键
            result: 训练结果
        """
        self._training_results[key] = result
        self._save_training_results()
    
    def get_training_result(self, key: str) -> Optional[Dict]:
        """
        获取训练结果
        
        Args:
            key: 缓存键
        
        Returns:
            训练结果，如果不存在则返回None
        """
        return self._training_results.get(key)
    
    def clear_cache(self):
        """
        清除所有缓存
        """
        # 清除内存缓存
        self._snapshots_cache = {}
        self._training_results = {}
        
        # 清除文件缓存
        if self.snapshots_cache.exists():
            self.snapshots_cache.unlink()
        
        if self.training_cache.exists():
            self.training_cache.unlink()
        
        # 清除模型缓存
        for model_file in self.model_cache.glob("*.pkl"):
            model_file.unlink()
        
        print("缓存已清除")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """
        获取缓存统计信息
        
        Returns:
            缓存统计信息
        """
        return {
            "snapshots_cache_size": len(self._snapshots_cache),
            "training_results_size": len(self._training_results),
            "models_count": len(list(self.model_cache.glob("*.pkl")))
        }


# 全局缓存管理器实例
cache_manager = CacheManager()


@lru_cache(maxsize=1000)
def cached_build_snapshots(
    cascade_id: str,
    observation_seconds: int,
    slice_seconds: int,
    events: Tuple[Tuple[str, int, Optional[str], Dict]],
) -> List[Snapshot]:
    """
    缓存版本的build_snapshots函数
    
    Args:
        cascade_id: 级联ID
        observation_seconds: 观测窗口
        slice_seconds: 时间切片大小
        events: 事件元组
    
    Returns:
        快照列表
    """
    # 重建Cascade对象
    from .data import Event
    
    cascade = type('Cascade', (), {
        'cascade_id': cascade_id,
        'events': [
            Event(
                user_id=e[0],
                timestamp=e[1],
                parent_id=e[2],
                extra_features=e[3]
            ) for e in events
        ]
    })()
    
    return build_snapshots(cascade, observation_seconds, slice_seconds)


def get_cached_snapshots(cascade: Cascade, config: PipelineConfig) -> List[Snapshot]:
    """
    获取缓存的快照
    
    Args:
        cascade: 级联对象
        config: 配置对象
    
    Returns:
        快照列表
    """
    # 转换事件为可哈希的元组
    events_tuple = tuple(
        (e.user_id, e.timestamp, e.parent_id, e.extra_features)
        for e in cascade.events
    )
    
    return cached_build_snapshots(
        cascade.cascade_id,
        config.observation_seconds,
        config.slice_seconds,
        events_tuple
    )


def benchmark_cache_performance(cascades: List[Cascade], config: PipelineConfig, iterations: int = 3):
    """
    基准测试缓存性能
    
    Args:
        cascades: 级联列表
        config: 配置对象
        iterations: 测试迭代次数
    """
    import time
    
    print("=" * 60)
    print("🚀 开始缓存性能基准测试")
    print("=" * 60)
    
    # 测试不使用缓存的情况
    print("\n1. 测试不使用缓存:")
    start_time = time.time()
    for i in range(iterations):
        print(f"  迭代 {i+1}/{iterations}...")
        for cascade in cascades[:10]:  # 只测试前10个级联
            build_snapshots(cascade, config.observation_seconds, config.slice_seconds)
    no_cache_time = time.time() - start_time
    print(f"  总耗时: {no_cache_time:.2f}秒")
    print(f"  平均耗时: {no_cache_time / iterations:.2f}秒/次")
    
    # 测试使用缓存的情况
    print("\n2. 测试使用缓存:")
    start_time = time.time()
    for i in range(iterations):
        print(f"  迭代 {i+1}/{iterations}...")
        for cascade in cascades[:10]:  # 只测试前10个级联
            get_cached_snapshots(cascade, config)
    cache_time = time.time() - start_time
    print(f"  总耗时: {cache_time:.2f}秒")
    print(f"  平均耗时: {cache_time / iterations:.2f}秒/次")
    
    # 计算性能提升
    speedup = no_cache_time / cache_time if cache_time > 0 else float('inf')
    print(f"\n3. 性能提升:")
    print(f"  速度提升: {speedup:.2f}x")
    
    print("\n" + "=" * 60)
    print("✅ 基准测试完成")
    print("=" * 60)