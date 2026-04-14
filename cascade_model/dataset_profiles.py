from pathlib import Path
from typing import Optional, Tuple

from .config import PipelineConfig
from .data import generate_synthetic_cascades, load_cascades_from_csv, load_wikipedia_cascades


DEFAULT_DATASET_PATHS = {
    "wikipedia": Path("sample_data/wikipedia.csv"),
    "reddit": Path("sample_data/reddit.csv"),
    "cascade": Path("sample_data/cascades.csv"),
}


def load_dataset(dataset: str, input_path: Optional[str]) -> Tuple[str, list]:
    dataset_name = dataset
    resolved_input_path = resolve_input_path(input_path, dataset_name)

    if dataset_name == "synthetic":
        return dataset_name, generate_synthetic_cascades()
    if dataset_name == "cascade":
        if not resolved_input_path:
            default_path = DEFAULT_DATASET_PATHS["cascade"]
            raise ValueError(
                "当 dataset=cascade 时，必须提供 --input，"
                f"或先生成默认样例文件: {default_path}。"
            )
        return dataset_name, load_cascades_from_csv(resolved_input_path)
    if dataset_name == "wikipedia":
        if not resolved_input_path:
            raise ValueError("未找到 Wikipedia 数据文件: sample_data/wikipedia.csv")
        return dataset_name, load_wikipedia_cascades(resolved_input_path)
    if dataset_name == "reddit":
        if not resolved_input_path:
            raise ValueError("未找到 Reddit 数据文件: sample_data/reddit.csv")
        return dataset_name, load_wikipedia_cascades(resolved_input_path)
    raise ValueError(f"未知 dataset 类型: {dataset_name}")


def get_dgnn_config(dataset_name: str) -> PipelineConfig:
    if dataset_name == "wikipedia":
        return PipelineConfig(
            observation_seconds=6 * 3600,
            slice_seconds=1800,
            learning_rate=0.0015,
            epochs=200,  # 减少训练轮数以加速训练
            l2_penalty=0.0001,
            use_log_target=True,
            knn_neighbors=11,
        )
    if dataset_name == "reddit":
        return PipelineConfig(
            observation_seconds=6 * 3600,
            slice_seconds=1800,
            learning_rate=0.0012,
            epochs=200,  # 减少训练轮数以加速训练
            l2_penalty=0.0001,
            use_log_target=True,
            knn_neighbors=13,
        )
    return PipelineConfig()


def get_baseline_config(dataset_name: str) -> PipelineConfig:
    if dataset_name == "wikipedia":
        return PipelineConfig(
            observation_seconds=6 * 3600,
            slice_seconds=1800,
            learning_rate=0.006,
            epochs=900,
            l2_penalty=0.001,
            use_log_target=True,
            knn_neighbors=11,
        )
    if dataset_name == "reddit":
        return PipelineConfig(
            observation_seconds=6 * 3600,
            slice_seconds=1800,
            learning_rate=0.005,
            epochs=1000,
            l2_penalty=0.0012,
            use_log_target=True,
            knn_neighbors=13,
        )
    return PipelineConfig()


def load_dataset_and_config(input_path: Optional[str], dataset: str) -> Tuple[str, list, PipelineConfig]:
    dataset_name, cascades = load_dataset(dataset, input_path)
    return dataset_name, cascades, get_dgnn_config(dataset_name)


def resolve_input_path(input_path: Optional[str], dataset_name: str) -> Optional[str]:
    if input_path:
        return input_path
    default_path = DEFAULT_DATASET_PATHS.get(dataset_name)
    if default_path and default_path.exists():
        return str(default_path)
    return None