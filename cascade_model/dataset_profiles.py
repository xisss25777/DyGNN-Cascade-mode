from pathlib import Path
from typing import Optional, Tuple

from .config import PipelineConfig
from .data import generate_synthetic_cascades, load_cascades_from_csv, load_wikipedia_cascades
from .tg_data_loader import load_tg_csv, augment_by_subcascade


DEFAULT_DATASET_PATHS = {
    "wikipedia": Path("sample_data/wikipedia.csv"),
    "reddit":    Path("sample_data/reddit.csv"),
    "enron":     Path("sample_data/enron.csv"),
    "mooc":      Path("sample_data/mooc.csv"),
    "cascade":   Path("sample_data/cascades.csv"),
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

    if dataset_name in {"wikipedia", "reddit", "enron", "mooc"}:
        if not resolved_input_path:
            raise ValueError(f"未找到 {dataset_name} 数据文件")
        # 使用统一的 TG 格式加载器
        raw_cascades = load_tg_csv(
            resolved_input_path,
            min_cascade_size=_min_size(dataset_name),
            max_cascades=_max_cascades(dataset_name),
        )
        # 数据增强：子级联采样（扩展小数据集）
        cascades = augment_by_subcascade(raw_cascades, n_splits=3, min_events=5)
        print(f"[{dataset_name}] 原始级联: {len(raw_cascades)}, 增强后: {len(cascades)}")
        return dataset_name, cascades

    raise ValueError(f"未知 dataset 类型: {dataset_name}")


def _min_size(dataset_name: str) -> int:
    return {"enron": 2, "mooc": 2}.get(dataset_name, 5)


def _max_cascades(dataset_name: str) -> int:
    return {"wikipedia": 150, "reddit": 150, "enron": 200, "mooc": 200}.get(dataset_name, 200)


def get_dgnn_config(dataset_name: str) -> PipelineConfig:
    base = {
        "wikipedia": PipelineConfig(
            observation_seconds=6 * 3600,
            slice_seconds=1800,
            learning_rate=0.001,
            epochs=300,
            l2_penalty=0.0001,
            use_log_target=True,
            knn_neighbors=11,
            patience=50,
        ),
        "reddit": PipelineConfig(
            observation_seconds=6 * 3600,
            slice_seconds=1800,
            learning_rate=0.001,
            epochs=300,
            l2_penalty=0.0001,
            use_log_target=True,
            knn_neighbors=13,
            patience=50,
        ),
        "enron": PipelineConfig(
            observation_seconds=24 * 3600,
            slice_seconds=3600,
            learning_rate=0.001,
            epochs=300,
            l2_penalty=0.0002,
            use_log_target=True,
            knn_neighbors=9,
            patience=50,
        ),
        "mooc": PipelineConfig(
            observation_seconds=12 * 3600,
            slice_seconds=2400,
            learning_rate=0.001,
            epochs=300,
            l2_penalty=0.0001,
            use_log_target=True,
            knn_neighbors=11,
            patience=50,
        ),
    }
    return base.get(dataset_name, PipelineConfig())


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