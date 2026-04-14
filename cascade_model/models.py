import random
from dataclasses import dataclass
from typing import List, Sequence


@dataclass
class StandardScaler:
    means: List[float] = None
    stds: List[float] = None

    def fit(self, rows: Sequence[Sequence[float]]) -> "StandardScaler":
        column_count = len(rows[0])
        self.means = []
        self.stds = []
        for col in range(column_count):
            values = [row[col] for row in rows]
            avg = sum(values) / len(values)
            variance = sum((value - avg) ** 2 for value in values) / len(values)
            std = variance ** 0.5
            self.means.append(avg)
            self.stds.append(std if std > 1e-9 else 1.0)
        return self

    def transform(self, rows: Sequence[Sequence[float]]) -> List[List[float]]:
        return [
            [(value - self.means[idx]) / self.stds[idx] for idx, value in enumerate(row)]
            for row in rows
        ]


@dataclass
class LinearRegressorGD:
    learning_rate: float = 0.01
    epochs: int = 2500
    l2_penalty: float = 0.0005
    random_seed: int = 42
    weights: List[float] = None
    bias: float = 0.0

    def fit(self, rows: Sequence[Sequence[float]], targets: Sequence[float]) -> "LinearRegressorGD":
        random.seed(self.random_seed)
        feature_count = len(rows[0])
        self.weights = [random.uniform(-0.05, 0.05) for _ in range(feature_count)]
        self.bias = 0.0

        sample_count = len(rows)
        for _ in range(self.epochs):
            grad_w = [0.0] * feature_count
            grad_b = 0.0
            for row, target in zip(rows, targets):
                pred = self._predict_one(row)
                error = pred - target
                for idx, value in enumerate(row):
                    grad_w[idx] += error * value
                grad_b += error
            for idx in range(feature_count):
                grad_w[idx] = (grad_w[idx] / sample_count) + self.l2_penalty * self.weights[idx]
                self.weights[idx] -= self.learning_rate * grad_w[idx]
            self.bias -= self.learning_rate * grad_b / sample_count
        return self

    def predict(self, rows: Sequence[Sequence[float]]) -> List[float]:
        return [max(1.0, self._predict_one(row)) for row in rows]

    def _predict_one(self, row: Sequence[float]) -> float:
        return sum(weight * value for weight, value in zip(self.weights, row)) + self.bias


@dataclass
class KNNRegressor:
    neighbors: int = 9
    train_rows: List[List[float]] = None
    train_targets: List[float] = None

    def fit(self, rows: Sequence[Sequence[float]], targets: Sequence[float]) -> "KNNRegressor":
        self.train_rows = [list(row) for row in rows]
        self.train_targets = list(targets)
        return self

    def predict(self, rows: Sequence[Sequence[float]]) -> List[float]:
        return [self._predict_one(row) for row in rows]

    def _predict_one(self, row: Sequence[float]) -> float:
        distances = []
        for train_row, target in zip(self.train_rows, self.train_targets):
            distance = _euclidean_distance(row, train_row)
            distances.append((distance, target))
        distances.sort(key=lambda item: item[0])
        top = distances[: max(1, self.neighbors)]
        weighted_sum = 0.0
        weight_total = 0.0
        for distance, target in top:
            weight = 1.0 / (distance + 1e-6)
            weighted_sum += weight * target
            weight_total += weight
        return weighted_sum / max(weight_total, 1e-9)


def _euclidean_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right)) ** 0.5
