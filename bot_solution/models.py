"""Четыре выбранные модели. Поиск параметров здесь уже не выполняется.

CatBoost получает исходные 250 признаков, LightGBM — все 472. Именно такое
сочетание использовалось в отправленном результате с метрикой 0.82278.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

# LightGBM импортирует matplotlib даже без графиков. Его служебный кеш держим
# внутри виртуального окружения, чтобы не создавать файлы в домашнем каталоге.
os.environ.setdefault("MPLCONFIGDIR", str(Path(sys.prefix) / ".cache" / "matplotlib"))
from lightgbm import LGBMClassifier


MODEL_SPECS = {
    "catboost_depth5": {
        "kind": "catboost", "features": "original", "depth": 5,
        "iterations": 1076, "learning_rate": 0.04, "l2": 10, "seed": 42,
    },
    "catboost_depth6": {
        "kind": "catboost", "features": "original", "depth": 6,
        "iterations": 840, "learning_rate": 0.04, "l2": 10, "seed": 42,
    },
    "lightgbm_15": {
        "kind": "lightgbm", "features": "advanced", "leaves": 15,
        "iterations": 1100, "learning_rate": 0.035, "l2": 8,
        "min_leaf": 45, "seed": 42,
    },
    "lightgbm_31": {
        "kind": "lightgbm", "features": "advanced", "leaves": 31,
        "iterations": 1000, "learning_rate": 0.025, "l2": 12,
        "min_leaf": 60, "seed": 42,
    },
}
FINAL_WEIGHTS = {name: 0.25 for name in MODEL_SPECS}
DEFAULT_THREADS = 8


def columns_for(spec: dict, features: pd.DataFrame) -> list[str]:
    """Порядок столбцов сохраняем: он входит в воспроизводимость обучения."""
    if spec["features"] == "original":
        return [column for column in features if not column.startswith("v2_")]
    if spec["features"] == "advanced":
        return features.columns.tolist()
    raise ValueError(f"Unknown feature set: {spec['features']}")


def make_model(spec: dict, threads: int = DEFAULT_THREADS):
    """Создаём CPU-модель с полным набором зафиксированных параметров.

    Число деревьев уже выбрано. Проверочный блок не передаём как eval_set,
    поэтому его метки не влияют ни на обучение, ни на раннюю остановку.
    """
    if threads < 1:
        raise ValueError("threads must be positive")
    if spec["kind"] == "catboost":
        return CatBoostClassifier(
            iterations=spec["iterations"], learning_rate=spec["learning_rate"],
            depth=spec["depth"], l2_leaf_reg=spec["l2"], random_strength=1,
            bagging_temperature=0.5, bootstrap_type="Bayesian", scale_pos_weight=1,
            loss_function="Logloss", border_count=128, random_seed=spec["seed"],
            thread_count=threads, verbose=False, allow_writing_files=False,
            task_type="CPU", eval_metric="Logloss",
        )
    if spec["kind"] == "lightgbm":
        return LGBMClassifier(
            n_estimators=spec["iterations"], learning_rate=spec["learning_rate"],
            num_leaves=spec["leaves"], min_child_samples=spec["min_leaf"],
            reg_lambda=spec["l2"], reg_alpha=0.1, colsample_bytree=0.85,
            subsample=0.9, subsample_freq=1, max_bin=127, objective="binary",
            random_state=spec["seed"], n_jobs=threads, verbosity=-1,
            deterministic=True, force_col_wise=True,
        )
    raise ValueError(f"Unknown model kind: {spec['kind']}")


def component_weights(recipe: list[str] | dict[str, float]) -> dict[str, float]:
    """Проверяем веса; список означает равные доли для всех его моделей."""
    if isinstance(recipe, list):
        if not recipe or len(recipe) != len(set(recipe)):
            raise ValueError("Ensemble components must be nonempty and unique")
        weights = {name: 1 / len(recipe) for name in recipe}
    else:
        weights = dict(recipe)
    values = np.asarray(list(weights.values()), dtype=float)
    if (not len(values) or not np.isfinite(values).all()
            or (values <= 0).any() or not np.isclose(values.sum(), 1)):
        raise ValueError("Ensemble weights must be positive, finite and sum to 1")
    return weights


def blend_predictions(predictions: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Среднее вероятностей. Никаких порогов или статистик по тестовым кукам.

    Порядок суммирования совпадает с v2. NaN вне OOS-периода в валидации
    остаются NaN: эти строки не превращаются в фиктивные нулевые прогнозы.
    """
    weights = component_weights(weights)
    shapes = {np.asarray(predictions[name]).shape for name in weights}
    if len(shapes) != 1:
        raise ValueError("Prediction arrays must have the same shape")
    return np.sum([predictions[name] * weight for name, weight in weights.items()], axis=0)
