"""Три расширяющихся временных обучения и сравнение v1 с финальной v2."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bot_solution.evaluation import metrics
from bot_solution.models import MODEL_SPECS, FINAL_WEIGHTS, blend_predictions, columns_for, make_model


FOLD_PERIODS = (
    ("2026-04-11", "2026-04-14"),
    ("2026-04-14", "2026-04-17"),
    ("2026-04-17", "2026-04-20"),
)


def make_folds(train: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """В fit попадают только окна, завершившиеся до проверочного периода.

    Это не случайный train/test split: финальный тест относится к более поздним
    датам. При пересекающихся окнах лучше остановиться, чем незаметно получить
    слишком оптимистичную оценку.
    """
    folds = []
    for start, end in FOLD_PERIODS:
        fit = train.window_start_ts.lt(pd.Timestamp(start)).to_numpy()
        valid = (train.window_start_ts.ge(pd.Timestamp(start))
                 & train.window_start_ts.lt(pd.Timestamp(end))).to_numpy()
        if not fit.any() or not valid.any():
            raise ValueError("Empty expanding-window fold")
        if train.loc[fit, "window_end_ts"].max() > train.loc[valid, "window_start_ts"].min():
            raise ValueError("Fit windows overlap a validation window")
        folds.append((fit, valid))
    return folds


def _summary(y: np.ndarray, prediction: np.ndarray, folds: list) -> dict:
    per_fold = [metrics(y[valid], prediction[valid]) for _, valid in folds]
    return {
        "folds": per_fold,
        "pooled": metrics(y[np.isfinite(prediction)], prediction[np.isfinite(prediction)]),
        # Среднее по блокам — критерий сравнения, не метрика на едином наборе.
        "weighted_mean_precision": float(np.average(
            [value["precision_at_recall_0.7"] for value in per_fold],
            weights=[value["n_positive"] for value in per_fold])),
    }


def evaluate_models(train: pd.DataFrame, features: pd.DataFrame, threads: int = 8) -> tuple[dict, pd.DataFrame]:
    """Повторяем оценку выбранных моделей, не ищем новый рецепт ансамбля.

    Все параметры уже зафиксированы. Данные проверки нужны только для метрик;
    даже early stopping по ним не включён. После оценки финальные модели будут
    заново обучены на всём train, а не взяты с одного из временных блоков.
    """
    if len(train) != len(features):
        raise ValueError("Train metadata and feature rows differ")
    y, folds = train.target.to_numpy(dtype=int), make_folds(train)
    predictions = {}
    for name, spec in MODEL_SPECS.items():
        columns = columns_for(spec, features)
        prediction = np.full(len(train), np.nan)
        for fold_number, (fit, valid) in enumerate(folds, 1):
            print(f"CV {name}, block {fold_number}: {fit.sum()} fit / {valid.sum()} check", flush=True)
            model = make_model(spec, threads)
            model.fit(features.loc[fit, columns], y[fit])
            prediction[valid] = model.predict_proba(features.loc[valid, columns])[:, 1]
        predictions[name] = prediction
    predictions["v1"] = blend_predictions(predictions, {"catboost_depth5": 0.5, "catboost_depth6": 0.5})
    predictions["v2"] = blend_predictions(predictions, FINAL_WEIGHTS)
    # Константа показывает, насколько метрика выше базовой доли положительных.
    predictions["constant"] = np.where(np.isfinite(predictions["v2"]), 0.5, np.nan)
    report = {
        "fold_periods": FOLD_PERIODS, "model_specs": MODEL_SPECS, "weights": FINAL_WEIGHTS,
        "validation_labels_used_for_early_stopping": False,
        "validation_is_internal_model_selection_not_an_untouched_test": True,
        "results": {name: _summary(y, score, folds) for name, score in predictions.items()},
    }
    valid = np.logical_or.reduce([mask for _, mask in folds])
    rows = train.loc[valid, ["cookie_id", "window_start_ts", "target"]].copy()
    for name, prediction in predictions.items():
        rows[name] = prediction[valid]
    return report, rows
