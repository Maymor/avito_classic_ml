"""Загрузка, признаки, финальное обучение и проверка результата.

Функции используются и из run.py, и из ноутбука. Поэтому пути, обработка
данных и параметры модели не расходятся между двумя способами запуска.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from bot_solution.advanced_features import build_advanced_features
from bot_solution.features import load_data, prepare_events
from bot_solution.models import MODEL_SPECS, FINAL_WEIGHTS, columns_for, make_model, blend_predictions
from lightgbm import Booster


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATA_FILES = ("train.csv", "test.csv", "events.csv.gz")


@dataclass
class PreparedData:
    train: pd.DataFrame
    test: pd.DataFrame
    meta: pd.DataFrame
    events: pd.DataFrame
    audit: dict
    input_hashes: dict


@dataclass
class FeatureMatrices:
    train: pd.DataFrame
    test: pd.DataFrame


@dataclass
class FittedEnsemble:
    models: dict
    model_info: list[dict]
    importance: pd.Series


def checksum(path: Path) -> str:
    """Читаем файл блоками; большие исходные события не держим ради SHA в памяти."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def reference_metadata() -> dict:
    """Эталон содержит контрольные суммы, не ответы под отдельные cookie_id."""
    return json.loads((REPOSITORY_ROOT / "metadata" / "reference.json").read_text(encoding="utf-8"))


def input_hashes(data_dir: Path) -> dict:
    hashes = {name: checksum(data_dir / name) for name in DATA_FILES}
    for name, expected in reference_metadata()["input_sha256"].items():
        if hashes[name] != expected:
            raise ValueError(f"Input differs from the original challenge: {name}")
    return hashes


def load_prepared_data(data_dir: Path) -> PreparedData:
    """Проверяем источники, затем оставляем только события суточного окна."""
    hashes = input_hashes(data_dir)
    train, test, raw = load_data(data_dir)
    if train.window_end_ts.max() > test.window_start_ts.min():
        raise ValueError("Training windows overlap the test period")
    # target намеренно убран: сборка признаков не должна его даже видеть.
    meta = pd.concat([train.drop(columns="target"), test], ignore_index=True)
    events, audit = prepare_events(raw, meta)
    return PreparedData(train, test, meta, events, audit, hashes)


def build_feature_matrices(data: PreparedData) -> FeatureMatrices:
    """Строим признаки с нуля, без чужих моделей или сохранённых кешей.

    Совместная обработка train/test безопасна именно потому, что каждое
    значение считается только внутри своей куки. Нет общих словарей частот,
    target encoding или нормировки по тестовой выборке.
    """
    features = build_advanced_features(data.events, data.meta)
    if len(features.columns) != 472 or any(column in features for column in ("cookie_id", "target")):
        raise AssertionError("Unexpected feature schema")
    return FeatureMatrices(
        train=features.loc[data.train.cookie_id].reset_index(drop=True),
        test=features.loc[data.test.cookie_id].reset_index(drop=True),
    )


def matrix_fingerprint(features: pd.DataFrame) -> str:
    """Короткая проверка всей матрицы: значения и порядок строк, не её сериализация."""
    values = pd.util.hash_pandas_object(features, index=True).to_numpy().tobytes()
    return hashlib.sha256(values).hexdigest()


def validate_submission(submission: pd.DataFrame, test: pd.DataFrame) -> None:
    """Ровно одна строка на тестовую куку, исходный порядок, конечный score 0..1."""
    if submission.columns.tolist() != ["cookie_id", "score"]:
        raise ValueError("Submission must contain exactly cookie_id,score")
    if not submission.cookie_id.is_unique or len(submission) != len(test):
        raise ValueError("Missing or duplicated test cookies")
    if submission.cookie_id.tolist() != test.cookie_id.tolist():
        raise ValueError("Test cookie coverage or order mismatch")
    if not np.isfinite(submission.score).all() or not submission.score.between(0, 1).all():
        raise ValueError("Invalid submission scores")


def fit_final_models(features: pd.DataFrame, y: np.ndarray, output_dir: Path,
                     threads: int = 8) -> FittedEnsemble:
    """Обучаем четыре модели на всём train и сохраняем их в нативных форматах."""
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    models, info, importance = {}, [], []
    for name, spec in MODEL_SPECS.items():
        columns = columns_for(spec, features)
        print(f"Fit {name}: {len(features)} cookies / {len(columns)} features", flush=True)
        model = make_model(spec, threads)
        model.fit(features[columns], y)
        models[name] = model
        if spec["kind"] == "catboost":
            path = model_dir / f"{name}.cbm"
            model.save_model(str(path))
            gain = model.feature_importances_.astype(float)
        else:
            path = model_dir / f"{name}.lgb.txt"
            model.booster_.save_model(str(path))
            gain = model.booster_.feature_importance(importance_type="gain").astype(float)
        info.append({"name": name, "kind": spec["kind"], "path": str(path.relative_to(output_dir)),
                     "columns": columns, "spec": spec, "weight": FINAL_WEIGHTS[name],
                     "model_sha256": checksum(path)})
        # Важности библиотек имеют разный масштаб. Нормируем каждую отдельно;
        # это диагностическая сводка, а не объяснение причин бот-трафика.
        gain /= max(float(gain.sum()), 1e-12)
        importance.append(pd.Series(gain * FINAL_WEIGHTS[name], index=columns))
    combined = pd.concat(importance, axis=1).fillna(0).sum(axis=1).sort_values(ascending=False)
    return FittedEnsemble(models, info, combined)


def predict_ensemble(ensemble: FittedEnsemble, features: pd.DataFrame) -> np.ndarray:
    """Передаём каждой модели её столбцы, затем усредняем вероятности."""
    predictions = {info["name"]: ensemble.models[info["name"]].predict_proba(features[info["columns"]])[:, 1]
                   for info in ensemble.model_info}
    return blend_predictions(predictions, FINAL_WEIGHTS)


def save_submission(test: pd.DataFrame, score: np.ndarray, path: Path) -> pd.DataFrame:
    """Записываем те же 17 значащих цифр, что в отправленном v2."""
    submission = pd.DataFrame({"cookie_id": test.cookie_id, "score": score})
    validate_submission(submission, test)
    if (path.exists() and path.is_dir()) or path.resolve().suffix.lower() != ".csv":
        raise ValueError("Submission path must be a CSV file")
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False, float_format="%.17g")
    validate_submission(pd.read_csv(path, dtype={"cookie_id": str}), test)
    return submission


def assert_reference_submission(path: Path) -> str:
    """Побайтовое совпадение — более строгая проверка, чем близость метрик."""
    actual, expected = checksum(path), reference_metadata()["submission_sha256"]
    if actual != expected:
        raise AssertionError(
            "Submission differs from submitted v2. Check Python/package versions, CPU platform "
            "and the default 8 training threads. Predictions were not replaced by stored answers.")
    return actual


def save_run_metadata(data: PreparedData, features: FeatureMatrices, ensemble: FittedEnsemble,
                      output_dir: Path, submission_path: Path, threads: int) -> dict:
    """Фиксируем происхождение результата, чтобы verify.py мог всё перепроверить."""
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "data_audit.json", {**data.audit, "input_sha256": data.input_hashes})
    write_json(output_dir / "feature_schema.json", {
        "n_features": len(features.train.columns), "columns": features.train.columns.tolist(),
        "train_fingerprint": matrix_fingerprint(features.train),
        "test_fingerprint": matrix_fingerprint(features.test),
    })
    ensemble.importance.rename("importance").rename_axis("feature").to_csv(output_dir / "feature_importance.csv")
    manifest = {
        "recipe": "submitted_v2", "models": ensemble.model_info, "weights": FINAL_WEIGHTS,
        "threads": threads, "python_version": platform.python_version(),
        "machine": platform.machine(), "system": platform.system(),
        "package_versions": {name: version(name) for name in (
            "numpy", "pandas", "scipy", "scikit-learn", "catboost", "lightgbm")},
        "input_sha256": data.input_hashes, "submission_sha256": checksum(submission_path),
        "source_sha256": {path.name: checksum(path) for path in sorted((REPOSITORY_ROOT / "bot_solution").glob("*.py"))},
    }
    write_json(output_dir / "model_manifest.json", manifest)
    return manifest


def reproduce(data_dir: Path, output_dir: Path, submission_path: Path, threads: int = 8,
              validate: bool = False) -> dict:
    """Полный путь запуска. По умолчанию без CV, но всегда с обучением с нуля."""
    started = time.perf_counter()
    protected = [(data_dir / name).resolve() for name in DATA_FILES]
    if submission_path.resolve() in protected:
        raise ValueError("Do not overwrite an input file with a submission")
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Load sources and filter observation windows", flush=True)
    data = load_prepared_data(data_dir)
    print(f"Build 472 features from {len(data.events):,} usable events", flush=True)
    features = build_feature_matrices(data)
    if validate:
        from bot_solution.validation import evaluate_models
        report, predictions = evaluate_models(data.train, features.train, threads)
        write_json(output_dir / "cross_validation.json", report)
        predictions.to_csv(output_dir / "cv_predictions.csv", index=False)
    ensemble = fit_final_models(features.train, data.train.target.to_numpy(dtype=int), output_dir, threads)
    score = predict_ensemble(ensemble, features.test)
    save_submission(data.test, score, submission_path)
    manifest = save_run_metadata(data, features, ensemble, output_dir, submission_path, threads)
    assert_reference_submission(submission_path)
    print(f"Saved {len(data.test)} scores; exact submitted-v2 SHA-256 matches", flush=True)
    print(f"Elapsed: {time.perf_counter() - started:.1f} seconds", flush=True)
    return manifest


def verify_saved_result(data_dir: Path, output_dir: Path, submission_path: Path,
                        threads: int = 8, test_features: pd.DataFrame | None = None) -> dict:
    """Проверяем CSV и заново предсказываем из сохранённых нативных моделей.

    Обычный вызов независимо пересобирает признаки только по test. Ноутбук может
    передать уже пересобранную тестовую матрицу, чтобы не повторять её расчёт.
    joblib/pickle-файлы для восстановления моделей не нужны.
    """
    manifest = json.loads((output_dir / "model_manifest.json").read_text(encoding="utf-8"))
    if input_hashes(data_dir) != manifest["input_sha256"]:
        raise AssertionError("Training inputs changed")
    for name, expected in manifest["source_sha256"].items():
        if checksum(REPOSITORY_ROOT / "bot_solution" / name) != expected:
            raise AssertionError(f"Code changed after training: {name}")
    schema = json.loads((output_dir / "feature_schema.json").read_text(encoding="utf-8"))
    if test_features is None:
        _, test, raw = load_data(data_dir)
        # Здесь нет событий обучающих кук: это отдельная проверка независимости.
        events, _ = prepare_events(raw.loc[raw.cookie_id.isin(test.cookie_id)].copy(), test)
        test_features = build_advanced_features(events, test).reset_index(drop=True)
    else:
        test = pd.read_csv(data_dir / "test.csv", dtype={"cookie_id": str})
    if test_features.columns.tolist() != schema["columns"]:
        raise AssertionError("Test feature schema differs")
    if matrix_fingerprint(test_features) != schema["test_fingerprint"]:
        raise AssertionError("Independent test features differ from the training run")
    submission = pd.read_csv(submission_path, dtype={"cookie_id": str})
    validate_submission(submission, test)
    if checksum(submission_path) != manifest["submission_sha256"]:
        raise AssertionError("Submission changed after training")
    assert_reference_submission(submission_path)
    if len(manifest["models"]) != len(MODEL_SPECS) or manifest["weights"] != FINAL_WEIGHTS:
        raise AssertionError("Final recipe differs from submitted v2")
    predictions = {}
    for info in manifest["models"]:
        path = output_dir / info["path"]
        if checksum(path) != info["model_sha256"]:
            raise AssertionError(f"Saved model changed: {info['name']}")
        if info["kind"] == "catboost":
            model = CatBoostClassifier()
            model.load_model(str(path))
            score = model.predict_proba(test_features[info["columns"]], thread_count=threads)[:, 1]
        elif info["kind"] == "lightgbm":
            model = Booster(model_file=str(path))
            score = model.predict(test_features[info["columns"]], num_threads=threads)
        else:
            raise AssertionError(f"Unsupported saved model: {info['kind']}")
        predictions[info["name"]] = score
    if set(predictions) != set(FINAL_WEIGHTS):
        raise AssertionError("Saved model coverage differs")
    recomputed = blend_predictions(predictions, FINAL_WEIGHTS)
    np.testing.assert_allclose(recomputed, submission.score.to_numpy(), atol=1e-12, rtol=1e-12)
    return {"n_cookies": len(submission), "n_models": len(predictions),
            "n_features": len(test_features.columns), "submission_sha256": checksum(submission_path)}
