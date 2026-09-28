# Боты в истории событий

Основное решение - [solution.ipynb](solution.ipynb): описание подхода,
обработка событий, построение признаков, сравнение моделей и временная
валидация. Функции вынесены в `bot_solution/` и используются также в `run.py`.
Приложенный `submission.csv` соответствует версии v2; сообщённый результат
на скрытом тесте - Precision при Recall ≥ 70% = 0.82278.

## Установка

Для повторения результата нужны Python 3.11.9, Linux x86_64 и исходные
`train.csv`, `test.csv`, `events.csv.gz` в `data/`. События распаковывать не нужно.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Ноутбук

```bash
.venv/bin/python -m pip install -r requirements-notebook.txt
```

Откройте `solution.ipynb` в Jupyter или VS Code, выберите `.venv` как ядро
и выполните Run All. Если данные находятся в другом каталоге, измените
`DATA_DIR` в первой ячейке с кодом.

## Запуск из терминала

```bash
.venv/bin/python run.py
```

Команда заново строит признаки, обучает четыре модели на всём train и записывает
`submission.csv`. Для повторения временной валидации используйте `run.py --validate`.
Другой каталог данных можно указать через `--data-dir /path/to/data`.

Seed моделей: 42; параметры находятся в
`bot_solution/models.py`, версии зависимостей закреплены. Совпадение итогового
CSV с отправленным файлом проверяется по SHA-256 из `metadata/reference.json`.
На другой архитектуре или с другими версиями библиотек последние разряды
scores могут отличаться, и проверка сообщит о несовпадении.

Проверка сохранённых моделей и тесты обработки данных:

```bash
.venv/bin/python verify.py
.venv/bin/python -m unittest discover -s tests -v
```

Использованы open-source NumPy, pandas, SciPy, scikit-learn, CatBoost, LightGBM
и Matplotlib; для ноутбука - Jupyter. Модели работают локально.
Метрика рассчитывается через предоставленный `metric.py`.
Модели и промежуточные результаты создаются в `outputs/`; исходные данные,
окружение и служебные файлы исключены из git.
