"""Исходные 250 признаков v1, которые также входят в финальное решение v2.

Здесь собраны проверки входных данных и базовые агрегаты одной куки. Сырые
cookie_id/item_id не становятся значениями признаков, target не используется.
Численные формулы сохранены: их изменение изменило бы отправленный результат.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


EVENT_TYPES = (
    "search_results_view", "item_view", "photo_swipe", "seller_page_view",
    "contact_phone_show", "contact_chat_open", "contact_message_sent",
    "favorite_add", "login", "captcha_shown",
)
META_DATES = ("cookie_created_at", "window_start_ts", "window_end_ts")
EVENT_COLUMNS = (
    "cookie_id", "event_ts", "eid", "event_name", "platform", "user_agent",
    "item_id", "item_category", "item_location", "seller_type", "search_query",
    "search_page", "pointer_x", "pointer_y",
)


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Читаем исходные файлы, проверяем уникальность кук, даты и схему событий.

    Сжатые события pandas читает напрямую. Сразу переводим даты в timestamps:
    сравнивать их как строки или незаметно превращать ошибки в NaT было бы
    опасно для границ наблюдения.
    """
    train = pd.read_csv(data_dir / "train.csv", dtype={"cookie_id": "str"})
    test = pd.read_csv(data_dir / "test.csv", dtype={"cookie_id": "str"})
    for frame, name in ((train, "train"), (test, "test")):
        if frame.cookie_id.isna().any() or not frame.cookie_id.is_unique:
            raise ValueError(f"{name}: cookie_id must be present and unique")
        for col in META_DATES:
            frame[col] = pd.to_datetime(frame[col], format="mixed", errors="raise")
        if frame[list(META_DATES)].isna().any().any():
            raise ValueError(f"{name}: missing metadata timestamps")
        if (frame.window_end_ts <= frame.window_start_ts).any():
            raise ValueError(f"{name}: invalid observation window")
        if (frame.cookie_created_at > frame.window_end_ts).any():
            raise ValueError(f"{name}: cookie created after observation window")
    if not train.target.isin([0, 1]).all() or "target" in test:
        raise ValueError("Invalid target schema")
    if set(train.cookie_id) & set(test.cookie_id):
        raise ValueError("Train and test cookies overlap")
    events = pd.read_csv(data_dir / "events.csv.gz", dtype={"cookie_id": "str"})
    if set(events.columns) != set(EVENT_COLUMNS):
        raise ValueError("Unexpected event schema")
    events.event_ts = pd.to_datetime(events.event_ts, format="mixed", errors="raise")
    if events[["cookie_id", "event_ts", "event_name"]].isna().any().any():
        raise ValueError("Missing event key, type or timestamp")
    return train, test, events


def prepare_events(events: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Сначала границы окна, затем точные дубликаты, нормализация и сортировка.

    Окно полуоткрытое: событие ровно в window_end_ts уже недоступно модели.
    many_to_one не даст случайно размножить события при соединении с метаданными.
    Аудит сохраняем отдельно, чтобы проверяющий видел, сколько строк исключено.
    """
    if not meta.cookie_id.is_unique:
        raise ValueError("Metadata must have one window per cookie")
    joined = events.merge(
        meta[["cookie_id", *META_DATES]], on="cookie_id", how="left",
        validate="many_to_one", indicator=True,
    )
    if joined._merge.ne("both").any():
        raise ValueError("Events contain cookies absent from metadata")
    before = joined.event_ts.lt(joined.window_start_ts)
    after = joined.event_ts.ge(joined.window_end_ts)
    in_window = joined.loc[~before & ~after].drop(columns="_merge")
    audit = {
        "raw_events": int(len(events)),
        "raw_exact_duplicates": int(events.duplicated().sum()),
        "events_before_window": int(before.sum()),
        "events_at_or_after_window_end": int(after.sum()),
        "in_window_events_before_deduplication": int(len(in_window)),
        "in_window_exact_duplicates": int(in_window.duplicated(subset=list(EVENT_COLUMNS)).sum()),
        "raw_missing_fraction": events.isna().mean().round(6).to_dict(),
        "events_before_cookie_creation_in_window": int(
            in_window.event_ts.lt(in_window.cookie_created_at).sum()
        ),
    }
    ev = in_window.drop_duplicates(subset=list(EVENT_COLUMNS)).copy()
    for col in ("event_name", "platform", "item_category", "item_location", "seller_type"):
        ev[col] = ev[col].astype("string").str.strip().str.lower().replace("", pd.NA)
    ev["platform"] = ev.platform.replace({"desktop": "web", "iphone": "ios"})
    ev["search_query"] = (
        ev.search_query.astype("string").str.strip().str.lower()
        .str.replace(r"\s+", " ", regex=True).replace("", pd.NA)
    )
    ev["user_agent"] = ev.user_agent.fillna("").astype(str).str.strip().str.lower()
    for col in ("item_id", "search_page", "pointer_x", "pointer_y"):
        ev[col] = pd.to_numeric(ev[col], errors="raise")
    # Tie order is independent of the shuffled source file. It does not imply
    # that an arbitrary order of simultaneous actions is a causal sequence.
    ev = ev.sort_values(
        ["cookie_id", "event_ts", "event_name", "item_id", "search_page",
         "platform", "user_agent", "pointer_x", "pointer_y", "item_category",
         "item_location", "seller_type", "search_query", "eid"],
        kind="mergesort", na_position="last",
    ).reset_index(drop=True)
    audit["usable_events"] = int(len(ev))
    audit["cookies_without_events"] = int(len(meta) - ev.cookie_id.nunique())
    audit["event_types"] = ev.event_name.value_counts().to_dict()
    audit["normalized_platforms"] = ev.platform.value_counts().to_dict()
    audit["usable_event_date_min"] = str(ev.event_ts.min())
    audit["usable_event_date_max"] = str(ev.event_ts.max())
    return ev, audit


def parse_user_agent(value: str) -> tuple[str, str, bool]:
    """Оставляем широкие семейства браузеров/ОС, не учим модель на версиях UA.

    Слова вроде headless/python дают обычный текстовый индикатор автоматизации,
    а не готовую метку: модель обучается на всех признаках вместе.
    """
    ua = value.lower()
    automated = bool(re.search(r"headless|python|curl|wget|scrapy|selenium|httpclient|bot", ua))
    if "headless" in ua:
        family = "headless"
    elif "yabrowser" in ua:
        family = "yandex"
    elif "edg/" in ua:
        family = "edge"
    elif "firefox" in ua:
        family = "firefox"
    elif "chrome" in ua or "crios" in ua:
        family = "chrome"
    elif "safari" in ua:
        family = "safari"
    else:
        family = "other"
    if "android" in ua:
        os = "android"
    elif "iphone" in ua or "ipad" in ua:
        os = "ios"
    elif "windows" in ua:
        os = "windows"
    elif "macintosh" in ua:
        os = "mac"
    elif "linux" in ua or "x11" in ua:
        os = "linux"
    else:
        os = "other"
    return family, os, automated


def _distribution_features(ev: pd.DataFrame, col: str) -> dict[str, pd.Series]:
    """Разнообразие внутри куки: число значений, главная доля, энтропия, HHI.

    Сначала считаем cookie_id + значение, затем вероятности внутри этой же
    куки. Частот по общей обучающей или тестовой выборке здесь нет.
    """
    counts = ev.groupby(["cookie_id", col], observed=True).size()
    totals = counts.groupby(level=0).transform("sum")
    p = counts / totals
    return {
        f"{col}_nunique": counts.groupby(level=0).size(),
        f"{col}_top_share": p.groupby(level=0).max(),
        f"{col}_entropy": (-(p * np.log2(p))).groupby(level=0).sum(),
        f"{col}_concentration": p.pow(2).groupby(level=0).sum(),
    }


def _summary(values: np.ndarray, prefix: str) -> dict[str, float]:
    """Короткое описание массива. Неопределённые статистики пока остаются NaN.

    Для этих последовательных агрегатов std — numpy.std с ddof=0. pandas-статистики
    в build_features используют своё стандартное ddof=1; менять это при упаковке
    готовой модели нельзя, поэтому финальное заполнение пропусков общее.
    """
    values = values[np.isfinite(values)]
    if not len(values):
        return {f"{prefix}_{s}": np.nan for s in ("mean", "std", "median", "q10", "q90", "max")}
    quantiles = np.quantile(values, [0.1, 0.5, 0.9])
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std()),
        f"{prefix}_median": float(quantiles[1]),
        f"{prefix}_q10": float(quantiles[0]),
        f"{prefix}_q90": float(quantiles[2]),
        f"{prefix}_max": float(values.max()),
    }


def _sequence_features(group: pd.DataFrame) -> dict[str, float]:
    """Временной ритм, сессии, действия, поиск и движение курсора одной куки.

    group уже отсортирован. Все временные расстояния измеряются в секундах,
    а относительное время отсчитывается от начала наблюдения, не от epoch.
    """
    t = group.relative_seconds.to_numpy(dtype=float)
    delta = np.diff(t)
    result = _summary(delta, "gap")
    local_delta = delta[delta <= 1800]
    result.update(_summary(local_delta, "within_session_gap"))
    result["within_session_gap_cv"] = (
        float(local_delta.std() / max(local_delta.mean(), 1)) if len(local_delta) else np.nan
    )
    result["gap_cv"] = float(delta.std() / max(delta.mean(), 1)) if len(delta) else np.nan
    result["gap_relative_iqr"] = (
        float((np.quantile(delta, 0.75) - np.quantile(delta, 0.25)) / max(np.median(delta), 1))
        if len(delta) else np.nan
    )
    for threshold in (0, 1, 3, 10, 30, 60, 300, 1800):
        result[f"gap_le_{threshold}_share"] = float((delta <= threshold).mean()) if len(delta) else np.nan
    if len(delta):
        _, frequency = np.unique(delta, return_counts=True)
        result["gap_top_share"] = float(frequency.max() / len(delta))
        result["gap_unique_share"] = float(len(frequency) / len(delta))
    result["first_event_seconds"] = float(t[0])
    result["last_event_seconds"] = float(t[-1])
    result["span_seconds"] = float(t[-1] - t[0])
    result["events_per_span_minute"] = float(len(t) / max((t[-1] - t[0]) / 60, 1))
    # Разрыв более 30 минут начинает новую сессию. Это фиксированная эвристика,
    # не параметр, который подбирается на скрытом тесте.
    session_starts = np.r_[0, np.flatnonzero(delta > 1800) + 1]
    session_ends = np.r_[session_starts[1:] - 1, len(t) - 1]
    session_sizes = session_ends - session_starts + 1
    session_spans = t[session_ends] - t[session_starts]
    result["session_count"] = int(len(session_sizes))
    result["session_events_max"] = int(session_sizes.max())
    result["session_events_mean"] = float(session_sizes.mean())
    result["session_single_event_share"] = float((session_sizes == 1).mean())
    result["session_span_max"] = float(session_spans.max())
    result["session_span_sum"] = float(session_spans.sum())

    # Базовые переходы и длины серий помогают отличить однообразный обход
    # объявлений от смешанного поведения. Для нулевых пауз порядок определяется
    # канонической сортировкой, а не порядком строк файла; это не причинный порядок.
    names = group.event_name.to_numpy(dtype=str)
    transitions = np.char.add(np.char.add(names[:-1], ">"), names[1:])
    result["same_action_share"] = float((names[1:] == names[:-1]).mean()) if len(delta) else np.nan
    runs = np.diff(np.r_[0, np.flatnonzero(names[1:] != names[:-1]) + 1, len(names)])
    result["action_run_max"] = int(runs.max())
    result["action_run_mean"] = float(runs.mean())
    result["action_run_count"] = int(len(runs))
    if len(transitions):
        _, counts = np.unique(transitions, return_counts=True)
        probabilities = counts / counts.sum()
        result["transition_entropy"] = float(-(probabilities * np.log2(probabilities)).sum())
    for source, dest in (
        ("item_view", "item_view"), ("search_results_view", "search_results_view"),
        ("search_results_view", "item_view"), ("item_view", "search_results_view"),
        ("item_view", "photo_swipe"), ("photo_swipe", "photo_swipe"),
        ("item_view", "contact_phone_show"), ("item_view", "contact_chat_open"),
        ("item_view", "seller_page_view"), ("item_view", "favorite_add"),
        ("contact_chat_open", "contact_message_sent"), ("captcha_shown", "item_view"),
    ):
        result[f"transition_{source}_to_{dest}_share"] = (
            float((transitions == f"{source}>{dest}").mean()) if len(transitions) else np.nan
        )

    views = group.loc[group.event_name.eq("item_view")]
    item_views = views.item_id.dropna().to_numpy()
    result["item_view_repeat_share"] = (
        float(1 - len(np.unique(item_views)) / len(item_views)) if len(item_views) else np.nan
    )
    result.update(_summary(np.diff(views.relative_seconds.to_numpy(dtype=float)), "item_view_gap"))
    for col in ("item_category", "item_location", "platform", "user_agent"):
        values = group[col].fillna("missing").to_numpy(dtype=str)
        result[f"{col}_switch_share"] = float((values[1:] != values[:-1]).mean()) if len(delta) else np.nan

    # Поиск: глубокая выдача, повтор страницы и возврат назад. Дополнение v2
    # ниже отдельно проверяет совпадение запроса, чтобы не путать сброс новой выдачи.
    search = group.loc[group.search_page.notna()]
    page_delta = np.diff(search.search_page.to_numpy(dtype=float))
    for label, condition in (
        ("next", page_delta == 1), ("repeat", page_delta == 0),
        ("back", page_delta < 0), ("skip", page_delta > 1),
    ):
        result[f"search_page_{label}_share"] = float(condition.mean()) if len(page_delta) else np.nan
    result.update(_summary(np.diff(search.relative_seconds.to_numpy(dtype=float)), "search_gap"))

    # Курсор виден не на каждой платформе. Считаем геометрию только там,
    # где есть обе координаты; отсутствующий курсор не заменяем координатой (0, 0).
    pointer = group.loc[group.pointer_x.notna() & group.pointer_y.notna()]
    xy = pointer[["pointer_x", "pointer_y"]].to_numpy(dtype=float)
    movement = np.diff(xy, axis=0)
    distance = np.linalg.norm(movement, axis=1)
    result.update(_summary(distance, "pointer_distance"))
    result["pointer_unique_share"] = float(len(np.unique(xy, axis=0)) / len(xy)) if len(xy) else np.nan
    result["pointer_stationary_share"] = float((distance == 0).mean()) if len(distance) else np.nan
    result["pointer_axis_aligned_share"] = (
        float(((movement[:, 0] == 0) | (movement[:, 1] == 0)).mean()) if len(distance) else np.nan
    )
    pointer_delta = np.diff(pointer.relative_seconds.to_numpy(dtype=float))
    result.update(_summary(distance[pointer_delta > 0] / pointer_delta[pointer_delta > 0], "pointer_speed"))
    result["pointer_path_efficiency"] = (
        float(np.linalg.norm(xy[-1] - xy[0]) / distance.sum())
        if len(distance) and distance.sum() > 0 else np.nan
    )
    return result


def build_features(ev: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Одна числовая строка на куку, в исходном порядке метаданных.

    Разные семейства признаков вычисляются отдельно и соединяются по индексу.
    В конце возвращаем и куки без событий: отсутствие активности — допустимый
    случай, а не причина потерять строку в submission.
    """
    ev = ev.copy()
    ev["relative_seconds"] = (ev.event_ts - ev.window_start_ts).dt.total_seconds()
    ev["hour"] = ev.event_ts.dt.hour
    ev["second"] = ev.event_ts.dt.second
    ev["minute_bucket"] = (ev.relative_seconds // 60).astype(int)
    ev["five_minute_bucket"] = (ev.relative_seconds // 300).astype(int)
    ua_map = {value: parse_user_agent(value) for value in ev.user_agent.unique()}
    ev["ua_family"] = ev.user_agent.map(lambda v: ua_map[v][0])
    ev["ua_os"] = ev.user_agent.map(lambda v: ua_map[v][1])
    ev["ua_automated"] = ev.user_agent.map(lambda v: ua_map[v][2]).astype(float)
    g = ev.groupby("cookie_id", sort=False, observed=True)
    count = g.size()
    f: dict[str, pd.Series] = {"n_events": count}
    # Абсолютные числа и доли дополняют друг друга: сто контактов из тысячи
    # событий и один контакт из десяти имеют одну долю, но разный объём.
    for event in EVENT_TYPES:
        n = ev.event_name.eq(event).groupby(ev.cookie_id).sum()
        f[f"event_{event}_count"] = n
        f[f"event_{event}_share"] = n / count
    for column in ("item_id", "item_category", "item_location", "seller_type",
                   "search_query", "event_name", "platform", "user_agent", "ua_family", "ua_os", "second"):
        f.update(_distribution_features(ev, column))
    # Raw strings/IDs are used only for within-cookie uniqueness/concentration.
    # Their actual values never become model inputs or shared target encodings.
    for column in ("item_id", "item_category", "item_location", "seller_type",
                   "search_query", "search_page", "pointer_x", "pointer_y"):
        f[f"{column}_present_share"] = g[column].count() / count
    for column, levels in (
        ("platform", ("web", "android", "ios")),
        ("ua_family", ("chrome", "firefox", "yandex", "edge", "safari", "headless", "other")),
        ("ua_os", ("windows", "mac", "linux", "android", "ios", "other")),
        ("seller_type", ("private", "pro")),
    ):
        for level in levels:
            f[f"{column}_{level}_share"] = ev[column].eq(level).fillna(False).groupby(ev.cookie_id).mean()
    f["ua_automated_share"] = g.ua_automated.mean()
    mismatch = (
        (ev.platform.eq("ios") & ev.ua_os.ne("ios"))
        | (ev.platform.eq("android") & ev.ua_os.ne("android"))
        | (ev.platform.eq("web") & ev.ua_os.isin(["android", "ios"]))
    )
    f["ua_platform_mismatch_share"] = mismatch.fillna(False).groupby(ev.cookie_id).mean()
    web = ev.loc[ev.platform.eq("web")]
    for column in ("pointer_x", "pointer_y"):
        f[f"web_{column}_present_share"] = web[column].notna().groupby(web.cookie_id).mean()

    # Для непрерывных полей нужны не только средние, но и разброс/диапазон.
    # Например, координаты с малой вариативностью могут быть полезны вместе
    # с ритмом действий, но сами по себе не доказывают, что это бот.
    for column in ("search_page", "pointer_x", "pointer_y"):
        stats = g[column].agg(["mean", "std", "min", "max", "median", "nunique"])
        for stat in stats:
            f[f"{column}_{stat}"] = stats[stat]
        f[f"{column}_range"] = stats["max"] - stats["min"]
    for name, flag in (
        ("search_first_page_share", ev.search_page.eq(1)),
        ("search_deep_page_share", ev.search_page.gt(3)),
        ("night_event_share", ev.hour.lt(6)),
        ("contact_event_share", ev.event_name.str.startswith("contact_")),
        ("engagement_event_share", ev.event_name.isin(["favorite_add", "photo_swipe", "login", "contact_message_sent"])),
    ):
        f[name] = flag.groupby(ev.cookie_id).mean()
    # Текст запроса описываем длиной и составом. Сам текст не превращается
    # в категорию модели и не сопоставляется с метками других пользователей.
    query = ev.loc[ev.search_query.notna()].copy()
    query["query_length"] = query.search_query.str.len().astype(float)
    query["query_words"] = query.search_query.str.split().str.len().astype(float)
    query["query_digits"] = query.search_query.str.count(r"\d").astype(float) / query.query_length.clip(lower=1)
    query["query_cyrillic"] = query.search_query.str.count(r"[а-яё]").astype(float) / query.query_length.clip(lower=1)
    for column in ("query_length", "query_words", "query_digits", "query_cyrillic"):
        stats = query.groupby("cookie_id")[column].agg(["mean", "std", "max"])
        for stat in stats:
            f[f"{column}_{stat}"] = stats[stat]
    # Концентрация по временным корзинам: активность в течение всего окна
    # и короткий всплеск могут иметь одинаковое число событий.
    for bucket in ("hour", "minute_bucket", "five_minute_bucket"):
        bins = ev.groupby(["cookie_id", bucket]).size()
        by_cookie = bins.groupby(level=0)
        f[f"{bucket}_active_count"] = by_cookie.size()
        f[f"{bucket}_events_max"] = by_cookie.max()
        f[f"{bucket}_events_mean"] = by_cookie.mean()
        f[f"{bucket}_top_share"] = by_cookie.max() / count
        probabilities = bins / by_cookie.transform("sum")
        f[f"{bucket}_entropy"] = (-(probabilities * np.log2(probabilities))).groupby(level=0).sum()

    sequence = pd.DataFrame.from_dict(
        {cookie: _sequence_features(group) for cookie, group in g}, orient="index",
    )
    base = pd.DataFrame(f)
    full = base.join(sequence)
    index = pd.Index(meta.cookie_id, name="cookie_id")
    full = full.reindex(index)
    count_columns = [c for c in full if c == "n_events" or c.endswith(("_count", "_nunique"))]
    for column in count_columns:
        full[column] = full[column].fillna(0)
    full["has_events"] = full.n_events.gt(0).astype(float)
    age_days = (meta.window_start_ts - meta.cookie_created_at).dt.total_seconds().to_numpy() / 86400
    full["cookie_age_days"] = np.maximum(age_days, 0)
    full["cookie_age_log1p"] = np.log1p(np.maximum(age_days, 0))
    full["cookie_created_within_window"] = (meta.cookie_created_at >= meta.window_start_ts).to_numpy(dtype=float)
    # Абсолютной календарной даты нет, но сохранён день недели из исходной v2.
    full["window_weekday"] = meta.window_start_ts.dt.dayofweek.to_numpy(dtype=float)
    for num, den, name in (
        ("item_id_nunique", "n_events", "unique_items_per_event"),
        ("search_query_nunique", "event_search_results_view_count", "unique_queries_per_search"),
        ("event_contact_phone_show_count", "event_item_view_count", "phone_per_item_view"),
        ("event_photo_swipe_count", "event_item_view_count", "photos_per_item_view"),
        ("event_favorite_add_count", "event_item_view_count", "favorites_per_item_view"),
        ("event_contact_message_sent_count", "event_contact_chat_open_count", "messages_per_chat"),
    ):
        full[name] = full[num] / full[den].replace(0, np.nan)
    full["log1p_events"] = np.log1p(full.n_events)
    full["log1p_unique_items"] = np.log1p(full.item_id_nunique.clip(lower=0))
    full = full.replace([np.inf, -np.inf], np.nan).astype(float)
    # Counts of absent actions are zero; undefined statistics use a separate
    # sentinel, rather than suggesting a measured zero interval or cursor speed.
    return full.fillna(-1)
