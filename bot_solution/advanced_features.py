"""Ещё 222 признака v2: ритм действий, сессии, воронки и геометрия курсора.

Каждая строка считается независимо. Нет обучаемой нормировки, общих частот,
меток других кук или событий за пределами заданного окна.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bot_solution.features import EVENT_TYPES, _summary, build_features


def _entropy(values: np.ndarray) -> float:
    """Шенноновская энтропия в битах: насколько разнообразны наблюдения."""
    if not len(values):
        return np.nan
    _, counts = np.unique(values, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def _cadence(gaps: np.ndarray, prefix: str) -> dict[str, float]:
    """Помимо длины пауз описываем их регулярность относительно своего масштаба.

    MAD и IQR менее чувствительны к отдельным длинным паузам, чем std. Совпадение
    интервалов, энтропия и автокорреляция описывают другие стороны регулярности.
    Знаменатели ограничены снизу единицей: timestamps имеют секундную точность.
    """
    gaps = gaps[np.isfinite(gaps) & (gaps >= 0)]
    result = _summary(gaps, prefix)
    keys = ("cv", "mad_ratio", "iqr_ratio", "q90_q10_ratio", "log_std",
            "entropy", "mode_share", "unique_share", "near_median_share", "autocorrelation")
    result.update({f"{prefix}_{key}": np.nan for key in keys})
    if not len(gaps):
        return result
    mean, std = gaps.mean(), gaps.std()
    q10, q25, median, q75, q90 = np.quantile(gaps, [.1, .25, .5, .75, .9])
    _, counts = np.unique(gaps, return_counts=True)
    result.update({
        f"{prefix}_cv": float(std / max(mean, 1)),
        f"{prefix}_mad_ratio": float(np.median(np.abs(gaps - median)) / max(median, 1)),
        f"{prefix}_iqr_ratio": float((q75 - q25) / max(median, 1)),
        f"{prefix}_q90_q10_ratio": float(q90 / max(q10, 1)),
        f"{prefix}_log_std": float(np.log1p(gaps).std()),
        f"{prefix}_entropy": _entropy(gaps),
        f"{prefix}_mode_share": float(counts.max() / len(gaps)),
        f"{prefix}_unique_share": float(len(counts) / len(gaps)),
        f"{prefix}_near_median_share": float((np.abs(gaps - median) <= max(.2 * median, 1)).mean()),
    })
    if len(gaps) >= 3:
        a, b = gaps[:-1] - gaps[:-1].mean(), gaps[1:] - gaps[1:].mean()
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        if denominator > 0:
            result[f"{prefix}_autocorrelation"] = float(a @ b / denominator)
    return result


def _cookie_features(group: pd.DataFrame) -> dict[str, float]:
    """Сборка всех дополнительных групп для одной отсортированной куки."""
    t = group.relative_seconds.to_numpy(dtype=float)
    names = group.event_name.to_numpy(dtype=str)
    items = group.item_id.to_numpy(dtype=float)
    categories = group.item_category.fillna("").to_numpy(dtype=str)
    locations = group.item_location.fillna("").to_numpy(dtype=str)
    queries = group.search_query.fillna("").to_numpy(dtype=str)
    pages = group.search_page.to_numpy(dtype=float)
    delta = np.diff(t)
    result: dict[str, float] = {}
    result.update(_cadence(delta[delta <= 1800], "cadence_all"))
    for label, mask in (
        ("views", names == "item_view"),
        ("searches", names == "search_results_view"),
        ("contacts", np.char.startswith(names, "contact_")),
    ):
        dt = np.diff(t[mask])
        result.update(_cadence(dt[dt <= 1800], f"cadence_{label}"))
        n = int(mask.sum())
        observed_items = items[mask & np.isfinite(items)]
        result[f"{label}_unique_item_fraction"] = (
            float(len(np.unique(observed_items)) / len(observed_items)) if len(observed_items) else np.nan
        )
        for field, values in (("category", categories), ("location", locations), ("query", queries)):
            observed = values[mask & (values != "")]
            result[f"{label}_{field}_entropy"] = _entropy(observed)
            result[f"{label}_{field}_unique_fraction"] = (
                float(len(np.unique(observed)) / len(observed)) if len(observed) else np.nan
            )
        result[f"{label}_events_per_category"] = (
            float(n / max(len(np.unique(categories[mask & (categories != "")])), 1)) if n else np.nan
        )

    # Проверяем два фиксированных масштаба сессий. Короткая граница отделяет
    # небольшие заходы; длинная сохраняет обычную сессию с паузами.
    for cutoff in (300, 1800):
        starts = np.r_[0, np.flatnonzero(delta > cutoff) + 1]
        ends = np.r_[starts[1:] - 1, len(t) - 1]
        sizes, spans = ends - starts + 1, t[ends] - t[starts]
        result.update(_summary(sizes.astype(float), f"session_{cutoff}_size"))
        result.update(_summary(spans, f"session_{cutoff}_span"))
        regularity = []
        for start, end in zip(starts, ends):
            dt = np.diff(t[start:end + 1])
            if len(dt) >= 2:
                regularity.append(float(dt.std() / max(dt.mean(), 1)))
        result.update(_summary(np.array(regularity), f"session_{cutoff}_cv"))
        result[f"session_{cutoff}_dominant_fraction"] = float(sizes.max() / len(t))
        result[f"session_{cutoff}_activity_rate"] = float(len(t) / max(spans.sum(), 1))
        result[f"session_{cutoff}_short_fraction"] = float((sizes <= 2).mean())

    # Unknown within-second order is not interpreted as a chronological transition.
    increasing = delta > 0
    pair_codes = np.char.add(np.char.add(names[:-1][increasing], ">"), names[1:][increasing])
    result["ordered_transition_entropy"] = _entropy(pair_codes)
    result["ordered_same_action_fraction"] = (
        float((names[:-1][increasing] == names[1:][increasing]).mean()) if increasing.any() else np.nan
    )
    for source in ("item_view", "search_results_view", "photo_swipe"):
        source_mask = (names[:-1] == source) & increasing
        outgoing = int(source_mask.sum())
        for dest in ("item_view", "search_results_view", "photo_swipe", "seller_page_view",
                     "favorite_add", "contact_phone_show", "contact_chat_open"):
            result[f"after_{source}_{dest}_fraction"] = (
                float(((names[1:] == dest) & source_mask).sum() / outgoing) if outgoing else np.nan
            )
    if len(names) >= 3:
        ordered = (delta[:-1] > 0) & (delta[1:] > 0)
        triples = np.char.add(np.char.add(names[:-2], ">"), names[1:-1])
        triples = np.char.add(np.char.add(triples, ">"), names[2:])[ordered]
        result["triple_entropy"] = _entropy(triples)
        result["triple_unique_fraction"] = float(len(np.unique(triples)) / len(triples)) if len(triples) else np.nan
    else:
        result["triple_entropy"] = result["triple_unique_fraction"] = np.nan

    # Воронка считается по множествам объявлений, а не только по числам событий:
    # десять контактов с одним объявлением отличаются от десяти разных контактов.
    valid_item = np.isfinite(items)
    all_items = np.unique(items[valid_item])
    viewed = set(items[valid_item & (names == "item_view")])
    contacts = set(items[valid_item & np.char.startswith(names, "contact_")])
    for label, mask in (
        ("photo", names == "photo_swipe"), ("favorite", names == "favorite_add"),
        ("contact", np.char.startswith(names, "contact_")), ("seller", names == "seller_page_view"),
    ):
        associated = set(items[valid_item & mask])
        result[f"viewed_items_with_{label}_fraction"] = float(len(viewed & associated) / len(viewed)) if viewed else np.nan
        result[f"{label}_items_without_view_fraction"] = (
            float(len(associated - viewed) / len(associated)) if associated else np.nan
        )
    # До контакта отсчитываем время от первого просмотра этого же объявления.
    # Контакты, случившиеся раньше первого просмотра, задержку не определяют.
    per_item_counts, per_item_types, contact_latencies = [], [], []
    for item in all_items:
        mask = items == item
        per_item_counts.append(int(mask.sum()))
        per_item_types.append(len(np.unique(names[mask])))
        if item in viewed and item in contacts:
            first_view = t[mask & (names == "item_view")].min()
            later_contact = t[mask & np.char.startswith(names, "contact_") & (t >= first_view)]
            if len(later_contact):
                contact_latencies.append(float(later_contact.min() - first_view))
    result.update(_summary(np.array(per_item_counts, dtype=float), "per_item_events"))
    result.update(_summary(np.array(per_item_types, dtype=float), "per_item_action_types"))
    result.update(_summary(np.array(contact_latencies), "contact_latency"))
    result["items_with_multiple_actions_fraction"] = (
        float((np.array(per_item_types) > 1).mean()) if per_item_types else np.nan
    )

    search = names == "search_results_view"
    sq, sp, st = queries[search], pages[search], t[search]
    page_diff = np.diff(sp)
    # Переход 3 -> 1 при смене запроса не означает возврат назад. Поэтому
    # пагинацию оцениваем только в пределах одного непустого запроса.
    same_query = (sq[:-1] == sq[1:]) & (sq[:-1] != "") & (np.diff(st) > 0) & np.isfinite(page_diff)
    for label, comparison in (("next", page_diff == 1), ("repeat", page_diff == 0),
                              ("back", page_diff < 0), ("skip", page_diff > 1)):
        result[f"same_query_page_{label}_fraction"] = float(comparison[same_query].mean()) if same_query.any() else np.nan
    result["search_query_switch_fraction"] = float((sq[:-1] != sq[1:]).mean()) if len(sq) > 1 else np.nan
    query_depth, query_page_coverage = [], []
    for query in np.unique(sq[sq != ""]):
        observed = sp[(sq == query) & np.isfinite(sp)]
        if len(observed):
            query_depth.append(float(observed.max()))
            query_page_coverage.append(float(len(np.unique(observed)) / max(observed.max() - observed.min() + 1, 1)))
    result.update(_summary(np.array(query_depth), "per_query_depth"))
    result["per_query_page_coverage_mean"] = float(np.mean(query_page_coverage)) if query_page_coverage else np.nan

    # Дополняем средние координаты: связь осей, вытянутость облака точек,
    # повторяемость шагов и повороты пути. Пропуски сохраняют отдельный смысл.
    xy = group[["pointer_x", "pointer_y"]].to_numpy(dtype=float)
    present = np.isfinite(xy).all(axis=1)
    for event in ("item_view", "search_results_view", "photo_swipe"):
        mask = names == event
        result[f"pointer_present_given_{event}"] = float(present[mask].mean()) if mask.any() else np.nan
    xy, pt = xy[present], t[present]
    for axis, label in ((0, "x"), (1, "y")):
        values = xy[:, axis]
        result[f"pointer_{label}_entropy"] = _entropy(values)
        result[f"pointer_{label}_unique_fraction"] = float(len(np.unique(values)) / len(values)) if len(values) else np.nan
    result["pointer_observation_count"] = int(len(xy))
    result["pointer_xy_correlation"] = np.nan
    result["pointer_spread_anisotropy"] = np.nan
    result["pointer_turn_cosine_mean"] = np.nan
    result["pointer_direction_change_fraction"] = np.nan
    if len(xy) >= 2:
        sx, sy = xy.std(axis=0)
        result["pointer_spread_anisotropy"] = float(min(sx, sy) / max(sx, sy, 1))
        if sx > 0 and sy > 0:
            centered = xy - xy.mean(axis=0)
            result["pointer_xy_correlation"] = float(np.mean(centered[:, 0] * centered[:, 1]) / (sx * sy))
        # Movement statistics only connect observations with distinct timestamps.
        movement = np.diff(xy, axis=0)[np.diff(pt) > 0]
        lengths = np.linalg.norm(movement, axis=1)
        result.update(_cadence(lengths, "pointer_step"))
        if len(movement) >= 2:
            denominator = lengths[:-1] * lengths[1:]
            usable = denominator > 0
            if usable.any():
                cosine = (movement[:-1][usable] * movement[1:][usable]).sum(axis=1) / denominator[usable]
                result["pointer_turn_cosine_mean"] = float(cosine.mean())
                result["pointer_direction_change_fraction"] = float((cosine < 0).mean())
    else:
        result.update(_cadence(np.array([]), "pointer_step"))
    return {f"v2_{key}": value for key, value in result.items()}


def build_advanced_features(events: pd.DataFrame, meta: pd.DataFrame,
                            base: pd.DataFrame | None = None) -> pd.DataFrame:
    """Добавляем v2-столбцы к базовым 250, сохраняя порядок кук и признаков.

    Аргумент base полезен для независимых проверок на маленьких примерах.
    Основной запуск строит всё с нуля и не читает старые кеши признаков.
    """
    ev = events.copy()
    ev["relative_seconds"] = (ev.event_ts - ev.window_start_ts).dt.total_seconds()
    records = {cookie: _cookie_features(group) for cookie, group in ev.groupby("cookie_id", sort=False)}
    additional = pd.DataFrame.from_dict(records, orient="index")
    index = pd.Index(meta.cookie_id, name="cookie_id")
    if base is None:
        base = build_features(events, meta)
    if not base.index.equals(index):
        raise ValueError("Base features must preserve metadata cookie order")
    additional = additional.reindex(index).replace([np.inf, -np.inf], np.nan).fillna(-1).astype(float)
    additional["v2_pointer_observation_count"] = additional["v2_pointer_observation_count"].clip(lower=0)
    return base.join(additional).astype(float)
