"""
Модуль сравнения научных школ по тематическим профилям.
Основная метрика - коэффициент силуэта (Silhouette Score).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances


# ==============================================================================
# ТИПЫ И КОНСТАНТЫ
# ==============================================================================

DistanceMetric = Literal[
    "euclidean_orthogonal",      # Евклидово в прямоугольном базисе
    "cosine_orthogonal",         # Косинусное в прямоугольном базисе
    "euclidean_oblique",         # Евклидово в косоугольном базисе
    "cosine_oblique"             # Косинусное в косоугольном базисе
]

ComparisonScope = Literal["direct", "all"]

DISTANCE_METRIC_LABELS: Dict[DistanceMetric, str] = {
    "euclidean_orthogonal": "Евклидово (прямоугольный базис)",
    "cosine_orthogonal": "Косинусное (прямоугольный базис)",
    "euclidean_oblique": "Евклидово (косоугольный базис)",
    "cosine_oblique": "Косинусное (косоугольный базис)",
}

SCOPE_LABELS: Dict[ComparisonScope, str] = {
    "direct": "Только прямые диссертанты",
    "all": "Все поколения диссертантов",
}


# ==============================================================================
# РАБОТА С ИЕРАРХИЕЙ КЛАССИФИКАТОРА
# ==============================================================================

def get_code_depth(code: str) -> int:
    """Возвращает глубину (уровень) кода в иерархии."""
    if not code:
        return 0
    return code.count(".") + 1


def get_parent_code(code: str) -> Optional[str]:
    """Возвращает родительский код или None для корневых."""
    if "." not in code:
        return None
    return code.rsplit(".", 1)[0]


def get_ancestor_codes(code: str) -> List[str]:
    """Возвращает список всех предков кода (от корня к текущему)."""
    ancestors = []
    current = code
    while current:
        ancestors.insert(0, current)
        current = get_parent_code(current)
    return ancestors


def is_descendant_of(code: str, ancestor: str) -> bool:
    """
    Проверяет, является ли code потомком ancestor.
    Возвращает True если code == ancestor или code начинается с ancestor.
    """
    if code == ancestor:
        return True
    return code.startswith(ancestor + ".")


def filter_columns_by_nodes(
    columns: List[str],
    selected_nodes: Optional[List[str]] = None
) -> List[str]:
    """
    Фильтрует колонки по выбранным узлам.

    Если selected_nodes is None — возвращает все колонки (весь базис).
    Если указаны узлы — возвращает эти узлы и ВСЕ их потомки на любой глубине.
    """
    if selected_nodes is None or len(selected_nodes) == 0:
        return columns

    filtered = []
    for col in columns:
        for node in selected_nodes:
            if is_descendant_of(col, node):
                filtered.append(col)
                break

    return filtered


def get_nodes_at_level(columns: List[str], level: int) -> List[str]:
    """Возвращает узлы указанного уровня."""
    return sorted(set(col for col in columns if get_code_depth(col) == level))


def get_selectable_nodes(columns: List[str], max_level: int = 3) -> List[str]:
    """Возвращает узлы уровней 1..max_level для выбора пользователем."""
    result = []
    for level in range(1, max_level + 1):
        result.extend(get_nodes_at_level(columns, level))
    return sorted(result)


# ==============================================================================
# ПОСТРОЕНИЕ МАТРИЦЫ ТРАНСФОРМАЦИИ ДЛЯ КОСОУГОЛЬНОГО БАЗИСА
# ==============================================================================

def build_oblique_transform_matrix(
    feature_columns: List[str],
    decay_factor: float = 0.5
) -> np.ndarray:
    """
    Строит матрицу трансформации для косоугольного базиса.
    """
    n = len(feature_columns)
    col_to_idx = {col: i for i, col in enumerate(feature_columns)}

    transform = np.eye(n)

    for i, col in enumerate(feature_columns):
        ancestors = get_ancestor_codes(col)
        for depth, ancestor in enumerate(ancestors[:-1]):
            if ancestor in col_to_idx:
                j = col_to_idx[ancestor]
                distance = len(ancestors) - depth - 1
                weight = decay_factor ** distance
                transform[i, j] = weight

    return transform


def apply_oblique_transform(
    data: np.ndarray,
    feature_columns: List[str],
    decay_factor: float = 0.5
) -> np.ndarray:
    """Применяет трансформацию косоугольного базиса к данным."""
    transform = build_oblique_transform_matrix(feature_columns, decay_factor)
    return data @ transform.T


# ==============================================================================
# ВЫЧИСЛЕНИЕ РАССТОЯНИЙ
# ==============================================================================

def compute_distance_matrix(
    data: np.ndarray,
    feature_columns: List[str],
    metric: DistanceMetric,
    decay_factor: float = 0.5
) -> np.ndarray:
    """Вычисляет матрицу расстояний между образцами."""
    if metric in ("euclidean_oblique", "cosine_oblique"):
        data = apply_oblique_transform(data, feature_columns, decay_factor)

    if metric in ("euclidean_orthogonal", "euclidean_oblique"):
        return euclidean_distances(data)
    else:
        return cosine_distances(data)


# ==============================================================================
# ЗАГРУЗКА ДАННЫХ
# ==============================================================================

def load_scores_from_folder(
    folder_path: str = "basic_scores",
    specific_files: Optional[List[str]] = None
) -> pd.DataFrame:
    """Загружает данные тематических профилей из CSV файлов."""
    base = Path(folder_path).expanduser().resolve()

    if specific_files:
        files = [base / f for f in specific_files if (base / f).exists()]
    else:
        files = sorted(base.glob("*.csv"))

    if not files:
        raise FileNotFoundError(f"CSV файлы не найдены в {base}")

    frames: List[pd.DataFrame] = []
    for file in files:
        try:
            frame = pd.read_csv(file)
            if "Code" not in frame.columns:
                raise KeyError(f"Файл {file.name} не содержит колонку 'Code'")
            frames.append(frame)
        except Exception as e:
            print(f"Ошибка при загрузке {file}: {e}")
            continue

    if not frames:
        raise ValueError("Не удалось загрузить ни один файл")

    scores = pd.concat(frames, ignore_index=True)
    scores = scores.dropna(subset=["Code"])
    scores["Code"] = scores["Code"].astype(str).str.strip()
    scores = scores[scores["Code"].str.len() > 0]
    scores = scores.drop_duplicates(subset=["Code"], keep="first")

    feature_columns = [c for c in scores.columns if c != "Code"]
    scores[feature_columns] = scores[feature_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    scores[feature_columns] = scores[feature_columns].fillna(0.0)

    return scores


def get_feature_columns(scores: pd.DataFrame) -> List[str]:
    """Возвращает список колонок с признаками (коды классификатора)."""
    return [c for c in scores.columns if c != "Code"]


# ==============================================================================
# СБОР ДАННЫХ ДЛЯ НАУЧНЫХ ШКОЛ
# ==============================================================================

def gather_school_dataset(
    df: pd.DataFrame,
    index: Dict[str, Set[int]],
    root: str,
    scores: pd.DataFrame,
    scope: ComparisonScope,
    lineage_func: Callable,
    rows_for_func: Callable,
    author_column: str = "candidate.name",
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Собирает данные тематических профилей для научной школы.

    Логика:
    1. Используем lineage_func или rows_for_func чтобы получить диссертации школы из df
    2. Извлекаем коды (Code) этих диссертаций
    3. По этим кодам находим соответствующие профили в scores

    Args:
        df: Основной DataFrame с диссертациями (из db_lineages)
        index: Индекс для поиска по именам
        root: Имя руководителя (корень школы)
        scores: DataFrame с тематическими профилями (из basic_scores)
        scope: "direct" - только прямые диссертанты, "all" - все поколения
        lineage_func: Функция построения генеалогии (возвращает (graph, subset))
        rows_for_func: Функция поиска строк по имени (возвращает DataFrame)
        author_column: Название колонки с именем автора

    Returns:
        (dataset, missing_info, total_count)
    """
    # Шаг 1: Получаем диссертации школы из основного df
    if scope == "direct":
        subset = rows_for_func(df, index, root)
    elif scope == "all":
        _, subset = lineage_func(df, index, root)
    else:
        raise ValueError(f"Неизвестный scope: {scope}")

    # Проверяем, что есть данные
    if subset is None or subset.empty:
        empty = pd.DataFrame(columns=["Code", "school", author_column])
        return empty, empty, 0

    # Шаг 2: Извлекаем коды диссертаций
    if "Code" not in subset.columns:
        raise KeyError("В данных отсутствует колонка 'Code'")

    cols_to_keep = ["Code"]
    if author_column in subset.columns:
        cols_to_keep.append(author_column)

    working = subset[cols_to_keep].copy()
    working["Code"] = working["Code"].astype(str).str.strip()
    working = working[working["Code"].str.len() > 0]
    working = working.drop_duplicates(subset=["Code"])

    if working.empty:
        empty = pd.DataFrame(columns=["Code", "school", author_column])
        return empty, empty, 0

    codes = working["Code"].tolist()
    total_count = len(codes)

    # Шаг 3: Находим профили по кодам
    scores_copy = scores.copy()
    scores_copy["Code"] = scores_copy["Code"].astype(str).str.strip()

    matched_scores = scores_copy[scores_copy["Code"].isin(codes)].copy()

    if matched_scores.empty:
        missing_info = working.copy()
        missing_info["school"] = root
        empty = pd.DataFrame(columns=list(scores.columns) + ["school", author_column])
        return empty, missing_info, total_count

    # Шаг 4: Добавляем метаданные
    matched_scores["school"] = root

    if author_column in working.columns:
        matched_scores = matched_scores.merge(
            working[["Code", author_column]],
            on="Code",
            how="left"
        )
    else:
        matched_scores[author_column] = None

    # Шаг 5: Определяем пропущенные профили
    found_codes = set(matched_scores["Code"].tolist())
    missing_codes = [c for c in codes if c not in found_codes]

    if missing_codes:
        missing_info = working[working["Code"].isin(missing_codes)].copy()
        missing_info["school"] = root
    else:
        missing_info = pd.DataFrame(columns=["Code", "school", author_column])

    return matched_scores, missing_info, total_count


# ==============================================================================
# ВЫЧИСЛЕНИЕ СИЛУЭТА
# ==============================================================================

def compute_silhouette_analysis(
    datasets: Dict[str, pd.DataFrame],
    feature_columns: List[str],
    metric: DistanceMetric,
    selected_nodes: Optional[List[str]] = None,
    decay_factor: float = 0.5,
) -> Tuple[float, np.ndarray, np.ndarray, List[str], List[str]]:
    """Вычисляет анализ силуэта для сравнения научных школ."""
    used_columns = filter_columns_by_nodes(feature_columns, selected_nodes)

    if not used_columns:
        raise ValueError("Нет колонок для анализа после фильтрации")

    all_data = []
    all_labels = []
    school_order = []

    for school_name, dataset in datasets.items():
        if dataset.empty:
            continue

        available_cols = [c for c in used_columns if c in dataset.columns]
        if not available_cols:
            continue

        school_data = dataset[available_cols].fillna(0.0).values

        if school_data.shape[0] > 0:
            all_data.append(school_data)
            all_labels.extend([len(school_order)] * school_data.shape[0])
            school_order.append(school_name)

    if len(school_order) < 2:
        raise ValueError("Необходимо минимум 2 школы с данными для сравнения")

    X = np.vstack(all_data)
    labels = np.array(all_labels)

    if X.shape[0] < 2:
        raise ValueError("Недостаточно образцов для анализа")

    distance_matrix = compute_distance_matrix(X, used_columns, metric, decay_factor)

    try:
        overall_score = silhouette_score(distance_matrix, labels, metric="precomputed")
        sample_scores = silhouette_samples(distance_matrix, labels, metric="precomputed")
    except Exception as e:
        raise ValueError(f"Ошибка вычисления силуэта: {e}")

    return overall_score, sample_scores, labels, school_order, used_columns


# ==============================================================================
# ВИЗУАЛИЗАЦИЯ
# ==============================================================================

def create_silhouette_plot(
    sample_scores: np.ndarray,
    labels: np.ndarray,
    school_order: List[str],
    overall_score: float,
    metric_label: str,
) -> plt.Figure:
    """Создаёт график силуэта для научных школ."""
    n_schools = len(school_order)
    fig, ax = plt.subplots(figsize=(10, max(6, n_schools * 1.5)))

    y_lower = 10
    colors = plt.cm.Set2(np.linspace(0, 1, n_schools))

    for idx, school in enumerate(school_order):
        mask = labels == idx
        cluster_scores = sample_scores[mask]

        if cluster_scores.size == 0:
            continue

        cluster_scores = np.sort(cluster_scores)
        size = cluster_scores.size
        y_upper = y_lower + size

        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            cluster_scores,
            facecolor=colors[idx],
            edgecolor=colors[idx],
            alpha=0.7,
        )

        ax.text(
            -0.05,
            y_lower + size / 2,
            f"{school} (n={size})",
            fontsize=10,
            va="center",
            ha="right",
        )

        y_lower = y_upper + 10

    ax.axvline(
        x=overall_score,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Средний силуэт: {overall_score:.3f}"
    )

    ax.set_xlim(-1, 1)
    ax.set_xlabel("Коэффициент силуэта", fontsize=12)
    ax.set_ylabel("Научные школы", fontsize=12)
    ax.set_title(
        f"Анализ силуэта тематических профилей\n{metric_label}",
        fontsize=14,
        fontweight="bold"
    )
    ax.set_yticks([])
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    ax.axvspan(-1, -0.25, alpha=0.1, color="red")
    ax.axvspan(-0.25, 0.25, alpha=0.1, color="yellow")
    ax.axvspan(0.25, 0.5, alpha=0.1, color="lightgreen")
    ax.axvspan(0.5, 1, alpha=0.1, color="green")

    fig.tight_layout()
    return fig


def create_comparison_summary(
    datasets: Dict[str, pd.DataFrame],
    feature_columns: List[str],
    school_order: List[str],
) -> pd.DataFrame:
    """Создаёт сводную таблицу сравнения школ."""
    summary_data = []

    for school in school_order:
        if school not in datasets:
            continue

        data = datasets[school]
        if data.empty:
            continue

        available_cols = [c for c in feature_columns if c in data.columns]
        numeric_data = data[available_cols].fillna(0.0)

        summary_data.append({
            "Научная школа": school,
            "Количество диссертаций": len(data),
            "Средняя сумма профиля": numeric_data.sum(axis=1).mean(),
            "Медиана суммы профиля": numeric_data.sum(axis=1).median(),
            "Стд. отклонение": numeric_data.sum(axis=1).std(),
            "Ненулевых признаков (среднее)": (numeric_data > 0).sum(axis=1).mean(),
        })

    return pd.DataFrame(summary_data)


def interpret_silhouette_score(score: float) -> str:
    """Возвращает текстовую интерпретацию коэффициента силуэта."""
    if score >= 0.71:
        return "🟢 Отличное разделение: школы имеют чётко различающиеся тематические профили"
    elif score >= 0.51:
        return "🟢 Хорошее разделение: школы достаточно хорошо различаются по тематике"
    elif score >= 0.26:
        return "🟡 Умеренное разделение: есть частичное пересечение тематических профилей"
    elif score >= 0:
        return "🟠 Слабое разделение: школы имеют схожие тематические профили"
    else:
        return "🔴 Плохое разделение: тематические профили перемешаны"
