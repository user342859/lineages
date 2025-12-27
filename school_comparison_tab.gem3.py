import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
from typing import List, Dict, Set, Tuple, Optional, Callable
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx

# --- Константы ---
# Коэффициент затухания связи в иерархии для косоугольного базиса.
# Если 1.0 - узлы сливаются, если 0.0 - ортогональны. 
# 0.5 - умеренная корреляция между родителем и ребенком.
DEFAULT_HIERARCHY_CORRELATION = 0.5 

class ThematicProfileManager:
    """
    Класс для управления загрузкой профилей и математическими операциями
    в пространстве тематических признаков.
    """
    def __init__(self, data_folder: str):
        self.data_folder = data_folder
        self.profiles: Dict[str, np.ndarray] = {}  # ID -> Вектор
        self.columns: List[str] = []   # Список кодов (1.1, 1.1.1...)
        self.column_indices: Dict[str, int] = {}
        
    def load_data(self, specific_files: Optional[List[str]] = None) -> Tuple[int, int]:
        """
        Загружает CSV файлы. Возвращает (кол-во файлов, кол-во профилей).
        """
        if not os.path.exists(self.data_folder):
            st.error(f"Папка {self.data_folder} не найдена.")
            return 0, 0

        all_files = glob.glob(os.path.join(self.data_folder, "*.csv"))
        if specific_files:
            # Фильтруем по выбранным именам файлов
            target_files = set(specific_files)
            all_files = [f for f in all_files if os.path.basename(f) in target_files]
        
        if not all_files:
            return 0, 0

        dfs = []
        for f in all_files:
            try:
                # Ожидаем, что первая колонка - это ID (Code)
                df = pd.read_csv(f, index_col=0)
                # Удаляем возможные дубликаты индексов
                df = df[~df.index.duplicated(keep='first')]
                dfs.append(df)
            except Exception as e:
                st.warning(f"Не удалось прочитать файл {os.path.basename(f)}: {e}")
        
        if not dfs:
            return 0, 0

        # Объединяем и заполняем пропуски нулями (если наборы колонок отличаются)
        full_df = pd.concat(dfs, axis=0).fillna(0)
        
        # Сортируем колонки для корректной работы иерархии (1.1, 1.1.1, 1.2...)
        sorted_cols = sorted(full_df.columns.astype(str))
        full_df = full_df[sorted_cols]
        
        self.columns = sorted_cols
        self.column_indices = {col: i for i, col in enumerate(self.columns)}
        # Сохраняем как словарь numpy array для быстрого доступа
        self.profiles = {str(k): v for k, v in zip(full_df.index.astype(str), full_df.values)}
        
        return len(dfs), len(self.profiles)

    def get_basis_indices(self, root_node: str = "Весь базис") -> List[int]:
        """
        Возвращает индексы колонок, входящих в выбранную ветку иерархии.
        """
        if root_node == "Весь базис" or not root_node:
            return list(range(len(self.columns)))
        
        indices = []
        for i, col in enumerate(self.columns):
            # Проверяем, является ли колонка потомком или самим узлом
            # Например, '1.1.1' startswith '1.1'
            if col == root_node or col.startswith(root_node + "."):
                indices.append(i)
        return indices

    def build_gram_matrix(self, basis_cols: List[str], metric_mode: str) -> np.ndarray:
        """
        Строит матрицу Грама (метрический тензор) для выбранного базиса.
        Для прямоугольного базиса - единичная матрица.
        Для косоугольного - учитывает иерархическую близость.
        """
        dim = len(basis_cols)
        
        # Если базис прямоугольный (ортогональный)
        if "прямоугольн" in metric_mode.lower():
            return np.eye(dim)
        
        # Строим матрицу Грама для Косоугольного/Кривоугольного базиса
        # G[i, j] = cos(angle between axis_i and axis_j)
        # Логика из автореферата: оси имеют ненулевые углы, если у них общий родитель.
        # Упрощенная реализация: корреляция зависит от иерархического расстояния.
        
        G = np.eye(dim)
        
        # Создаем карту для быстрого поиска предков
        # col -> parts ['1', '1', '1']
        col_parts = [c.split('.') for c in basis_cols]
        
        for i in range(dim):
            for j in range(i + 1, dim):
                # Вычисляем близость
                parts_i = col_parts[i]
                parts_j = col_parts[j]
                
                # Находим длину общего префикса (LCP)
                common_len = 0
                min_len = min(len(parts_i), len(parts_j))
                for k in range(min_len):
                    if parts_i[k] == parts_j[k]:
                        common_len += 1
                    else:
                        break
                
                # Если общего родителя нет (кроме корня), считаем ортогональными (или малая связь)
                if common_len > 0:
                    # Расстояние в дереве:
                    # dist = (len_i - common) + (len_j - common)
                    dist = (len(parts_i) - common_len) + (len(parts_j) - common_len)
                    
                    # Формула затухания: alpha ^ dist
                    val = DEFAULT_HIERARCHY_CORRELATION ** dist
                    G[i, j] = G[j, i] = val
                    
        return G

def calculate_distance_matrix(X: np.ndarray, G: np.ndarray, metric_mode: str) -> np.ndarray:
    """
    Вычисляет матрицу попарных расстояний.
    X: (N_samples, N_features)
    G: (N_features, N_features) - Матрица Грама
    """
    # 1. Если прямоугольный базис (G - единичная), используем стандартные функции для скорости
    if "прямоугольн" in metric_mode.lower():
        if "евклидов" in metric_mode.lower():
            return euclidean_distances(X, X)
        elif "косинус" in metric_mode.lower():
            return cosine_distances(X, X) # Возвращает 1 - cos
    
    # 2. Работа с неортогональным базисом (Косоугольный/Кривоугольный)
    # Переводим данные в пространство, где метрика становится евклидовой,
    # используя разложение Холецкого для матрицы G (если она положительно определена)
    # Или используем формулу махаланобиса-подобную.
    
    # Расчет скалярных произведений в метрике G: <x, y>_G = x^T G y
    # X shape: (n, d), G shape: (d, d)
    # Inner products matrix K (n x n): K_ij = x_i G x_j^T
    X_G = X @ G  # (n, d)
    K = X_G @ X.T # (n, n)
    
    # Диагональ K (квадраты норм векторов в метрике G)
    diag_K = np.diag(K) # (n,)
    
    if "косинус" in metric_mode.lower():
        # Cosine distance in curvilinear basis
        # Dist = 1 - Cosine_similarity
        # Sim = <x,y>_G / (||x||_G * ||y||_G)
        
        norms = np.sqrt(diag_K)
        # Внешнее произведение норм для знаменателя
        norms_outer = np.outer(norms, norms)
        # Защита от деления на 0
        norms_outer[norms_outer == 0] = 1.0
        
        sim = K / norms_outer
        # Обрезаем численные ошибки за пределами [-1, 1]
        sim = np.clip(sim, -1.0, 1.0)
        return 1.0 - sim

    else:
        # Евклидово расстояние в косоугольном базисе
        # ||x - y||^2_G = <x-y, x-y>_G = <x,x>_G + <y,y>_G - 2<x,y>_G
        # Используем broadcasting
        # diag_K.reshape(-1, 1) -> столбец (x)
        # diag_K.reshape(1, -1) -> строка (y)
        dist_sq = diag_K.reshape(-1, 1) + diag_K.reshape(1, -1) - 2 * K
        
        # Убираем отрицательные значения от ошибок округления
        dist_sq[dist_sq < 0] = 0
        return np.sqrt(dist_sq)


def render_school_comparison_tab(df: pd.DataFrame, 
                                 idx: Dict[str, Set[int]], 
                                 lineage_func: Callable, 
                                 rows_for_func: Callable,
                                 default_scores_folder: str = "basic_scores",
                                 classifier_labels: Optional[Dict[str, str]] = None):
    """
    Основная функция рендеринга вкладки сравнения научных школ.
    """
    
    # --- 1. Сайдбар / Настройки (внутри expander для экономии места) ---
    with st.expander("⚙️ Настройки сравнения и метрик", expanded=True):
        col_sets_1, col_sets_2 = st.columns(2)
        
        with col_sets_1:
            # Выбор режима генерации
            comparison_mode = st.radio(
                "Кого сравниваем:",
                ["Непосредственное руководство (только прямые ученики)", 
                 "Все поколения (полное дерево школы)"],
                help="В первом случае берутся только защитившиеся непосредственно под руководством персоны. Во втором - все дерево потомков."
            )
            
            # Выбор папки и файлов
            scores_folder = st.text_input("Папка с CSV профилями:", value=default_scores_folder)
            
            # Попытка прочитать файлы для списка
            available_files = []
            if os.path.isdir(scores_folder):
                available_files = [os.path.basename(f) for f in glob.glob(os.path.join(scores_folder, "*.csv"))]
            
            selected_files = st.multiselect(
                "Файлы данных (Basic Scores)", 
                available_files, 
                default=available_files,
                help="Выберите конкретные файлы, если нужно ограничить выборку."
            )

        with col_sets_2:
            # Метрика
            metric_mode = st.selectbox(
                "Тип расстояния и базис:",
                [
                    "Косинусное расстояние (Прямоугольный базис)",
                    "Евклидово расстояние (Прямоугольный базис)",
                    "Косинусное расстояние (Кривоугольный базис)",
                    "Евклидово расстояние (Косоугольный базис)"
                ],
                help="Косоугольный/Кривоугольный базис учитывает, что тематические узлы иерархии (1.1 и 1.1.1) связаны семантически."
            )
            
            # Инициализация менеджера для получения списка колонок
            manager = ThematicProfileManager(scores_folder)
            # Предзагрузка нужна, чтобы узнать доступные узлы для базиса
            has_files, _ = manager.load_data(selected_files if selected_files else None)
            
            available_nodes = ["Весь базис"]
            if has_files:
                # Берем узлы 1-го и 2-го уровня для списка (чтобы не перегружать selectbox)
                # Например: 1.1, 1.2, 2.1...
                top_nodes = [c for c in manager.columns if c.count('.') <= 2]
                
                # Если есть словарь с названиями, используем его для красивого отображения
                if classifier_labels:
                    # Функция форматирования для selectbox
                    def format_node(code):
                        if code == "Весь базис": return code
                        label = classifier_labels.get(code, "")
                        return f"{code} - {label}" if label else code
                    
                    # Сортируем и добавляем
                    available_nodes += top_nodes
                    basis_selection = st.selectbox("Область сравнения (Узел классификатора):", available_nodes, format_func=format_node)
                else:
                    available_nodes += top_nodes
                    basis_selection = st.selectbox("Область сравнения (Узел классификатора):", available_nodes)
            else:
                basis_selection = "Весь базис"
                st.warning("Нет данных для выбора базиса. Проверьте путь к папке.")

    # --- 2. Выбор школ (Основная зона) ---
    st.subheader("Выбор научных школ")
    
    # Получаем список всех доступных имен из датафрейма для выбора глав школ
    # Предполагаем, что колонка автора называется candidate_name (как в основном app)
    # или используем первую строковую колонку.
    author_col = "candidate_name" 
    if author_col not in df.columns:
        # Fallback
        author_col = df.columns[1] 
    
    all_candidates = sorted(df[author_col].dropna().unique())
    
    selected_roots = st.multiselect(
        "Выберите основателей научных школ для сравнения:",
        all_candidates,
        placeholder="Начните вводить фамилию..."
    )

    if not selected_roots:
        st.info("👆 Выберите хотя бы одну персону, чтобы начать анализ.")
        return

    if not has_files:
        st.error("❌ Файлы с профилями не загружены. Проверьте настройки.")
        return

    # Кнопка запуска расчета
    if st.button("🚀 Рассчитать сравнение"):
        
        # --- 3. Сбор данных (Person -> ID -> Vector) ---
        
        # Нужно понять, какая колонка в df отвечает за ID, который используется в именах файлов CSV.
        # Обычно это id, author_id или что-то подобное.
        # Попытаемся найти колонку 'id' или 'code'.
        id_col = None
        for col in ['id', 'code', 'author_id', 'rosrid_id']:
            if col in df.columns:
                id_col = col
                break
        
        # Если ID не найден, попробуем использовать имя, но это ненадежно
        use_name_as_id = False
        if id_col is None:
            st.warning("⚠️ В данных не найдена колонка ID (id, code). Будем искать профили по ФИО (это может быть неточно).")
            use_name_as_id = True
            id_col = author_col

        # Подготовка структур данных
        X_vectors = []
        Y_labels = []      # Метка кластера (Имя школы)
        Point_labels = []  # Метка точки (Имя ученого)
        
        schools_stats = {} # Для вывода инфо

        # Определяем базисные индексы (фильтрация колонок)
        basis_indices = manager.get_basis_indices(basis_selection)
        basis_cols_names = [manager.columns[i] for i in basis_indices]
        
        if not basis_indices:
            st.error("Выбранный базис пуст (нет колонок).")
            return

        with st.spinner("Построение деревьев и поиск профилей..."):
            
            for root_name in selected_roots:
                # 1. Строим дерево (Lineage)
                # lineage_func возвращает (Graph, DataFrame подмножества)
                # Аргумент filter передаем как None, если хотим всех, или функцию фильтрации
                G, subset_df = lineage_func(df, idx, root_name, None)
                
                # 2. Определяем список персон
                members = []
                if "Непосредственное" in comparison_mode:
                    # Только дети корня
                    if root_name in G:
                        members = list(G.successors(root_name))
                    # + сам основатель
                    members.append(root_name)
                else:
                    # Все узлы графа
                    members = list(G.nodes())
                
                # Убираем дубликаты
                members = list(set(members))
                
                # 3. Сопоставляем персоны с профилями
                found_count = 0
                
                for member_name in members:
                    # Находим строку в subset_df (или main df)
                    # lineage возвращает subset, там искать быстрее
                    row = subset_df[subset_df[author_col] == member_name]
                    if row.empty:
                        # Если вдруг нет в subset, ищем в основном (на всякий случай)
                        row = rows_for_func(df, idx, member_name)
                    
                    if row.empty:
                        continue
                        
                    # Получаем ID для поиска в manager.profiles
                    # Берем первую запись (обычно уникально)
                    person_id = str(row.iloc[0][id_col]).strip()
                    
                    # Ищем вектор
                    vec = None
                    
                    # Прямой поиск по ID
                    if person_id in manager.profiles:
                        vec = manager.profiles[person_id]
                    # Если не нашли, пробуем вариации (иногда в CSV id с добавками)
                    # В примере CSV id выглядят как '000199_000009_...'
                    elif not use_name_as_id:
                        # Попробуем найти ключ, который содержит этот ID
                        for key in manager.profiles:
                            if person_id in key:
                                vec = manager.profiles[key]
                                break
                    
                    if vec is not None:
                        # Фильтруем вектор по базису
                        vec_basis = vec[basis_indices]
                        
                        # Проверяем, что вектор не пустой (не одни нули), 
                        # иначе он испортит косинусное расстояние
                        if np.sum(np.abs(vec_basis)) > 0:
                            X_vectors.append(vec_basis)
                            Y_labels.append(root_name) # Кластер = Имя школы
                            Point_labels.append(member_name)
                            found_count += 1
                
                schools_stats[root_name] = {"total": len(members), "found": found_count}

        # --- 4. Отображение статистики поиска ---
        st.write("### Результаты поиска профилей")
        cols = st.columns(len(selected_roots))
        for i, (school, stats) in enumerate(schools_stats.items()):
            cols[i % len(cols)].metric(
                label=school, 
                value=f"{stats['found']} профилей",
                delta=f"из {stats['total']} персон"
            )

        if len(X_vectors) < 2:
            st.error("Недостаточно данных для сравнения (найдено менее 2 профилей).")
            return
        
        if len(set(Y_labels)) < 2:
            st.warning("⚠️ Выбрана только одна школа (или данные есть только для одной). Межкластерное сравнение невозможно, но мы покажем внутреннюю структуру.")

        X = np.array(X_vectors)
        
        # --- 5. Вычисления ---
        with st.spinner(f"Вычисление расстояний ({metric_mode})..."):
            # Строим матрицу Грама
            G_matrix = manager.build_gram_matrix(basis_cols_names, metric_mode)
            
            # Считаем матрицу расстояний
            dist_matrix = calculate_distance_matrix(X, G_matrix, metric_mode)
            
        # --- 6. Силуэтный анализ (Silhouette) ---
        
        # Силуэт считается только если есть >= 2 кластера
        # Но если кластер 1 (одна школа), силуэт не определен (вернет ошибку).
        # В этом случае можно пропустить общую метрику.
        
        can_calculate_silhouette = len(set(Y_labels)) > 1
        
        if can_calculate_silhouette:
            sil_avg = silhouette_score(dist_matrix, Y_labels, metric="precomputed")
            sample_silhouette_values = silhouette_samples(dist_matrix, Y_labels, metric="precomputed")
            
            st.divider()
            col_res1, col_res2 = st.columns([1, 3])
            
            with col_res1:
                st.metric("Средний Silhouette Score", f"{sil_avg:.3f}")
                if sil_avg > 0.5:
                    st.success("Высокая обособленность школ")
                elif sil_avg > 0.2:
                    st.info("Средняя обособленность")
                else:
                    st.warning("Школы сильно пересекаются тематически")
            
            with col_res2:
                # --- 7. График Силуэта (Plotly) ---
                fig = go.Figure()
                
                # Группируем значения по школам
                unique_labels = sorted(list(set(Y_labels)))
                colors = px.colors.qualitative.Plotly
                
                y_pos_start = 0
                
                for i, label in enumerate(unique_labels):
                    # Индексы, относящиеся к этому кластеру
                    indices = [j for j, x in enumerate(Y_labels) if x == label]
                    
                    # Значения силуэта для них
                    values = sample_silhouette_values[indices]
                    names = [Point_labels[j] for j in indices]
                    
                    # Сортируем внутри кластера для красивого "силуэта"
                    # Zip, sort, unzip
                    sorted_data = sorted(zip(values, names), key=lambda x: x[0])
                    sorted_values = [x[0] for x in sorted_data]
                    sorted_names = [x[1] for x in sorted_data]
                    
                    y_pos_end = y_pos_start + len(values)
                    y_range = np.arange(y_pos_start, y_pos_end)
                    
                    # Добавляем бар (ориентация h)
                    fig.add_trace(go.Bar(
                        x=sorted_values,
                        y=y_range,
                        name=label,
                        orientation='h',
                        marker_color=colors[i % len(colors)],
                        text=sorted_names, # Имена при наведении
                        hoverinfo='x+text+name',
                        width=1.0 # Бары вплотную
                    ))
                    
                    y_pos_start = y_pos_end + 5 # Отступ между школами
                
                fig.update_layout(
                    title="График силуэта (Тематические профили)",
                    xaxis_title="Коэффициент силуэта",
                    yaxis_title="Диссертации",
                    yaxis=dict(showticklabels=False), # Скрываем цифры оси Y
                    bargap=0,
                    height=600,
                    legend_title="Научные школы"
                )
                
                # Вертикальная линия на уровне средней
                fig.add_vline(x=sil_avg, line_width=2, line_dash="dash", line_color="red", annotation_text="Avg")
                
                st.plotly_chart(fig, use_container_width=True)
                
        else:
            st.info("Для расчета метрики Силуэта выберите как минимум две разные научные школы.")

        # --- Дополнительно: Матрица расстояний (тепловая карта) ---
        with st.expander("Показать матрицу расстояний (Heatmap)"):
            # Для визуализации матрица может быть большой, поэтому ограничим или сделаем интерактивной
            # Сортируем матрицу по кластерам для наглядности
            
            # Получаем индексы сортировки
            sort_indices = np.argsort(Y_labels)
            
            sorted_dist = dist_matrix[sort_indices][:, sort_indices]
            sorted_names = np.array(Point_labels)[sort_indices]
            sorted_schools = np.array(Y_labels)[sort_indices]
            
            # Создаем подписи: "Школа: Имя"
            axis_labels = [f"{s}: {n}" for s, n in zip(sorted_schools, sorted_names)]
            
            fig_matrix = go.Figure(data=go.Heatmap(
                z=sorted_dist,
                x=axis_labels,
                y=axis_labels,
                colorscale='Viridis_r', # Реверс, т.к. 0 (близко) - это хорошо (светло/темно)
            ))
            
            fig_matrix.update_layout(
                title=f"Матрица расстояний ({metric_mode})",
                width=800, height=800,
                xaxis=dict(tickfont=dict(size=8)),
                yaxis=dict(tickfont=dict(size=8))
            )
            st.plotly_chart(fig_matrix)


