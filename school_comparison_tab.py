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

# --- Константы ---
DEFAULT_HIERARCHY_CORRELATION = 0.5 

class ThematicProfileManager:
    """
    Класс для управления загрузкой профилей и математическими операциями.
    """
    def __init__(self, data_folder: str):
        self.data_folder = data_folder
        self.profiles: Dict[str, np.ndarray] = {}  # ID (Code) -> Вектор
        self.columns: List[str] = []   # 1.1, 1.1.1...
        
    def load_data(self, specific_files: Optional[List[str]] = None) -> Tuple[int, int]:
        if not os.path.exists(self.data_folder):
            st.error(f"Папка {self.data_folder} не найдена.")
            return 0, 0

        all_files = glob.glob(os.path.join(self.data_folder, "*.csv"))
        if specific_files:
            target_files = set(specific_files)
            all_files = [f for f in all_files if os.path.basename(f) in target_files]
        
        if not all_files:
            return 0, 0

        dfs = []
        for f in all_files:
            try:
                # Читаем CSV. Предполагаем, что ID находится в первой колонке (index_col=0)
                # Пример: Code, 1.1, 1.1.1 ...
                df = pd.read_csv(f, index_col=0)
                # Индекс приводим к строке, чтобы точно совпадал с ID из базы
                df.index = df.index.astype(str)
                # Убираем дубликаты индексов
                df = df[~df.index.duplicated(keep='first')]
                dfs.append(df)
            except Exception as e:
                st.warning(f"Не удалось прочитать файл {os.path.basename(f)}: {e}")
        
        if not dfs:
            return 0, 0

        full_df = pd.concat(dfs, axis=0).fillna(0)
        
        # Сортируем колонки
        sorted_cols = sorted(full_df.columns.astype(str))
        full_df = full_df[sorted_cols]
        
        self.columns = sorted_cols
        # Сохраняем map: ID -> Vector
        self.profiles = {k: v for k, v in zip(full_df.index, full_df.values)}
        
        return len(dfs), len(self.profiles)

    def get_basis_indices(self, root_node: str = "Весь базис") -> List[int]:
        if root_node == "Весь базис" or not root_node:
            return list(range(len(self.columns)))
        
        indices = []
        for i, col in enumerate(self.columns):
            if col == root_node or col.startswith(root_node + "."):
                indices.append(i)
        return indices

    def build_gram_matrix(self, basis_cols: List[str], metric_mode: str) -> np.ndarray:
        dim = len(basis_cols)
        if "прямоугольн" in metric_mode.lower():
            return np.eye(dim)
        
        # Матрица для косоугольного базиса
        G = np.eye(dim)
        col_parts = [c.split('.') for c in basis_cols]
        
        for i in range(dim):
            for j in range(i + 1, dim):
                parts_i = col_parts[i]
                parts_j = col_parts[j]
                
                common_len = 0
                min_len = min(len(parts_i), len(parts_j))
                for k in range(min_len):
                    if parts_i[k] == parts_j[k]:
                        common_len += 1
                    else:
                        break
                
                if common_len > 0:
                    dist = (len(parts_i) - common_len) + (len(parts_j) - common_len)
                    val = DEFAULT_HIERARCHY_CORRELATION ** dist
                    G[i, j] = G[j, i] = val
        return G

def calculate_distance_matrix(X: np.ndarray, G: np.ndarray, metric_mode: str) -> np.ndarray:
    if "прямоугольн" in metric_mode.lower():
        if "евклидов" in metric_mode.lower():
            return euclidean_distances(X, X)
        elif "косинус" in metric_mode.lower():
            return cosine_distances(X, X)
    
    # Расчет в метрике G
    X_G = X @ G
    K = X_G @ X.T
    diag_K = np.diag(K)
    
    if "косинус" in metric_mode.lower():
        norms = np.sqrt(diag_K)
        norms_outer = np.outer(norms, norms)
        norms_outer[norms_outer == 0] = 1.0
        sim = K / norms_outer
        sim = np.clip(sim, -1.0, 1.0)
        return 1.0 - sim
    else:
        dist_sq = diag_K.reshape(-1, 1) + diag_K.reshape(1, -1) - 2 * K
        dist_sq[dist_sq < 0] = 0
        return np.sqrt(dist_sq)


def render_school_comparison_tab(df: pd.DataFrame, 
                                 idx: Dict[str, Set[int]], 
                                 lineage_func: Callable, 
                                 rows_for_func: Callable,
                                 default_scores_folder: str = "basic_scores",
                                 classifier_labels: Optional[Dict[str, str]] = None):
    
    # --- 1. Настройки ---
    with st.expander("⚙️ Настройки сравнения и метрик", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            comparison_mode = st.radio(
                "Кого сравниваем:",
                ["Непосредственное руководство (только прямые ученики)", 
                 "Все поколения (полное дерево школы)"]
            )
            scores_folder = st.text_input("Папка с CSV профилями:", value=default_scores_folder)
            
            available_files = []
            if os.path.isdir(scores_folder):
                available_files = [os.path.basename(f) for f in glob.glob(os.path.join(scores_folder, "*.csv"))]
            
            selected_files = st.multiselect("Файлы данных", available_files, default=available_files)

        with col2:
            metric_mode = st.selectbox(
                "Тип расстояния и базис:",
                [
                    "Косинусное расстояние (Прямоугольный базис)",
                    "Евклидово расстояние (Прямоугольный базис)",
                    "Косинусное расстояние (Кривоугольный базис)",
                    "Евклидово расстояние (Косоугольный базис)"
                ]
            )
            
            manager = ThematicProfileManager(scores_folder)
            has_files, _ = manager.load_data(selected_files if selected_files else None)
            
            available_nodes = ["Весь базис"]
            if has_files:
                top_nodes = [c for c in manager.columns if c.count('.') <= 2]
                if classifier_labels:
                    def fmt(c): return f"{c} - {classifier_labels.get(c, '')}" if c != "Весь базис" else c
                    available_nodes += top_nodes
                    basis_selection = st.selectbox("Область сравнения:", available_nodes, format_func=fmt)
                else:
                    available_nodes += top_nodes
                    basis_selection = st.selectbox("Область сравнения:", available_nodes)
            else:
                basis_selection = "Весь базис"

    # --- 2. Выбор школ ---
    st.subheader("Выбор научных школ")
    
    # Определяем колонку с ФИО
    author_col = "candidate_name"
    if author_col not in df.columns:
        # Пытаемся угадать, если вдруг название другое
        str_cols = df.select_dtypes(include=['object']).columns
        if len(str_cols) > 0:
            author_col = str_cols[0]
            
    all_candidates = sorted(df[author_col].dropna().unique())
    selected_roots = st.multiselect("Основатели школ:", all_candidates)

    if not selected_roots or not has_files:
        if not has_files: st.error("Файлы профилей не найдены.")
        return

    if st.button("🚀 Рассчитать сравнение"):
        
        # --- 3. ОПРЕДЕЛЕНИЕ ID КОЛОНКИ (FIX) ---
        # Нам нужно найти колонку в df, которая соответствует ID в basic_scores (Code)
        
        # Возможные названия колонки ID в df
        possible_id_cols = ['id', 'code', 'rosrid_id', 'author_id', 'dis_id']
        id_col = None
        
        # Ищем точное совпадение
        for col in possible_id_cols:
            if col in df.columns:
                id_col = col
                break
        
        # Если не нашли, ищем колонку, содержащую 'id' (case insensitive)
        if id_col is None:
            for col in df.columns:
                if 'id' in col.lower():
                    id_col = col
                    break
        
        if id_col is None:
            st.error("Не удалось найти колонку с ID (id, code, etc.) в основном файле данных. Невозможно связать персону с вектором.")
            return

        # Создаем быстрый lookup словарь: Имя -> ID
        # Если имен много одинаковых, берем первый попавшийся ID (или обрабатываем список)
        # df[author_col] -> df[id_col]
        
        # Приводим ID к строке для матчинга с keys словаря profiles
        name_to_id_map = dict(zip(df[author_col], df[id_col].astype(str)))
        
        
        # --- 4. Сбор векторов ---
        X_vectors = []
        Y_labels = []
        Point_labels = []
        schools_stats = {}

        basis_indices = manager.get_basis_indices(basis_selection)
        basis_cols_names = [manager.columns[i] for i in basis_indices]
        
        if not basis_indices:
            st.error("Пустой базис.")
            return

        with st.spinner("Анализ структур школ и поиск векторов..."):
            for root_name in selected_roots:
                # Строим дерево
                G, _ = lineage_func(df, idx, root_name, None)
                
                members = []
                if "Непосредственное" in comparison_mode:
                    if root_name in G:
                        members = list(G.successors(root_name))
                    members.append(root_name)
                else:
                    members = list(G.nodes())
                
                members = list(set(members))
                found_count = 0
                
                for member_name in members:
                    # 1. Получаем ID персоны из main DF
                    person_id = name_to_id_map.get(member_name)
                    
                    if not person_id:
                        continue
                        
                    # 2. Ищем вектор по ID
                    vec = manager.profiles.get(person_id)
                    
                    if vec is not None:
                        # Фильтрация по базису
                        vec_basis = vec[basis_indices]
                        
                        # Пропускаем нулевые вектора (нет данных по этому разделу)
                        if np.sum(np.abs(vec_basis)) > 0:
                            X_vectors.append(vec_basis)
                            Y_labels.append(root_name)
                            Point_labels.append(member_name)
                            found_count += 1
                            
                schools_stats[root_name] = {"total": len(members), "found": found_count}

        # Вывод статистики
        cols = st.columns(len(selected_roots))
        for i, (school, stats) in enumerate(schools_stats.items()):
            cols[i % len(cols)].metric(school, f"{stats['found']} / {stats['total']}")

        if len(X_vectors) < 2:
            st.warning("Недостаточно векторов для анализа.")
            return

        # --- 5. Расчет и Визуализация ---
        X = np.array(X_vectors)
        G_matrix = manager.build_gram_matrix(basis_cols_names, metric_mode)
        dist_matrix = calculate_distance_matrix(X, G_matrix, metric_mode)
        
        # Силуэт
        if len(set(Y_labels)) > 1:
            try:
                sil_avg = silhouette_score(dist_matrix, Y_labels, metric="precomputed")
                sample_vals = silhouette_samples(dist_matrix, Y_labels, metric="precomputed")
                
                st.metric("Silhouette Score", f"{sil_avg:.3f}")
                
                fig = go.Figure()
                unique_labels = sorted(list(set(Y_labels)))
                colors = px.colors.qualitative.Plotly
                y_pos = 0
                
                for i, label in enumerate(unique_labels):
                    indices = [j for j, x in enumerate(Y_labels) if x == label]
                    vals = sample_vals[indices]
                    names = [Point_labels[j] for j in indices]
                    
                    # Сортировка
                    zipped = sorted(zip(vals, names), key=lambda x: x[0])
                    s_vals = [z[0] for z in zipped]
                    s_names = [z[1] for z in zipped]
                    
                    fig.add_trace(go.Bar(
                        x=s_vals, 
                        y=np.arange(y_pos, y_pos + len(s_vals)),
                        name=label,
                        orientation='h',
                        marker_color=colors[i % len(colors)],
                        text=s_names,
                        width=1.0
                    ))
                    y_pos += len(s_vals) + 5
                    
                fig.update_layout(
                    title="График силуэта", 
                    bargap=0, 
                    yaxis=dict(showticklabels=False),
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Ошибка расчета силуэта: {e}")
        else:
            st.info("Для расчета силуэта нужно минимум 2 школы.")
