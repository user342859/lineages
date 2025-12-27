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
                # Читаем CSV. Индекс - первая колонка (Code)
                csv_df = pd.read_csv(f, index_col=0)
                # Приводим индекс к строке и убираем пробелы
                csv_df.index = csv_df.index.astype(str).str.strip()
                # Убираем дубликаты индексов
                csv_df = csv_df[~csv_df.index.duplicated(keep='first')]
                dfs.append(csv_df)
            except Exception as e:
                st.warning(f"Не удалось прочитать файл {os.path.basename(f)}: {e}")
        
        if not dfs:
            return 0, 0

        full_df = pd.concat(dfs, axis=0).fillna(0)
        
        # Сортируем колонки
        sorted_cols = sorted(full_df.columns.astype(str))
        full_df = full_df[sorted_cols]
        
        self.columns = sorted_cols
        # Сохраняем map: ID -> Vector (явно как numpy array)
        for idx_val in full_df.index:
            self.profiles[idx_val] = full_df.loc[idx_val].values.astype(float)
        
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
    
    X_G = X @ G
    K = X_G @ X.T
    diag_K = np.diag(K)
    
    if "косинус" in metric_mode.lower():
        norms = np.sqrt(np.maximum(diag_K, 0))  # защита от отрицательных значений
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
            has_files, num_profiles = manager.load_data(selected_files if selected_files else None)
            
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
    
    # Определяем колонку автора
    author_col = "candidate_name"
    if author_col not in df.columns:
        possible_auth_cols = [c for c in df.columns if "name" in c.lower() or "author" in c.lower()]
        if possible_auth_cols:
            author_col = possible_auth_cols[0]
            
    # Собираем все имена (авторы + руководители)
    all_names_set = set()
    if author_col in df.columns:
        all_names_set.update(df[author_col].dropna().unique())
    for col in df.columns:
        if 'supervisor' in col.lower() and 'name' in col.lower():
            all_names_set.update(df[col].dropna().unique())
            
    all_candidates_list = sorted([str(x).strip() for x in all_names_set if str(x).strip()])
    
    selected_roots = st.multiselect(
        "Выберите основателей научных школ:", 
        all_candidates_list,
        placeholder="Начните вводить фамилию..."
    )

    if not selected_roots or not has_files:
        if not has_files: 
            st.error("Файлы профилей не найдены.")
        return

    if st.button("🚀 Рассчитать сравнение"):
        
        # --- 3. ИНТЕЛЛЕКТУАЛЬНЫЙ ПОИСК СВЯЗУЮЩЕЙ КОЛОНКИ (ID) ---
        
        # Получаем все ID профилей
        available_profile_ids = set(manager.profiles.keys())
        
        best_col = None
        max_overlap = 0
        
        # Перебираем все колонки
        for col in df.columns:
            try:
                col_values = set(df[col].dropna().astype(str).str.strip())
                overlap = len(col_values.intersection(available_profile_ids))
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_col = col
            except Exception:
                continue

        if best_col and max_overlap > 0:
            id_col = best_col
        else:
            st.error("❌ Не удалось найти колонку, совпадающую с ID профилей.")
            with st.expander("Отладка"):
                st.write("Колонки в данных:", df.columns.tolist())
                st.write("Пример ID из профилей:", list(available_profile_ids)[:5])
            return

        # Словарь Имя -> ID
        df_clean = df[[author_col, id_col]].dropna()
        name_to_id_map = dict(zip(df_clean[author_col], df_clean[id_col].astype(str).str.strip()))
        
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

        with st.spinner("Поиск профилей и расчет метрик..."):
            for root_name in selected_roots:
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
                    person_id = name_to_id_map.get(member_name)
                    
                    if not person_id:
                        continue
                        
                    vec = manager.profiles.get(person_id)
                    
                    if vec is not None:
                        try:
                            # Явно приводим к numpy array float
                            vec_arr = np.asarray(vec, dtype=float)
                            vec_basis = vec_arr[basis_indices]
                            
                            # Проверяем, что вектор не нулевой
                            if np.nansum(np.abs(vec_basis)) > 0:
                                X_vectors.append(vec_basis.copy())
                                Y_labels.append(root_name)
                                Point_labels.append(member_name)
                                found_count += 1
                        except Exception:
                            continue
                            
                schools_stats[root_name] = {"total": len(members), "found": found_count}

        # Статистика
        cols = st.columns(len(selected_roots))
        for i, (school, stats) in enumerate(schools_stats.items()):
            cols[i % len(cols)].metric(school, f"{stats['found']} / {stats['total']}")

        if len(X_vectors) < 2:
            st.warning("Недостаточно найденных профилей (нужно минимум 2).")
            return

        # --- 5. Расчет ---
        X = np.array(X_vectors, dtype=float)
        G_matrix = manager.build_gram_matrix(basis_cols_names, metric_mode)
        dist_matrix = calculate_distance_matrix(X, G_matrix, metric_mode)
        
        # --- 6. Визуализация ---
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
                        hoverinfo='x+text+name',
                        width=1.0
                    ))
                    y_pos += len(s_vals) + 5
                    
                fig.update_layout(
                    title="График силуэта (Тематические профили)", 
                    bargap=0, 
                    yaxis=dict(showticklabels=False),
                    height=max(500, y_pos * 15),
                    legend_title="Научные школы"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Ошибка расчета силуэта: {e}")
        else:
            st.info("⚠️ Для расчета силуэта нужно минимум 2 школы.")
