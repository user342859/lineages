# streamlit_app.py (RU, preloaded, simplified UI)
# -------------------------------------------------------------
# Академический конструктор родословных (без загрузки файлов)
# Данные берутся из локальной папки ./db_lineages (в репозитории).
# Интерфейс на русском, без технических настроек в сайдбаре.
# -------------------------------------------------------------

from __future__ import annotations

import csv
import io
import json
import os
import re
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
from urllib.parse import urlencode, urlsplit
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
except Exception:  # pragma: no cover - совместимость со старыми версиями streamlit
    get_script_run_ctx = None  # type: ignore
import zipfile
from pyvis.network import Network
from sklearn.metrics import silhouette_samples, silhouette_score

# ---------------------- Константы -----------------------------------------
DATA_DIR = "db_lineages"      # папка с CSV внутри репозитория
CSV_GLOB = "*.csv"            # какие файлы брать
AUTHOR_COLUMN = "candidate_name"
SUPERVISOR_COLUMNS = [f"supervisors_{i}.name" for i in (1, 2)]

BASIC_SCORES_DIR = "basic_scores"  # тематические профили диссертаций

FEEDBACK_FILE = Path("feedback.csv")
FEEDBACK_FORM_STATE_KEY = "feedback_form_state"
FEEDBACK_FORM_RESULT_KEY = "feedback_form_result"

# Публичный адрес приложения для формирования ссылок "Поделиться".
# При необходимости его можно переопределить через переменную окружения
# PUBLIC_APP_URL.
PUBLIC_APP_URL = os.environ.get(
    "PUBLIC_APP_URL",
    "https://lineages-trceuocpnvyaxysnpis72f.streamlit.app/",
).strip().rstrip("/")

# ---------------------- Оформление страницы -------------------------------
st.set_page_config(page_title="Академические родословные", layout="wide")

# Полноширинный (full-bleed) контейнер для компонентов
st.markdown("""
<style>
  iframe {
        width: 100%;
  }
</style>
""", unsafe_allow_html=True)


def _default_feedback_state() -> Dict[str, str]:
    return {"name": "", "email": "", "message": ""}


def _get_feedback_state() -> Dict[str, str]:
    state = st.session_state.get(FEEDBACK_FORM_STATE_KEY)
    if isinstance(state, dict):
        return state
    state = _default_feedback_state()
    st.session_state[FEEDBACK_FORM_STATE_KEY] = state
    return state


def _store_feedback(name: str, email: str, message: str) -> None:
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    record = [
        datetime.utcnow().isoformat(timespec="seconds") + "Z",
        name.strip(),
        email.strip(),
        message.replace("\r\n", "\n").replace("\r", "\n"),
    ]
    file_exists = FEEDBACK_FILE.exists()
    with FEEDBACK_FILE.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        if not file_exists:
            writer.writerow(["timestamp", "name", "email", "message"])
        writer.writerow(record)


def _trigger_rerun() -> None:
    try:  # Streamlit >= 1.32
        st.rerun()
    except AttributeError:  # pragma: no cover - старые версии Streamlit
        st.experimental_rerun()  # type: ignore[attr-defined]


def feedback_button() -> None:
    @st.dialog("Обратная связь")
    def _show_feedback_dialog() -> None:
        st.write("Будем рады предложениям по улучшению и информации об ошибках.")

        feedback_state = _get_feedback_state()
        pending_message = st.session_state.pop(FEEDBACK_FORM_RESULT_KEY, None)
        if pending_message:
            status, context = pending_message
            if status == "success":
                st.success(
                    f"Спасибо, {context or 'коллега'}! Мы получили ваше сообщение."
                )
            elif status == "warning":
                st.warning("Пожалуйста, заполните поле «Сообщение».")

        with st.form(key="feedback_form"):
            name = st.text_input("Имя", value=feedback_state.get("name", ""))
            email = st.text_input("E-mail", value=feedback_state.get("email", ""))
            message = st.text_area(
                "Сообщение", value=feedback_state.get("message", ""), height=180
            )
            submitted = st.form_submit_button("Отправить")

        if submitted:
            feedback_state = {
                "name": name,
                "email": email,
                "message": message,
            }
            if message.strip():
                _store_feedback(name, email, message)
                st.session_state[FEEDBACK_FORM_RESULT_KEY] = ("success", name)
                st.session_state[FEEDBACK_FORM_STATE_KEY] = _default_feedback_state()
            else:
                st.session_state[FEEDBACK_FORM_RESULT_KEY] = ("warning", None)
                st.session_state[FEEDBACK_FORM_STATE_KEY] = feedback_state
            _trigger_rerun()

    if st.button("Обратная связь", key="feedback_button", use_container_width=True):
        _show_feedback_dialog()


header_left, header_right = st.columns([0.78, 0.22])
with header_left:
    st.title("📚 Конструктор академических родословных")
    st.caption(
        "Данные заранее загружены в репозиторий (папка db_lineages). Выберите начальных руководителей и создайте деревья."
    )
with header_right:
    feedback_button()

# ---------------------- Хелперы -------------------------------------------

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace(".", " ").strip().lower())


def _split(full: str) -> Tuple[str, str, str]:
    p = full.split()
    p += ["", "", ""]
    return (p[0], p[1] if len(p) > 1 else "", p[2] if len(p) > 2 else "")


def variants(full: str) -> Set[str]:
    last, first, mid = _split(full.strip())
    fi, mi = first[:1], mid[:1]
    init = fi + mi
    init_dots = ".".join(init) + "." if init else ""
    return {
        v.strip()
        for v in {
            full,
            f"{last} {first} {mid}".strip(),
            f"{last} {init}",
            f"{last} {init_dots}",
            f"{init} {last}",
            f"{init_dots} {last}",
        }
        if v
    }


def degree_level(row: pd.Series) -> str:
    raw = str(row.get("degree.degree_level", ""))
    value = raw.strip().lower()
    if value.startswith("док"):
        return "doctor"
    if value.startswith("кан"):
        return "candidate"
    return ""


def is_doctor(row: pd.Series) -> bool:
    return degree_level(row) == "doctor"


def is_candidate(row: pd.Series) -> bool:
    return degree_level(row) == "candidate"


TREE_OPTIONS: List[tuple[str, str, Callable[[pd.Series], bool] | None]] = [
    ("Общее дерево", "general", None),
    ("Дерево докторов наук", "doctors", is_doctor),
    ("Дерево кандидатов наук", "candidates", is_candidate),
]


def build_index(df: pd.DataFrame, supervisor_cols: List[str]) -> Dict[str, Set[int]]:
    idx: Dict[str, Set[int]] = {}
    for col in supervisor_cols:
        if col not in df.columns:
            continue
        for i, raw in df[col].dropna().items():
            for v in variants(str(raw)):
                idx.setdefault(_norm(v), set()).add(i)
    return idx


def rows_for(df: pd.DataFrame, index: Dict[str, Set[int]], name: str) -> pd.DataFrame:
    hits: Set[int] = set()
    for v in variants(name):
        hits.update(index.get(_norm(v), set()))
    return df.loc[list(hits)] if hits else df.iloc[0:0]


def lineage(
    df: pd.DataFrame,
    index: Dict[str, Set[int]],
    root: str,
    first_level_filter: Callable[[pd.Series], bool] | None = None,
) -> tuple[nx.DiGraph, pd.DataFrame]:
    G = nx.DiGraph()
    selected_indices: Set[int] = set()
    Q, seen = [root], set()
    while Q:
        cur = Q.pop(0)
        if cur in seen:
            continue
        seen.add(cur)
        rows = rows_for(df, index, cur)
        for idx, r in rows.iterrows():
            child = str(r.get(AUTHOR_COLUMN, "")).strip()
            if child:
                if cur == root and first_level_filter is not None:
                    if not first_level_filter(r):
                        continue
                G.add_edge(cur, child)
                Q.append(child)
                selected_indices.add(idx)
    subset = df.loc[sorted(selected_indices)] if selected_indices else df.iloc[0:0]
    return G, subset


def multiline(name: str) -> str:
    return "\n".join(str(name).split())


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-zА-Яа-я0-9]+", "_", s).strip("_")


def _clean_path(*parts: str) -> str:
    cleaned = "/".join(p.strip("/") for p in parts if p and p.strip("/"))
    return f"/{cleaned}" if cleaned else ""


def _configured_base_url() -> str | None:
    if PUBLIC_APP_URL:
        return PUBLIC_APP_URL
    keys = ("public_base_url", "base_url", "BASE_URL")
    for key in keys:
        try:
            val = st.secrets.get(key)  # type: ignore[attr-defined]
        except Exception:
            val = None
        if val:
            return str(val).rstrip("/")
    for key in ("PUBLIC_BASE_URL", "BASE_URL"):
        val = os.environ.get(key)
        if val:
            return val.rstrip("/")
    return None


def _base_url_from_headers() -> str | None:
    if get_script_run_ctx is None:
        return None
    try:
        ctx = get_script_run_ctx()
    except Exception:
        ctx = None
    if not ctx:
        return None
    headers = getattr(ctx, "request_headers", None)
    if not headers:
        return None
    lowered = {str(k).lower(): str(v) for k, v in headers.items() if v}
    prefix = lowered.get("x-forwarded-prefix", "")
    base_path = st.get_option("server.baseUrlPath") or ""

    host = lowered.get("x-forwarded-host") or lowered.get("host")
    if host:
        proto = lowered.get("x-forwarded-proto")
        if proto:
            proto = proto.split(",")[0].strip()
        else:
            forwarded_port = lowered.get("x-forwarded-port")
            proto = "https" if forwarded_port == "443" or host.endswith(":443") else "http"
        path = _clean_path(prefix, base_path)
        return f"{proto}://{host}{path}".rstrip("/")

    referer = lowered.get("referer") or lowered.get("origin")
    if not referer:
        return None
    parsed = urlsplit(referer)
    if not parsed.scheme or not parsed.netloc:
        return None
    path = _clean_path(prefix or parsed.path, base_path)
    base = f"{parsed.scheme}://{parsed.netloc}"
    return f"{base}{path}".rstrip("/")


def _base_url_from_options() -> str | None:
    try:
        addr = st.get_option("browser.serverAddress")
        port = st.get_option("browser.serverPort")
    except Exception:
        return None
    if not addr:
        return None
    base_path = st.get_option("server.baseUrlPath") or ""
    proto = "https" if str(port) == "443" else "http"
    if (proto == "https" and str(port) in ("", "443")) or (proto == "http" and str(port) in ("", "80")):
        host = addr
    else:
        host = f"{addr}:{port}"
    path = _clean_path(base_path)
    return f"{proto}://{host}{path}".rstrip("/")


def build_share_url(names: List[str]) -> str:
    params = urlencode([("root", n) for n in names])
    query = f"?{params}" if params else ""
    base_url = _configured_base_url() or _base_url_from_headers() or _base_url_from_options()
    return f"{base_url}{query}" if base_url else query


def share_button(names: List[str], key: str) -> None:
    @st.dialog("Ссылка для доступа")
    def _show_dialog(url: str) -> None:
        st.text_input("URL", url, key=f"share_url_{key}")

    if st.button("🔗 Поделиться", key=key):
        try:
            st.query_params.clear()
            st.query_params["root"] = names
        except Exception:
            try:
                st.experimental_set_query_params(root=names)
            except Exception:
                pass
        url = build_share_url(names)
        _show_dialog(url)


# --------- Рисование PNG (уменьшаем шрифты и узлы) -----------------------

def _hierarchy_pos(G: nx.DiGraph, root: str):
    from collections import deque
    levels: Dict[int, List[str]] = {}
    q = deque([(root, 0)])
    seen = set()
    while q:
        n, d = q.popleft()
        if n in seen:
            continue
        seen.add(n)
        levels.setdefault(d, []).append(n)
        for c in G.successors(n):
            q.append((c, d + 1))
    pos: Dict[str, tuple[float, float]] = {}
    for depth, nodes in levels.items():
        width = len(nodes)
        for i, n in enumerate(nodes):
            x = (i + 1) / (width + 1)
            y = -depth
            pos[n] = (x, y)
    return pos


def draw_matplotlib(G: nx.DiGraph, root: str) -> plt.Figure:
    if G.number_of_nodes() == 0:
        fig = plt.figure(figsize=(6, 3.5))
        plt.axis("off")
        plt.text(0.5, 0.5, "Потомки не найдены", ha="center", va="center")
        return fig
    try:
        import networkx.drawing.nx_pydot as nx_pydot  # type: ignore
        pos = nx_pydot.graphviz_layout(G, prog="dot")
    except Exception:
        pos = _hierarchy_pos(G, root)
    fig = plt.figure(figsize=(max(6, len(G) * 0.45), 6))
    nx.draw(
        G,
        pos,
        with_labels=True,
        labels={n: multiline(n) for n in G.nodes},
        node_color="#ADD8E6",
        node_size=2000,   # было 3200 → немного меньше
        font_size=7,      # заметно меньше шрифт
        arrows=True,
    )
    plt.title(f"Академическая родословная – {root}", fontsize=10)
    plt.tight_layout()
    return fig


# --------- Интерактивная HTML-визуализация (уменьшаем шрифты) -----------

def build_pyvis_html(G: nx.DiGraph, root: str) -> str:
    net = Network(height="1000px", width="100%", directed=True, bgcolor="#ffffff")
    net.toggle_physics(True)

    children_map: Dict[str, List[str]] = {}
    nodes_payload: List[str] = []
    for n in G.nodes:
        node_id = str(n)
        nodes_payload.append(node_id)
        successors = [str(child) for child in G.successors(n)]
        if successors:
            children_map[node_id] = successors
        net.add_node(
            node_id,
            label=multiline(n),
            title=str(n),
            shape="box",
            color="#ADD8E6",
        )

    edges_payload: List[Dict[str, str]] = []
    for u, v in G.edges:
        src = str(u)
        dst = str(v)
        edges_payload.append({"from": src, "to": dst})
        net.add_edge(src, dst, arrows="to")

    vis_opts = {
        "nodes": {"font": {"size": 12}},  # шрифт поменьше
        "layout": {"hierarchical": {"direction": "UD", "sortMethod": "directed"}},
        "interaction": {"hover": True},
        "physics": {
            "hierarchicalRepulsion": {
                "nodeDistance": 140,
                "springLength": 160,
                "springConstant": 0.01,
            },
            "solver": "hierarchicalRepulsion",
            "stabilization": {"iterations": 200},
            "minVelocity": 0.1,
        },
    }
    net.set_options(json.dumps(vis_opts))

    try:
        html = net.generate_html()  # type: ignore[attr-defined]
    except Exception:
        tmp = Path("_tmp.html")
        net.save_graph(str(tmp))
        html = tmp.read_text(encoding="utf-8")
        try:
            tmp.unlink()
        except Exception:
            pass

    config = {
        "root": str(root),
        "childrenMap": children_map,
        "nodes": nodes_payload,
        "edges": edges_payload,
    }
    config_json = json.dumps(config, ensure_ascii=False)

    injection = textwrap.dedent(
        """
        <style>
          #mynetwork .branch-toggle-layer {
            position: absolute;
            inset: 0;
            pointer-events: none;
          }

          #mynetwork .branch-toggle {
            position: absolute;
            transform: translate(-50%, 0);
            border-radius: 50%;
            border: 1px solid #2d3f5f;
            background: #ffffff;
            color: #2d3f5f;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            pointer-events: auto;
            user-select: none;
            padding: 0;
            min-width: 16px;
            min-height: 16px;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
            transition: background-color 0.2s ease, color 0.2s ease;
            z-index: 10;
          }

          #mynetwork .branch-toggle:hover {
            background: #2d3f5f;
            color: #ffffff;
          }
        </style>
        <script>
        (function() {
          const config = __CONFIG_JSON__;
          const network = window.network;
          if (!network || !network.body || !network.body.data) {
            return;
          }
          const container = document.getElementById("mynetwork");
          if (!container) {
            return;
          }

          const childrenMap = config.childrenMap || {};
          const rootId = config.root;
          const originalNodes = Array.isArray(config.nodes) ? config.nodes : [];
          const originalEdges = Array.isArray(config.edges) ? config.edges : [];
          const originalNodeSet = new Set(originalNodes);
          const originalEdgeSet = new Set(
            originalEdges.map(function(edge) {
              return edge.from + "\u2192" + edge.to;
            })
          );

          const toggleLayer = document.createElement("div");
          toggleLayer.className = "branch-toggle-layer";
          container.appendChild(toggleLayer);

          const toggles = new Map();
          const collapsed = {};
          const descendantCache = {};

          function getDescendants(nodeId) {
            if (descendantCache[nodeId]) {
              return descendantCache[nodeId];
            }
            const result = [];
            const queue = (childrenMap[nodeId] || []).slice();
            const seen = new Set();
            while (queue.length) {
              const current = queue.shift();
              if (seen.has(current)) {
                continue;
              }
              seen.add(current);
              result.push(current);
              const children = childrenMap[current];
              if (children && children.length) {
                queue.push.apply(queue, children);
              }
            }
            descendantCache[nodeId] = result;
            return result;
          }

          function updateButton(nodeId) {
            const button = toggles.get(nodeId);
            if (!button) {
              return;
            }
            button.textContent = collapsed[nodeId] ? "+" : "\u2212";
            const node = network.body.data.nodes.get(nodeId);
            if (node && node.hidden) {
              button.style.display = "none";
            } else {
              button.style.display = "flex";
            }
          }

          function setNodesHidden(ids, hidden) {
            if (!ids.length) {
              return;
            }
            const updates = [];
            ids.forEach(function(id) {
              if (!originalNodeSet.has(id)) {
                return;
              }
              updates.push({ id: id, hidden: hidden });
            });
            if (updates.length) {
              network.body.data.nodes.update(updates);
            }
          }

          function setEdgesHidden(idSet, hidden) {
            if (!idSet.size) {
              return;
            }
            const updates = [];
            network.body.data.edges.forEach(function(edge) {
              const key = edge.from + "\u2192" + edge.to;
              if (!originalEdgeSet.has(key)) {
                return;
              }
              if (idSet.has(edge.from) || idSet.has(edge.to)) {
                updates.push({ id: edge.id, hidden: hidden });
              }
            });
            if (updates.length) {
              network.body.data.edges.update(updates);
            }
          }

          function hideBranch(nodeId) {
            if (!childrenMap[nodeId] || !childrenMap[nodeId].length) {
              return;
            }
            collapsed[nodeId] = true;
            const descendants = getDescendants(nodeId);
            const idSet = new Set(descendants);
            setNodesHidden(descendants, true);
            setEdgesHidden(idSet, true);
            descendants.forEach(function(id) {
              const button = toggles.get(id);
              if (button) {
                button.style.display = "none";
              }
            });
            updateButton(nodeId);
            window.requestAnimationFrame(updatePositions);
          }

          function showBranch(nodeId) {
            if (!childrenMap[nodeId] || !childrenMap[nodeId].length) {
              return;
            }
            collapsed[nodeId] = false;
            const descendants = getDescendants(nodeId);
            const idSet = new Set(descendants);
            setNodesHidden(descendants, false);
            setEdgesHidden(idSet, false);
            updateButton(nodeId);
            descendants.forEach(function(id) {
              updateButton(id);
            });
            descendants.forEach(function(id) {
              if (collapsed[id]) {
                hideBranch(id);
                const button = toggles.get(id);
                if (button) {
                  button.style.display = "flex";
                }
              }
            });
            if (descendants.length > 8) {
              network.stabilize();
            }
            window.requestAnimationFrame(updatePositions);
          }

          function toggleBranch(nodeId) {
            if (collapsed[nodeId]) {
              showBranch(nodeId);
            } else {
              hideBranch(nodeId);
            }
          }

          function updatePositions() {
            toggles.forEach(function(button, nodeId) {
              const node = network.body.data.nodes.get(nodeId);
              if (!node || node.hidden) {
                return;
              }
              const bounding = network.getBoundingBox(nodeId);
              if (!bounding) {
                return;
              }

              const bottomCenterCanvas = {
                x: (bounding.left + bounding.right) / 2,
                y: bounding.bottom,
              };
              const topLeftDom = network.canvasToDOM({
                x: bounding.left,
                y: bounding.top,
              });
              const bottomRightDom = network.canvasToDOM({
                x: bounding.right,
                y: bounding.bottom,
              });
              const bottomCenterDom = network.canvasToDOM(bottomCenterCanvas);

              const width = bottomRightDom.x - topLeftDom.x;
              const height = bottomRightDom.y - topLeftDom.y;
              let verticalOffset = 14;

              if (Number.isFinite(width) && Number.isFinite(height)) {
                const minDimension = Math.max(0, Math.min(width, height));
                const size = Math.max(16, Math.min(36, minDimension * 0.5));
                const roundedSize = Math.round(size);
                const fontSize = Math.max(9, Math.round(size * 0.45));
                button.style.width = roundedSize + "px";
                button.style.height = roundedSize + "px";
                button.style.fontSize = fontSize + "px";
                verticalOffset = Math.max(10, Math.round(roundedSize / 2 + 6));
              }

              button.style.left = bottomCenterDom.x + "px";
              button.style.top = bottomCenterDom.y + verticalOffset + "px";
            });
          }

          Object.keys(childrenMap).forEach(function(nodeId) {
            if (nodeId === rootId) {
              collapsed[nodeId] = false;
              return;
            }
            if (!childrenMap[nodeId] || !childrenMap[nodeId].length) {
              collapsed[nodeId] = false;
              return;
            }
            const button = document.createElement("button");
            button.type = "button";
            button.className = "branch-toggle";
            button.style.width = "20px";
            button.style.height = "20px";
            button.style.fontSize = "12px";
            button.textContent = "\u2212";
            button.title = "Свернуть/развернуть ветку";
            button.addEventListener("click", function(evt) {
              evt.preventDefault();
              evt.stopPropagation();
              toggleBranch(nodeId);
            });
            toggleLayer.appendChild(button);
            toggles.set(nodeId, button);
            collapsed[nodeId] = false;
            updateButton(nodeId);
          });

          if (!toggles.size) {
            return;
          }

          network.on("afterDrawing", updatePositions);
          network.once("stabilizationIterationsDone", function() {
            window.requestAnimationFrame(updatePositions);
          });
          window.addEventListener("resize", updatePositions);
          updatePositions();
        })();
        </script>
        """
    ).replace("__CONFIG_JSON__", config_json)

    if "</body>" in html:
        html = html.replace("</body>", f"{injection}\n</body>")
    else:
        html += injection

    return html


# ---------------------- Загрузка данных ----------------------------------
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    base = Path(DATA_DIR).expanduser().resolve()
    files = sorted(base.glob(CSV_GLOB))
    if not files:
        raise FileNotFoundError(f"В {base} не найдено CSV по маске '{CSV_GLOB}'")

    # простая авто‑детекция разделителя по первому файлу
    try:
        sample = pd.read_csv(files[0], nrows=5, dtype=str)
        sep = ";" if sample.shape[1] == 1 else ","
    except Exception:
        sep = ","

    frames = [pd.read_csv(f, dtype=str, keep_default_na=False, sep=sep) for f in files]
    return pd.concat(frames, ignore_index=True)


@st.cache_data(show_spinner=False)
def load_basic_scores() -> pd.DataFrame:
    base = Path(BASIC_SCORES_DIR).expanduser().resolve()
    files = sorted(base.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"В {base} не найдено CSV с тематическими профилями"
        )

    frames: list[pd.DataFrame] = []
    for file in files:
        frame = pd.read_csv(file)
        if "Code" not in frame.columns:
            raise KeyError(f"В файле {file.name} нет столбца 'Code'")
        frames.append(frame)

    scores = pd.concat(frames, ignore_index=True)
    scores = scores.dropna(subset=["Code"])
    scores["Code"] = scores["Code"].astype(str).str.strip()
    scores = scores[scores["Code"].str.len() > 0]
    scores = scores.drop_duplicates(subset="Code", keep="first")

    feature_columns = [c for c in scores.columns if c != "Code"]
    if not feature_columns:
        raise ValueError("Не найдены столбцы с тематическими компонентами")

    scores[feature_columns] = scores[feature_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    scores[feature_columns] = scores[feature_columns].fillna(0.0)

    return scores


def gather_school_dataset(
    df: pd.DataFrame,
    index: Dict[str, Set[int]],
    root: str,
    scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    _, subset = lineage(df, index, root)
    if subset.empty:
        empty = pd.DataFrame(columns=[*scores.columns, "school", AUTHOR_COLUMN])
        return empty, empty, 0

    working = subset[["Code", AUTHOR_COLUMN]].copy()
    working["Code"] = working["Code"].astype(str).str.strip()
    working = working[working["Code"].str.len() > 0]
    codes = working["Code"].unique().tolist()

    dataset = scores[scores["Code"].isin(codes)].copy()
    dataset["school"] = root
    dataset = dataset.merge(
        working.drop_duplicates(subset="Code"), on="Code", how="left"
    )

    missing_codes = sorted(set(codes) - set(dataset["Code"]))
    missing_info = (
        working[working["Code"].isin(missing_codes)]
        .drop_duplicates(subset="Code")
        .rename(columns={AUTHOR_COLUMN: "candidate_name"})
    )

    dataset = dataset.rename(columns={AUTHOR_COLUMN: "candidate_name"})
    if "candidate_name" not in dataset.columns:
        dataset["candidate_name"] = None

    return dataset, missing_info, len(codes)


def make_silhouette_plot(
    sample_scores: np.ndarray,
    labels: np.ndarray,
    school_order: list[str],
    overall_score: float,
    metric: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    y_lower = 10
    colors = [plt.cm.tab10(i) for i in range(len(school_order))]

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
            alpha=0.7,
        )
        ax.text(
            -0.98,
            y_lower + size / 2,
            f"{school} (n={size})",
            fontsize=10,
            va="center",
        )
        y_lower = y_upper + 10

    ax.axvline(x=overall_score, color="gray", linestyle="--", linewidth=1.5)
    ax.set_xlim([-1, 1])
    ax.set_xlabel("Коэффициент силуэта")
    ax.set_ylabel("Диссертации")
    ax.set_title(f"Silhouette plot (метрика: {metric})")
    ax.set_yticks([])
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    fig.tight_layout()
    return fig


# ====================== ИНТЕРФЕЙС (без технического сайдбара) ============
try:
    df = load_data()
except Exception as e:
    st.error(f"Ошибка при загрузке данных: {e}")
    st.stop()

# Проверяем обязательные колонки
missing = [c for c in [AUTHOR_COLUMN, *SUPERVISOR_COLUMNS] if c not in df.columns]
if missing:
    st.error("Отсутствуют нужные колонки: " + ", ".join(f"`{c}`" for c in missing))
    st.stop()

# Индекс для поиска по руководителям
idx = build_index(df, SUPERVISOR_COLUMNS)

# Список доступных руководителей для выбора
all_supervisor_names: Set[str] = set()
for col in SUPERVISOR_COLUMNS:
    all_supervisor_names.update({v for v in df[col].dropna().astype(str).unique() if v})

# Параметры из адресной строки (?root=...)
shared_roots = st.query_params.get_all("root")
valid_shared_roots = [r for r in shared_roots if r in all_supervisor_names]
manual_prefill = "\n".join(r for r in shared_roots if r not in all_supervisor_names)

tab_lineages, tab_silhouette = st.tabs(
    ["Построение деревьев", "Сравнение научных школ"]
)

with tab_lineages:
    st.subheader("Выбор научных руководителей для построения деревьев")
    roots = st.multiselect(
        "Выберите имена из базы",
        options=sorted(all_supervisor_names),
        default=valid_shared_roots,  # если пришли по ссылке, подставляем имена
        help="Список формируется из столбцов с руководителями",
    )
    manual = st.text_area(
        "Или добавьте имена вручную в формате: Фамилия Имя Отчество (по одному на строку)",
        height=120,
        value=manual_prefill,
    )
    manual_list = [r.strip() for r in manual.splitlines() if r.strip()]
    roots = list(dict.fromkeys([*roots, *manual_list]))  # убрать дубликаты, сохранить порядок

    build_clicked = st.button("Построить деревья", type="primary", key="build_trees")
    if build_clicked or shared_roots:
        st.session_state["built"] = True
    build = st.session_state.get("built", False)

    tree_option_labels = [label for label, _, _ in TREE_OPTIONS]
    selected_tree_labels = st.multiselect(
        "Типы деревьев для построения",
        options=tree_option_labels,
        default=[tree_option_labels[0]],
        help="Фильтрация по степени применяется только к первому уровню относительно выбранного руководителя.",
    )
    selected_tree_labels = selected_tree_labels or [tree_option_labels[0]]
    selected_tree_configs = [opt for opt in TREE_OPTIONS if opt[0] in selected_tree_labels]
    export_md_outline = st.checkbox("Также сохранить оглавление (.md)", value=False)

    if build:
        if not roots:
            st.warning("Пожалуйста, выберите или введите хотя бы одно имя руководителя.")
        else:
            all_zip_buf = io.BytesIO()
            zf = zipfile.ZipFile(all_zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED)

            for root in roots:
                st.markdown("---")
                st.subheader(f"▶ {root}")

                tree_results = []
                for label, suffix, first_level_filter in selected_tree_configs:
                    G, subset = lineage(
                        df, idx, root, first_level_filter=first_level_filter
                    )
                    tree_results.append(
                        {
                            "label": label,
                            "suffix": suffix,
                            "graph": G,
                            "subset": subset,
                        }
                    )

                root_slug = slug(root)
                person_entries: List[tuple[str, bytes]] = []
                has_content = False

                for tree in tree_results:
                    label = tree["label"]
                    suffix = tree["suffix"]
                    G = tree["graph"]
                    subset = tree["subset"]

                    if G.number_of_edges() == 0:
                        st.info(
                            f"{label}: потомки не найдены для выбранного типа дерева."
                        )
                        continue

                    has_content = True
                    st.markdown(f"#### 🌳 {label}")

                    fig = draw_matplotlib(G, root)
                    png_buf = io.BytesIO()
                    fig.savefig(png_buf, format="png", dpi=300, bbox_inches="tight")
                    png_bytes = png_buf.getvalue()

                    st.image(png_bytes, caption="Миниатюра PNG", width=220)

                    html = build_pyvis_html(G, root)
                    st.components.v1.html(html, height=800, width=2000, scrolling=True)
                    html_bytes = html.encode("utf-8")

                    csv_bytes = subset.to_csv(
                        index=False, encoding="utf-8-sig"
                    ).encode("utf-8-sig")

                    md_bytes = None
                    if export_md_outline:
                        out_lines: List[str] = []

                        def walk(n: str, d: int = 0) -> None:
                            out_lines.append(f"{'  ' * d}- {n}")
                            for c in G.successors(n):
                                walk(c, d + 1)

                        walk(root)
                        md_bytes = ("\n".join(out_lines)).encode("utf-8")

                    file_prefix = (
                        root_slug if suffix == "general" else f"{root_slug}.{suffix}"
                    )

                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.download_button(
                            "Скачать PNG",
                            data=png_bytes,
                            file_name=f"{file_prefix}.png",
                            mime="image/png",
                            key=f"png_{file_prefix}",
                        )
                    with c2:
                        st.download_button(
                            "Скачать HTML",
                            data=html_bytes,
                            file_name=f"{file_prefix}.html",
                            mime="text/html",
                            key=f"html_{file_prefix}",
                        )
                    with c3:
                        st.download_button(
                            "Скачать выборку CSV",
                            data=csv_bytes,
                            file_name=f"{file_prefix}.sampling.csv",
                            mime="text/csv",
                            key=f"csv_{file_prefix}",
                        )
                    with c4:
                        if md_bytes is not None:
                            st.download_button(
                                "Скачать оглавление .md",
                                data=md_bytes,
                                file_name=f"{file_prefix}.xmind.md",
                                mime="text/markdown",
                                key=f"md_{file_prefix}",
                            )
                        else:
                            st.empty()

                    person_entries.append((f"{file_prefix}.png", png_bytes))
                    person_entries.append((f"{file_prefix}.html", html_bytes))
                    person_entries.append((f"{file_prefix}.sampling.csv", csv_bytes))
                    zf.writestr(f"{file_prefix}.png", png_bytes)
                    zf.writestr(f"{file_prefix}.html", html_bytes)
                    zf.writestr(f"{file_prefix}.sampling.csv", csv_bytes)
                    if md_bytes is not None:
                        person_entries.append((f"{file_prefix}.xmind.md", md_bytes))
                        zf.writestr(f"{file_prefix}.xmind.md", md_bytes)

                if not has_content:
                    continue

                person_zip: bytes | None = None
                if person_entries:
                    person_zip_buf = io.BytesIO()
                    try:
                        with zipfile.ZipFile(
                            person_zip_buf,
                            mode="w",
                            compression=zipfile.ZIP_DEFLATED,
                        ) as z_person:
                            for filename, data in person_entries:
                                z_person.writestr(filename, data)
                        person_zip = person_zip_buf.getvalue()
                    except Exception:
                        person_zip = None

                col_zip_person, col_share_person = st.columns([3, 1])
                with col_zip_person:
                    if person_zip is not None:
                        st.download_button(
                            label="⬇️ Скачать всё архивом (ZIP)",
                            data=person_zip,
                            file_name=f"{root_slug}.zip",
                            mime="application/zip",
                            key=f"zip_{root_slug}",
                        )
                with col_share_person:
                    share_button([root], key=f"share_{root_slug}")

            zf.close()
            if all_zip_buf.getbuffer().nbytes > 0:
                col_zip, col_share = st.columns([3, 1])
                with col_zip:
                    st.download_button(
                        label="⬇️ Скачать всё архивом (ZIP)",
                        data=all_zip_buf.getvalue(),
                        file_name="lineages_export.zip",
                        mime="application/zip",
                    )
                with col_share:
                    share_button(roots, key="share_all")
    else:
        st.info(
            "Выберите или добавьте имена руководителей и нажмите ‘Построить деревья’."
        )

with tab_silhouette:
    st.subheader("Сравнение научных школ по тематическим профилям")
    st.write(
        "Введите двух научных руководителей, чтобы сравнить тематические профили их "
        "школ по коэффициенту силуэта. Потомки берутся из базы родословных, а оценки "
        "тематических векторов — из CSV-файлов в папке `basic_scores`."
    )

    select_options = ["—"] + sorted(all_supervisor_names)
    col_left, col_right = st.columns(2)
    with col_left:
        root_a_choice = st.selectbox(
            "Персона 1",
            options=select_options,
            index=0,
            help="Можно выбрать имя из базы или ввести вручную ниже.",
            key="silhouette_root_a_choice",
        )
        root_a_manual = st.text_input(
            "Персона 1 (ручной ввод)",
            value="",
            placeholder="Фамилия Имя Отчество",
            key="silhouette_root_a_manual",
        )
    with col_right:
        root_b_choice = st.selectbox(
            "Персона 2",
            options=select_options,
            index=0,
            help="Можно выбрать имя из базы или ввести вручную ниже.",
            key="silhouette_root_b_choice",
        )
        root_b_manual = st.text_input(
            "Персона 2 (ручной ввод)",
            value="",
            placeholder="Фамилия Имя Отчество",
            key="silhouette_root_b_manual",
        )

    def _resolve_root(choice: str, manual: str) -> str:
        manual_clean = manual.strip()
        if manual_clean:
            return manual_clean
        choice_clean = choice.strip()
        return choice_clean if choice_clean and choice_clean != "—" else ""

    root_a = _resolve_root(root_a_choice, root_a_manual)
    root_b = _resolve_root(root_b_choice, root_b_manual)

    metric_options = {
        "cosine": "Косинусное расстояние",
        "euclidean": "Евклидово расстояние",
    }
    metric_key = st.selectbox(
        "Метрика расстояния",
        options=list(metric_options.keys()),
        format_func=lambda m: metric_options[m],
        index=0,
        key="silhouette_metric",
    )

    run_analysis = st.button(
        "Рассчитать коэффициент силуэта",
        type="primary",
        key="run_silhouette_analysis",
    )

    if run_analysis:
        if not root_a or not root_b:
            st.warning("Укажите две разные научные школы для сравнения.")
        elif root_a == root_b:
            st.warning("Для анализа нужны две разные персоналии.")
        else:
            with st.spinner("Выполняется анализ тематических профилей..."):
                try:
                    scores_df = load_basic_scores()
                except Exception as exc:
                    st.error(f"Не удалось загрузить тематические профили: {exc}")
                    scores_df = None

                if scores_df is not None:
                    dataset_a, missing_a, total_a = gather_school_dataset(
                        df, idx, root_a, scores_df
                    )
                    dataset_b, missing_b, total_b = gather_school_dataset(
                        df, idx, root_b, scores_df
                    )

                    messages: list[str] = []
                    if total_a == 0:
                        messages.append(
                            f"Для школы {root_a} не найдены диссертации-потомки в базе."
                        )
                    if total_b == 0:
                        messages.append(
                            f"Для школы {root_b} не найдены диссертации-потомки в базе."
                        )
                    if messages:
                        st.warning("\n".join(messages))

                    combined = pd.concat([dataset_a, dataset_b], ignore_index=True)
                    if combined.empty:
                        st.info(
                            "Недостаточно данных для расчёта коэффициента силуэта."
                        )
                    else:
                        feature_columns = [c for c in scores_df.columns if c != "Code"]
                        combined = combined.dropna(subset=["Code"])
                        combined = combined.drop_duplicates(subset="Code")
                        combined["candidate_name"] = combined["candidate_name"].fillna("")
                        combined["school"] = combined["school"].astype(str)

                        label_mapping = {root_a: 0, root_b: 1}
                        combined["label"] = combined["school"].map(label_mapping)

                        analysis_valid = True
                        if combined["label"].isna().any():
                            st.error(
                                "Не удалось сопоставить все диссертации с выбранными школами."
                            )
                            analysis_valid = False

                        if analysis_valid and combined["label"].nunique() < 2:
                            st.warning(
                                "В выборку попали диссертации только одной школы — сравнение невозможно."
                            )
                            analysis_valid = False

                        if analysis_valid and len(combined) < 2:
                            st.warning("Недостаточно диссертаций для вычисления метрики.")
                            analysis_valid = False

                        if analysis_valid:
                            X = combined[feature_columns].to_numpy(dtype=float)
                            labels_array = combined["label"].to_numpy(dtype=int)

                            try:
                                sample_scores = silhouette_samples(
                                    X, labels_array, metric=metric_key
                                )
                                overall_score = float(
                                    silhouette_score(
                                        X, labels_array, metric=metric_key
                                    )
                                )
                            except Exception as exc:
                                st.error(
                                    f"Ошибка при вычислении коэффициента силуэта: {exc}"
                                )
                                analysis_valid = False

                        if analysis_valid:
                            combined["silhouette"] = sample_scores

                            counts = combined["school"].value_counts()
                            mean_by_school = combined.groupby("school")[
                                "silhouette"
                            ].mean()

                            summary_cols = st.columns(3)
                            with summary_cols[0]:
                                st.metric(
                                    "Средний silhouette (общий)",
                                    f"{overall_score:.3f}",
                                    delta=f"n={len(combined)}",
                                )
                            with summary_cols[1]:
                                st.metric(
                                    f"{root_a}",
                                    f"{mean_by_school.get(root_a, float('nan')):.3f}",
                                    delta=f"n={counts.get(root_a, 0)}",
                                )
                            with summary_cols[2]:
                                st.metric(
                                    f"{root_b}",
                                    f"{mean_by_school.get(root_b, float('nan')):.3f}",
                                    delta=f"n={counts.get(root_b, 0)}",
                                )

                            if not missing_a.empty or not missing_b.empty:
                                warn_parts: list[str] = []
                                if not missing_a.empty:
                                    warn_parts.append(
                                        f"У школы {root_a} нет тематических профилей для {len(missing_a)} диссертаций."
                                    )
                                if not missing_b.empty:
                                    warn_parts.append(
                                        f"У школы {root_b} нет тематических профилей для {len(missing_b)} диссертаций."
                                    )
                                st.warning("\n".join(warn_parts))

                                with st.expander(
                                    "Показать диссертации без тематических оценок"
                                ):
                                    if not missing_a.empty:
                                        st.markdown(f"**{root_a}**")
                                        st.dataframe(
                                            missing_a.rename(
                                                columns={
                                                    "Code": "Код диссертации",
                                                    "candidate_name": "Автор",
                                                }
                                            ),
                                            use_container_width=True,
                                        )
                                    if not missing_b.empty:
                                        st.markdown(f"**{root_b}**")
                                        st.dataframe(
                                            missing_b.rename(
                                                columns={
                                                    "Code": "Код диссертации",
                                                    "candidate_name": "Автор",
                                                }
                                            ),
                                            use_container_width=True,
                                        )

                            display_df = (
                                combined[
                                    ["Code", "candidate_name", "school", "silhouette"]
                                ]
                                .sort_values(
                                    by=["school", "silhouette"],
                                    ascending=[True, False],
                                )
                                .rename(
                                    columns={
                                        "Code": "Код диссертации",
                                        "candidate_name": "Автор",
                                        "school": "Школа",
                                        "silhouette": "Silhouette",
                                    }
                                )
                            )
                            display_df["Silhouette"] = display_df["Silhouette"].map(
                                lambda x: round(float(x), 4)
                            )

                            st.markdown("### Подробности по диссертациям")
                            st.dataframe(display_df, use_container_width=True)

                            csv_bytes = display_df.to_csv(
                                index=False, encoding="utf-8-sig"
                            ).encode("utf-8-sig")
                            slug_a = slug(root_a) or "school_a"
                            slug_b = slug(root_b) or "school_b"
                            st.download_button(
                                "Скачать таблицу (CSV)",
                                data=csv_bytes,
                                file_name=f"silhouette_{slug_a}_vs_{slug_b}.csv",
                                mime="text/csv",
                                key="silhouette_download_csv",
                            )

                            st.markdown("### Диаграмма силуэта")
                            fig = make_silhouette_plot(
                                sample_scores,
                                labels_array,
                                [root_a, root_b],
                                overall_score,
                                metric_key,
                            )
                            st.pyplot(fig, use_container_width=True)
                            plot_buf = io.BytesIO()
                            fig.savefig(
                                plot_buf, format="png", dpi=300, bbox_inches="tight"
                            )
                            st.download_button(
                                "Скачать диаграмму (PNG)",
                                data=plot_buf.getvalue(),
                                file_name=f"silhouette_{slug_a}_vs_{slug_b}.png",
                                mime="image/png",
                                key="silhouette_download_plot",
                            )

