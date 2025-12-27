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
from typing import Callable, Dict, List, Literal, Optional, Set, Tuple
    
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
from urllib.parse import urlencode, urlsplit
try:
    import openpyxl
except ImportError:
    openpyxl = None
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
except Exception:  # pragma: no cover - совместимость со старыми версиями streamlit
    get_script_run_ctx = None  # type: ignore
import zipfile
from pyvis.network import Network
from sklearn.metrics import silhouette_samples, silhouette_score

from school_comparison_tab import render_school_comparison_tab

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
    "https://academic-genealogy.streamlit.app/",
).strip().rstrip("/")

# ---------------------- Тематический классификатор ------------------------
ClassifierItem = Tuple[str, str, bool]

THEMATIC_CLASSIFIER: List[ClassifierItem] = [
    ("1.1", "Среда и предметная область", True),
    ("1.1.1", "Уровень формального образования", True),
    ("1.1.1.1", "Дошкольное образование", False),
    ("1.1.1.2", "Школьное образование", False),
    ("1.1.1.2.1", "Начальное общее образование", False),
    ("1.1.1.2.2", "Основное общее образование", False),
    ("1.1.1.2.3", "Среднее общее образование", False),
    ("1.1.1.3", "Среднее профессиональное образование (СПО)", False),
    ("1.1.1.4", "Высшее образование", False),
    ("1.1.1.4.1", "Высшее образование – Бакалавриат", False),
    ("1.1.1.4.2", "Высшее образование – Специалитет", False),
    ("1.1.1.4.3", "Высшее образование – Магистратура", False),
    ("1.1.1.5", "Подготовка кадров высшей квалификации", False),
    ("1.1.1.5.1", "Подготовка кадров высшей квалификации – Аспирантура", False),
    ("1.1.1.5.2", "Подготовка кадров высшей квалификации – Ординатура", False),
    ("1.1.1.6", "Дополнительное профессиональное образование", False),
    ("1.1.1.6.1", "Дополнительное профессиональное образование – Повышение квалификации", False),
    ("1.1.1.6.2", "Дополнительное профессиональное образование – Профессиональная переподготовка", False),
    ("1.1.1.7", "Дополнительное образование детей и взрослых", False),
    ("1.1.1.8", "Профессиональное обучение", False),
    ("1.1.2", "Предметный цикл / Область знания", True),
    ("1.1.2.1", "Математические и естественнонаучные дисциплины", False),
    ("1.1.2.1.1", "Математика", False),
    ("1.1.2.1.2", "Информатика", False),
    ("1.1.2.1.3", "Физика", False),
    ("1.1.2.1.4", "Химия", False),
    ("1.1.2.1.5", "Биология", False),
    ("1.1.2.1.6", "Другие естественнонаучные дисциплины", False),
    ("1.1.2.2", "Гуманитарные и социальные дисциплины", False),
    ("1.1.2.2.1", "Филология", False),
    ("1.1.2.2.1.1", "Языки", False),
    ("1.1.2.2.1.1.1", "Русский язык", False),
    ("1.1.2.2.1.1.2", "Язык народов России", False),
    ("1.1.2.2.1.1.3", "Иностранный язык", False),
    ("1.1.2.2.1.1.3.1", "Английский язык", False),
    ("1.1.2.2.1.1.3.2", "Немецкий язык", False),
    ("1.1.2.2.1.1.3.3", "Французский язык", False),
    ("1.1.2.2.1.1.3.4", "Иной иностранный язык", False),
    ("1.1.2.2.1.2", "Литература", False),
    ("1.1.2.2.2", "История", False),
    ("1.1.2.2.3", "Обществознание, социология", False),
    ("1.1.2.2.4", "Социально-экономическая география", False),
    ("1.1.2.2.5", "Экономика", False),
    ("1.1.2.2.6", "Право", False),
    ("1.1.2.2.7", "Другие гуманитарные и общественные дисциплины", False),
    ("1.1.2.3", "Изобразительное и музыкальное искусство", False),
    ("1.1.2.4", "Технология", False),
    ("1.1.2.5", "Физическая культура и спорт", False),
    ("1.1.2.6", "Инженерно-техническое направление", False),
    ("1.1.2.7", "Психолого-педагогическое направление", False),
    ("1.1.2.8", "Междисциплинарные и надпредметные области", False),
    ("1.2", "Субъект образования (когорта)", True),
    ("1.2.1", "Социально-профессиональная группа", True),
    ("1.2.1.1", "Обучающиеся общеобразовательных организаций", False),
    ("1.2.1.2", "Студенты колледжей и техникумов (СПО)", False),
    ("1.2.1.3", "Студенты вузов (ВО)", False),
    ("1.2.1.4", "Действующие специалисты (в системе ДПО)", False),
    ("1.2.1.5", "Педагогические и научные кадры (как объект развития)", False),
    ("1.2.1.6", "Военнослужащие / сотрудники силовых структур", False),
    ("1.2.1.7", "Спортсмены", False),
    ("1.2.2", "Специфические образовательные потребности", True),
    ("1.2.2.1", "Обучающиеся с ОВЗ и/или инвалидностью", False),
    ("1.2.2.2", "Одаренные и высокомотивированные обучающиеся", False),
    ("1.2.2.3", "Обучающиеся из групп социального риска", False),
    ("1.2.2.4", "Инофоны / образовательные мигранты", False),
    ("2.1", "Тип и масштаб разрабатываемого решения", True),
    ("2.1.1", "Уровень абстракции", True),
    ("2.1.1.1", "Теоретико-методологическая концепция", False),
    ("2.1.1.2", "Структурно-функциональная модель", False),
    ("2.1.1.3", "Организационно-педагогическая система", False),
    ("2.1.1.4", "Педагогическая технология", False),
    ("2.1.1.5", "Частная методика преподавания", False),
    ("2.1.1.6", "Комплекс педагогических условий", False),
    ("2.1.1.7", "Образовательная/воспитательная программа", False),
    ("2.1.2", "Доминирующая педагогическая парадигма", True),
    ("2.1.2.1", "Знаниевая (традиционная)", False),
    ("2.1.2.2", "Компетентностная", False),
    ("2.1.2.3", "Системно-деятельностная", False),
    ("2.1.2.4", "Личностно-ориентированная / Гуманистическая", False),
    ("2.1.2.5", "Аксиологическая (ценностная)", False),
    ("2.1.2.6", "Контекстная / Проблемная", False),
    ("2.2", "Технологии и методы педагогического воздействия", True),
    ("2.2.1", "Ведущая форма организации деятельности", True),
    ("2.2.1.1", "Репродуктивная учебная деятельность", False),
    ("2.2.1.2", "Проектная деятельность", False),
    ("2.2.1.3", "Исследовательская деятельность", False),
    ("2.2.1.4", "Игровая деятельность (дидактическая, деловая, ролевая)", False),
    ("2.2.1.5", "Дискуссионные формы (дискуссия, дебаты)", False),
    ("2.2.1.6", "Художественно-творческая деятельность", False),
    ("2.2.1.7", "Спортивно-тренировочная деятельность", False),
    ("2.2.2", "Используемые средства обучения", True),
    ("2.2.2.1", "Текстовые и печатные средства (учебник, рабочая тетрадь)", False),
    ("2.2.2.2", "Аудиовизуальные средства (видео, презентации)", False),
    ("2.2.2.3", "Интерактивные цифровые ресурсы (тренажеры, симуляторы, ЭОР)", False),
    ("2.2.2.4", "Средства виртуальной и дополненной реальности (VR/AR)", False),
    ("2.2.2.5", "Платформенные решения (LMS, онлайн-курсы)", False),
    ("2.2.2.6", "Средства геймификации", False),
    ("2.2.3", "Доминирующий тип педагогического взаимодействия", True),
    ("2.2.3.1", "Прямое управление (инструктаж, объяснение)", False),
    ("2.2.3.2", "Фасилитация / Модерация", False),
    ("2.2.3.3", "Тьюторское сопровождение / Наставничество", False),
    ("2.2.3.4", "Консультирование", False),
    ("3.1", "Когнитивная сфера", True),
    ("3.1.1", "Предметные знания и представления", False),
    ("3.1.2", "Методологические знания", False),
    ("3.1.3", "Критическое мышление", False),
    ("3.1.4", "Креативное (творческое) мышление", False),
    ("3.1.5", "Системное / Проектное мышление", False),
    ("3.1.6", "Научное / Профессиональное мировоззрение", False),
    ("3.2", "Деятельностно-практическая сфера", True),
    ("3.2.1", "Предметные умения и навыки (Hard Skills)", False),
    ("3.2.2", "Универсальные навыки (Soft Skills)", False),
    ("3.2.2.1", "Коммуникативные навыки", False),
    ("3.2.2.2", "Навыки кооперации и работы в команде", False),
    ("3.2.2.3", "Навыки самоорганизации и тайм-менеджмента", False),
    ("3.2.2.4", "Навыки решения проблем (problem-solving)", False),
    ("3.2.3", "Профессиональные компетенции", False),
    ("3.2.4", "Метапредметные / Ключевые компетенции", False),
    ("3.2.5", "Социальная компетентность", False),
    ("3.3", "Личностно-ценностная сфера", True),
    ("3.3.1", "Мотивация (учебная, профессиональная)", False),
    ("3.3.2", "Ценностные ориентации", False),
    ("3.3.3", "Волевая саморегуляция", False),
    ("3.3.4", "Рефлексивные способности", False),
    ("3.3.5", "Интегративные личностные конструкты", True),
    ("3.3.5.1", "Готовность (к деятельности, самообразованию и др.)", False),
    ("3.3.5.2", "Субъектность / Субъектная позиция", False),
    ("3.3.5.3", "Идентичность (профессиональная, гражданская)", False),
    ("3.3.5.4", "Патриотизм / Гражданственность", False),
    ("3.3.6", "Формируемый тип культуры личности", True),
    ("3.3.6.1", "Информационная культура", False),
    ("3.3.6.2", "Правовая культура", False),
    ("3.3.6.3", "Экологическая культура", False),
    ("3.3.6.4", "Физическая культура", False),
    ("3.3.6.5", "Профессиональная культура", False),
    ("3.3.6.6", "Культура безопасности жизнедеятельности", False),
]

CLASSIFIER_BY_CODE: Dict[str, ClassifierItem] = {
    code: (code, title, disabled) for code, title, disabled in THEMATIC_CLASSIFIER
}

PROFILE_SELECTION_SESSION_KEY = "profile_selected_codes"
PROFILE_SELECTION_LIMIT = 5
PROFILE_MIN_SCORE = 4.0


def classifier_depth(code: str) -> int:
    return code.count(".") if code else 0


def classifier_format(option: Optional[ClassifierItem]) -> str:
    if option is None:
        return "— выберите пункт —"
    code, title, disabled = option
    indent = "\u2003" * classifier_depth(code)
    label = f"{code} {title}"
    if disabled:
        label += " (нельзя выбрать)"
    return f"{indent}{label}"


def classifier_label(code: str) -> str:
    item = CLASSIFIER_BY_CODE.get(code)
    if not item:
        return code
    _, title, _ = item
    return f"{code} · {title}"

# ---------------------- Инструкции для вкладок ---------------------------------

INSTRUCTIONS = {
    "lineages": """
## Вкладка «Построение деревьев»

На этой вкладке осуществляется построение деревьев научного руководства (связь «научный руководитель/консультант – диссертант»). Возможно одновременное построение деревьев для нескольких научных руководителей. Возможны некоторые неточности в случае совпадения ФИО руководителей или диссертантов.

### 1. Выбор руководителя

В поле «Выберите имена из базы» начните вводить фамилию интересующего ученого и выберите его из списка. Можно выбрать нескольких персоналий сразу.

### 2. Опции

**Настройка типа дерева:**
- **Общее дерево.** Показывает всех защитившихся учеников.
- **Дерево докторов наук.** Оставляет только тех учеников, кто защитил докторскую диссертацию (фильтр применяется к первому уровню).
- **Дерево кандидатов наук.** Оставляет только кандидатов наук.

Галочка **«Также сохранить оглавление (.md)»** создаст текстовый файл со списком иерархии. Данные в формате markdown (.md) можно использовать для построения красивого дерева научного руководства в программе XMind.

Для запуска процесса нажмите кнопку **«Построить деревья»**.

### 3. Результат

Построенное дерево научного руководства будет доступно в нескольких форматах:
- Статичная картинка (PNG) для вставки в статьи.
- Интерактивный граф: можно уменьшать/увеличивать масштаб, двигать узлы, сворачивать/разворачивать ветки (нажимая на кнопки − и + на графе).
- Доступны кнопки для скачивания результатов (картинка, HTML-файл, таблица данных CSV и XLSX).
- Внизу есть кнопка **«🔗 Поделиться»**, которая генерирует ссылку на текущий выбор ученых.
""",
    
    "dissertations": """
## Вкладка «Поиск информации о диссертациях»

На этой вкладке доступен поиск диссертаций по формальным критериям. Код диссертации соответствует коду автореферата диссертации на сайте Национальной электронной библиотеки ([http://rusneb.ru](http://rusneb.ru)).

### 1. Выбор критериев поиска

В поле «Выберите критерии поиска» отметьте необходимые параметры фильтрации (максимум 5 критериев одновременно). Среди доступных параметров: Название диссертации, ФИО автора, ФИО научного руководителя, ФИО оппонента, Организация выполнения, Ведущая организация, Место защиты, Город защиты, Год защиты, Специальность.

### 2. Ввод данных

- После выбора критериев появятся соответствующие поля для ввода.
- Для текстовых полей (например, ФИО или Организация) достаточно ввести часть названия или фамилии – поиск работает по вхождению текста без учета регистра.
- Для полей «Год защиты», «Город» и «Специальность» доступен выбор из выпадающего списка существующих значений.
- Для запуска поиска нажмите кнопку **«Найти»**.

### 3. Результат

Система выведет таблицу со списком найденных диссертаций, содержащую полные выходные данные работ.
- Таблица отображает информацию о диссертационном исследовании, доступную на первых двух страницах автореферата.
- Полученную выборку можно сохранить на устройство, нажав кнопку **«Скачать результаты»**, выбрав формат CSV или XLSX.
""",
    
    "profiles": """
## Вкладка «Поиск по тематическим профилям»

На этой вкладке реализован содержательный поиск диссертаций. Он основан не на совпадении слов в заголовке, а на анализе всего текста автореферата диссертационной работы. Поиск осуществляется с использованием иерархического классификатора, содержащего различные критерии, отражающие объект, процесс и результат диссертационного исследования.

### 1. Выбор тематики

В выпадающем списке «Элемент классификатора» выберите интересующую тему, метод или педагогическую технологию (например, «Начальное общее образование», «Информатика», «Интерактивные цифровые ресурсы», «Инклюзия»). Нажмите кнопку **«Добавить в подборку»**.

### 2. Логика поиска

- Можно добавить в подборку от одного до пяти пунктов классификатора.
- Система отберет только те диссертации, у которых оценка присутствия каждого выбранного пункта (темы) составляет не менее 4 баллов (по 10-балльной шкале).
- Для запуска алгоритма нажмите кнопку **«Найти диссертации»**.

### 3. Результат

Вы получите список работ, наиболее полно раскрывающих выбранные темы.
- Результаты ранжируются по сумме баллов: сверху списка находятся диссертации, в которых искомые темы проработаны максимально глубоко.
- Доступен фильтр по результатам. Поле **«🔍 Фильтр по таблице»** позволяет найти работу по автору или слову в названии внутри полученной выборки. Например, введя «Московский городской педагогический университет», можно отфильтровать диссертации, связанные с МГПУ.
- Итоговую таблицу с баллами и метаданными можно выгрузить, нажав кнопку **«Скачать результаты»**. Чем выше балл, тем в большей степени тематика диссертации соответствует тем содержательным критериям, по которым осуществлялся поиск.
"""
}

def show_instruction(tab_key: str) -> None:
    """Показывает инструкцию для конкретной вкладки в модальном окне."""
    @st.dialog("📖 Инструкция", width="large")
    def _show_dialog():
        st.markdown(INSTRUCTIONS.get(tab_key, "Инструкция не найдена."))
    
    _show_dialog()

def download_data_dialog(df: pd.DataFrame, file_base: str, key_prefix: str) -> None:
    """
    Открывает модальное окно для выбора формата скачивания (XLSX или CSV).
    XLSX предлагается первым и выделен как основной вариант.
    """
    @st.dialog(f"Скачать данные: {file_base}")
    def _show_dialog():
        st.write("Выберите удобный формат:")
        
        # Подготовка Excel
        try:
            buf_xlsx = io.BytesIO()
            with pd.ExcelWriter(buf_xlsx, engine="openpyxl") as writer:
                df.to_excel(writer, index=False)
            data_xlsx = buf_xlsx.getvalue()
            
            st.download_button(
                label="📊 Скачать Excel (.xlsx)",
                data=data_xlsx,
                file_name=f"{file_base}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"{key_prefix}_dl_xlsx",
                type="primary",  # Делаем кнопку акцентной
                use_container_width=True
            )
        except ImportError:
            st.error("Для скачивания в Excel установите библиотеку openpyxl.")
        except Exception as e:
            st.error(f"Ошибка формирования Excel: {e}")

        # Подготовка CSV
        data_csv = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            label="📄 Скачать CSV (.csv)",
            data=data_csv,
            file_name=f"{file_base}.csv",
            mime="text/csv",
            key=f"{key_prefix}_dl_csv",
            use_container_width=True
        )
    
    _show_dialog()
    
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


ComparisonScope = Literal["direct", "all"]


def gather_school_dataset(
    df: pd.DataFrame,
    index: Dict[str, Set[int]],
    root: str,
    scores: pd.DataFrame,
    scope: ComparisonScope = "direct",
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    if scope == "direct":
        subset = rows_for(df, index, root)
    elif scope == "all":
        _, subset = lineage(df, index, root)
    else:  # pragma: no cover - защитная ветка
        raise ValueError(f"Неизвестный режим сравнения: {scope}")
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


SILHOUETTE_HELP_TEXT = """
Силуэтные графики визуализируют, насколько хорошо разделены кластеры диссертаций. В нашем случае: насколько диссертации одной научной школы тематически и содержательно близки к диссертация другой научной школы.

Горизонтальная ось – это значение коэффициента силуэта.

Вертикальная ось – отдельные диссертации, сгруппированные по кластерам (научным руководителям). «Лезвие» графика представляет собой один кластер.

Ширина «лезвия» показывает количество работ в кластере.

Вертикальная пунктирная линия – это среднее значение коэффициента силуэта для всех работ. Это интегральная метрика, которая оценивает качество кластеризации, то есть насколько хорошо работы сгруппированы. Значение варьируется от -1 до +1.
- Значение, близкое к +1 указывает на то, что кластеры плотные, четко разделены, и работы внутри одной группы очень похожи друг на друга, но сильно отличаются от работ в других группах. В нашем случае: между диссертациями двух научных школ практически отсутствуют тематические и содержательные пересечения.
- Значение, близкое к 0, говорит о том, что кластеры сильно пересекаются, и тематические границы между ними размыты.
- Отрицательное значение (близкое к -1) указывает на то, что работы, вероятно, были отнесены к неверному кластеру. Это указывает на очень высокую тематическую близость и отсутствие четкого разделения.  В нашем случае: диссертации двух научных школ тематически и содержательно очень лизки друг к другу.

Горизонтальные линии, образующие «лезвие» – это элементы кластера. В нашем случае одна горизонтальная лина = одна диссертация.

Длина горизонтальной линии = степень соответствия:
- Длинная линия вправо (значение близко к +1) означает, что диссертация – образцовый представитель своей группы. Ее тема очень близка к другим работам в этой же школе и сильно отличается от тем в сравниваемой школе. Она находится в «ядре» кластера.
- Короткая линия (значение близко к 0) означает, что диссертация находится на «границе». Ее тема расположена на стыке двух научных школ. Она одинаково (не)похожа как на своих, так и на чужих.
- Линия, уходящая влево в отрицательную зону (значение < 0) означает, что это «аномалия» или признак сильного смешения. Эта диссертация, хотя формально и принадлежит к своей группе, по своему содержанию (тематическому профилю) оказалась ближе к центру другого кластера.
"""


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
            f"{school} (кол-во: {size})",
            fontsize=10,
            va="center",
        )
        y_lower = y_upper + 10

    ax.axvline(x=overall_score, color="gray", linestyle="--", linewidth=1.5)
    ax.set_xlim([-1, 1])
    ax.set_xlabel("Коэффициент силуэта")
    ax.set_ylabel("Диссертации")
    ax.set_title(f"Силуэтный график (метрика: {metric})")
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

tab_lineages, tab_dissertations, tab_profiles, tab_schoolcomparison = st.tabs(
    [
        "Построение деревьев",
        "Поиск информации о диссертациях",
        "Поиск по тематическим профилям",
        "Сравнение научных школ"    ]
)

with tab_lineages:
    if st.button("📖 Инструкция", key="instruction_lineages"):
        show_instruction("lineages")
        
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
                        if st.button("📥 Таблица данных", key=f"data_{file_prefix}"):
                            download_data_dialog(subset, f"{file_prefix}.sampling", f"tree_{file_prefix}")
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

#with tab_silhouette: Здесь был код вкладки сравнения научных школ

with tab_dissertations:
    if st.button("📖 Инструкция", key="instruction_dissertations"):
        show_instruction("dissertations")
        
    st.subheader("Поиск информации о диссертациях")
    st.write(
        """
        На этой вкладке доступен поиск диссертаций по формальным критериям.
        """
    )
    
    # Получаем уникальные значения для выпадающих списков
    all_years = sorted([str(y) for y in df["year"].dropna().unique() if str(y).strip()], reverse=True)
    all_cities = sorted([str(c) for c in df["city"].dropna().unique() if str(c).strip()])
    
    # Собираем все специальности
    all_specialties = set()
    for col in ["specialties_1.code", "specialties_1.name", "specialties_2.code", "specialties_2.name"]:
        if col in df.columns:
            vals = df[col].dropna().unique()
            all_specialties.update([str(v).strip() for v in vals if str(v).strip()])
    all_specialties = sorted(all_specialties)
    
    # Доступные критерии поиска
    available_criteria = {
        "title": "Название диссертации",
        "candidate_name": "ФИО автора",
        "supervisors": "ФИО научного руководителя",
        "opponents": "ФИО оппонента",
        "institution_prepared": "Организация выполнения",
        "leading_organization": "Ведущая организация",
        "defense_location": "Место защиты",
        "city": "Город защиты",
        "year": "Год защиты",
        "specialties": "Специальность"
    }
    
    st.markdown("### 1. Выбор критериев поиска")
    selected_criteria = st.multiselect(
        "Выберите критерии поиска (максимум 5 одновременно)",
        options=list(available_criteria.keys()),
        format_func=lambda x: available_criteria[x],
        max_selections=5,
        key="dissertation_search_criteria"
    )
    
    if not selected_criteria:
        st.info("Выберите хотя бы один критерий для поиска.")
    else:
        st.markdown("### 2. Ввод данных")
        search_params = {}
        
        for criterion in selected_criteria:
            if criterion == "year":
                search_params[criterion] = st.selectbox(
                    available_criteria[criterion],
                    options=["Все"] + all_years,
                    key=f"diss_search_{criterion}"
                )
            elif criterion == "city":
                search_params[criterion] = st.selectbox(
                    available_criteria[criterion],
                    options=["Все"] + all_cities,
                    key=f"diss_search_{criterion}"
                )
            elif criterion == "specialties":
                search_params[criterion] = st.selectbox(
                    available_criteria[criterion],
                    options=["Все"] + all_specialties,
                    key=f"diss_search_{criterion}"
                )
            else:
                search_params[criterion] = st.text_input(
                    available_criteria[criterion],
                    placeholder=f"Введите {available_criteria[criterion].lower()}...",
                    key=f"diss_search_{criterion}"
                )
        
        st.markdown("### 3. Результат")
        
        if st.button("Найти", type="primary", key="dissertation_search_button"):
            result_df = df.copy()
            
            # Применяем фильтры
            for criterion, value in search_params.items():
                if not value or value == "Все":
                    continue
                
                if criterion in ["title", "candidate_name", "institution_prepared", 
                               "leading_organization", "defense_location"]:
                    result_df = result_df[result_df[criterion].astype(str).str.contains(value, case=False, na=False)]
                
                elif criterion == "supervisors":
                    mask = pd.Series([False] * len(result_df), index=result_df.index)
                    for col in ["supervisors_1.name", "supervisors_2.name"]:
                        if col in result_df.columns:
                            mask |= result_df[col].astype(str).str.contains(value, case=False, na=False)
                    result_df = result_df[mask]
                
                elif criterion == "opponents":
                    mask = pd.Series([False] * len(result_df), index=result_df.index)
                    for col in ["opponents_1.name", "opponents_2.name", "opponents_3.name"]:
                        if col in result_df.columns:
                            mask |= result_df[col].astype(str).str.contains(value, case=False, na=False)
                    result_df = result_df[mask]
                
                elif criterion in ["city", "year"]:
                    result_df = result_df[result_df[criterion].astype(str).str.contains(value, case=False, na=False)]
                
                elif criterion == "specialties":
                    mask = pd.Series([False] * len(result_df), index=result_df.index)
                    for col in ["specialties_1.code", "specialties_1.name", "specialties_2.code", "specialties_2.name"]:
                        if col in result_df.columns:
                            mask |= result_df[col].astype(str).str.contains(value, case=False, na=False)
                    result_df = result_df[mask]
            
            # Сохраняем результаты в session_state
            st.session_state["diss_search_result"] = result_df
        
        # Показываем результаты, если они есть в session_state
        if "diss_search_result" in st.session_state:
            result_df = st.session_state["diss_search_result"]
            
            if result_df.empty:
                st.warning("По заданным критериям ничего не найдено.")
            else:
                st.success(f"Найдено диссертаций: {len(result_df)}")
                
                # Колонки для отображения
                display_columns = []
                for col in ["Code", "candidate_name", "title", "year", "city", "institution_prepared", 
                           "defense_location", "supervisors_1.name", "supervisors_2.name",
                           "opponents_1.name", "opponents_2.name", "opponents_3.name",
                           "leading_organization", "specialties_1.code", "specialties_1.name",
                           "specialties_2.code", "specialties_2.name", "degree.degree_level", 
                           "degree.science_field"]:
                    if col in result_df.columns:
                        display_columns.append(col)
                
                rename_map = {
                    "Code": "Код", "candidate_name": "Автор", "title": "Название",
                    "year": "Год", "city": "Город", "institution_prepared": "Организация",
                    "defense_location": "Место защиты",
                    "supervisors_1.name": "Научный руководитель 1",
                    "supervisors_2.name": "Научный руководитель 2",
                    "opponents_1.name": "Оппонент 1",
                    "opponents_2.name": "Оппонент 2",
                    "opponents_3.name": "Оппонент 3",
                    "leading_organization": "Ведущая организация",
                    "specialties_1.code": "Специальность 1 (код)",
                    "specialties_1.name": "Специальность 1",
                    "specialties_2.code": "Специальность 2 (код)",
                    "specialties_2.name": "Специальность 2",
                    "degree.degree_level": "Степень",
                    "degree.science_field": "Область науки"
                }
                
                display_df = result_df[display_columns].rename(columns=rename_map)
                st.dataframe(display_df, use_container_width=True)
                
                # Кнопка скачивания — данные теперь доступны из session_state
                if st.button("📥 Скачать результаты", key="diss_show_download"):
                    download_data_dialog(result_df[display_columns], "dissertations_search", "diss")


with tab_profiles:
    if st.button("📖 Инструкция", key="instruction_profiles"):
        show_instruction("profiles")
        
    st.subheader("Поиск диссертаций по тематическим профилям")
    st.write(
        "Выберите до пяти элементов классификатора тематических профилей. В "
        "подборку попадут диссертации, у которых баллы по каждому из выбранных "
        f"пунктов не ниже {PROFILE_MIN_SCORE}, а ранжирование будет происходить по "
        "сумме баллов."
    )

    if PROFILE_SELECTION_SESSION_KEY not in st.session_state:
        st.session_state[PROFILE_SELECTION_SESSION_KEY] = []

    selected_codes: List[str] = list(
        st.session_state.get(PROFILE_SELECTION_SESSION_KEY, [])
    )

    selection_container = st.container()
    with selection_container:
        st.markdown("#### Добавление позиций классификатора")
        options: List[Optional[ClassifierItem]] = [None, *THEMATIC_CLASSIFIER]
        choice = st.selectbox(
            "Элемент классификатора",
            options=options,
            format_func=classifier_format,
            key="profile_classifier_choice",
        )

        add_reason: Optional[str] = None
        add_code: Optional[str] = None
        if choice is not None:
            add_code = choice[0]
            if choice[2]:
                add_reason = "Этот пункт является обобщённым и не участвует в подборке."
            elif add_code in selected_codes:
                add_reason = "Пункт уже добавлен в список."
            elif len(selected_codes) >= PROFILE_SELECTION_LIMIT:
                add_reason = (
                    f"Можно выбрать не более {PROFILE_SELECTION_LIMIT} пунктов классификатора."
                )

        add_disabled = add_code is None or add_reason is not None
        add_clicked = st.button(
            "Добавить в подборку",
            disabled=add_disabled,
            key="profile_add_button",
        )

        if add_clicked and add_code is not None:
            updated = [*selected_codes, add_code]
            st.session_state[PROFILE_SELECTION_SESSION_KEY] = updated
            selected_codes = updated

        if add_reason and choice is not None:
            st.caption(add_reason)

    st.markdown("#### Выбранные позиции")
    if selected_codes:
        st.caption(
            f"Учтём диссертации, у которых баллы ≥ {PROFILE_MIN_SCORE} по каждому выбранному пункту."
        )
        for code in list(selected_codes):
            cols = st.columns([0.85, 0.15])
            with cols[0]:
                st.markdown(f"• **{classifier_label(code)}**")
            with cols[1]:
                if st.button(
                    "Удалить",
                    key=f"profile_remove_{code}",
                    use_container_width=True,
                ):
                    updated = [c for c in selected_codes if c != code]
                    st.session_state[PROFILE_SELECTION_SESSION_KEY] = updated
                    selected_codes = updated
                    _trigger_rerun()

        col_clear, col_dummy = st.columns([0.2, 0.8])
        with col_clear:
            if st.button("Очистить выбор", key="profile_clear_selection"):
                st.session_state[PROFILE_SELECTION_SESSION_KEY] = []
                selected_codes = []
                _trigger_rerun()
    else:
        st.info(
            "Добавьте от одного до пяти пунктов классификатора, чтобы сформировать подборку."
        )

    # Кнопка запуска поиска
    run_search_click = st.button(
        "Найти диссертации",
        type="primary",
        disabled=not selected_codes,
        key="profile_run_search",
    )

    # Логика сохранения состояния: если нажали кнопку, запоминаем это
    if run_search_click:
        st.session_state["profile_search_active"] = True

    # Если пользователь очистил коды, сбрасываем состояние поиска
    if not selected_codes:
        st.session_state["profile_search_active"] = False

    # Условие отображения: проверяем не кнопку, а сохраненное состояние
    if st.session_state.get("profile_search_active") and selected_codes:
        try:
            scores_df = load_basic_scores()
        except Exception as exc:
            st.error(f"Не удалось загрузить тематические профили: {exc}")
            scores_df = None

        if scores_df is not None:
            missing_columns = [
                code for code in selected_codes if code not in scores_df.columns
            ]
            if missing_columns:
                st.error(
                    "В файлах с тематическими профилями отсутствуют столбцы: "
                    + ", ".join(f"`{c}`" for c in missing_columns)
                )
            else:
                working = scores_df[["Code", *selected_codes]].copy()
                for code in selected_codes:
                    working = working[working[code] >= PROFILE_MIN_SCORE]

                if working.empty:
                    st.info(
                        "Не найдено диссертаций, соответствующих заданным ограничениям."
                    )
                else:
                    working["profile_total"] = working[selected_codes].sum(axis=1)
                    sort_columns = ["profile_total", *selected_codes]
                    working = working.sort_values(
                        by=sort_columns,
                        ascending=[False] * len(sort_columns),
                    )

                    info_columns = [
                        "Code",
                        "candidate_name",
                        "title",
                        "year",
                        "degree.degree_level",
                        "degree.science_field",
                        "institution_prepared",
                        "supervisors_1.name",
                        "supervisors_2.name",
                        "specialties_1.name",
                        "specialties_2.name",
                    ]
                    available_info_columns = [
                        col for col in info_columns if col in df.columns
                    ]
                    if available_info_columns:
                        info_df = (
                            df[available_info_columns]
                            .copy()
                            .drop_duplicates(subset="Code", keep="first")
                        )
                    else:
                        info_df = pd.DataFrame(columns=["Code"])

                    results = working.merge(info_df, on="Code", how="left")
                    results["profile_total"] = results["profile_total"].round(2)

                    score_labels = {code: classifier_label(code) for code in selected_codes}
                    for code, label in score_labels.items():
                        results[label] = results[code].round(2)

                    supervisor_cols = [
                        col
                        for col in ["supervisors_1.name", "supervisors_2.name"]
                        if col in results.columns
                    ]

                    if supervisor_cols:
                        def _join_names(row: pd.Series) -> str:
                            names = []
                            for value in row.tolist():
                                if isinstance(value, str):
                                    clean = value.strip()
                                    if clean and clean not in names:
                                        names.append(clean)
                            return ", ".join(names)

                        results["Научные руководители"] = (
                            results[supervisor_cols]
                            .replace({"": pd.NA})
                            .apply(_join_names, axis=1)
                        )

                    rename_map = {
                        "Code": "Код диссертации",
                        "candidate_name": "Автор",
                        "title": "Название",
                        "year": "Год защиты",
                        "degree.degree_level": "Степень",
                        "degree.science_field": "Отрасль науки",
                        "institution_prepared": "Организация выполнения",
                        "specialties_1.name": "Специальность",
                        "specialties_2.name": "Дополнительная специальность",
                        "profile_total": "Суммарный балл",
                    }

                    column_order = [
                        "Code",
                        "candidate_name",
                        "title",
                        "year",
                        "degree.degree_level",
                        "degree.science_field",
                        "institution_prepared",
                        "Научные руководители",
                        "specialties_1.name",
                        "specialties_2.name",
                        "profile_total",
                        *score_labels.values(),
                    ]
                    display_columns = [
                        col for col in column_order if col in results.columns
                    ]
                    display_df = results[display_columns].rename(columns=rename_map)

                    # --- БЛОК ФИЛЬТРАЦИИ ---
                    st.markdown("---")
                    
                    f_col1, f_col2 = st.columns([0.6, 0.4])
                    with f_col1:
                        search_query = st.text_input(
                            "🔍 Фильтр по таблице",
                            placeholder="Введите автора, год или часть названия...",
                            help="Строки, не содержащие введенный текст, будут скрыты.",
                            key="profile_result_filter"
                        )
                    
                    if search_query:
                        mask = display_df.astype(str).apply(
                            lambda x: x.str.contains(search_query, case=False, na=False)
                        ).any(axis=1)
                        filtered_df = display_df[mask]
                    else:
                        filtered_df = display_df

                    if len(filtered_df) != len(display_df):
                        st.success(f"Найдено всего: {len(display_df)}. После фильтрации: {len(filtered_df)}")
                    else:
                        st.success(f"Найдено диссертаций: {len(display_df)}")
                    
                    st.dataframe(filtered_df, use_container_width=True)

                    selection_slug = slug("_".join(selected_codes)) or "profiles"
                    if st.button("📥 Скачать результаты", key="profile_show_download"):
                        download_data_dialog(filtered_df, f"profiles_{selection_slug}", "profile")
                        
with tab_schoolcomparison:
    # Словарь {код: название} из THEMATIC_CLASSIFIER
    classifier_labels = {code: title for code, title, _ in THEMATIC_CLASSIFIER}
    
    render_school_comparison_tab(
        df=df,
        idx=idx,
        lineage_func=lineage,
        rows_for_func=rows_for,
        scores_folder="basic_scores",        # ← Папка с профилями
        specific_files=None,                  # ← None = все CSV, или ["file1.csv", "file2.csv"]
        classifier_labels=classifier_labels,  # ← Опционально: названия узлов
    )
