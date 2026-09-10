from pathlib import Path

import kagglehub
import pandas as pd
import plotly.express as px
import streamlit as st


# ==========================================
# CONFIGURAÇÕES
# ==========================================

DATASET_SLUG = "manaswinsripatnala/social-media-dopamine-and-productivity-dataset"

USAGE = "avg_daily_sm_hours"
PRODUCTIVITY = "productivity_self_rating"
NIGHT = "late_night_scrolling"
SLEEP = "sleep_quality"
FOCUS = "difficulty_maintaining_focus"
DOOM = "doomscrolling_frequency"
FOMO = "fomo_score"
AGE = "age"

REQUIRED = [USAGE, PRODUCTIVITY, NIGHT, SLEEP, FOCUS, DOOM, FOMO]

SLEEP_MAP = {"Very poor": 1, "Poor": 2, "Fair": 3, "Good": 4, "Very good": 5}
NIGHT_MAP = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3, "Daily": 4}
FOCUS_MAP = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Yes": 3}

DOOM_MAP = {
    "Never": "Nunca",
    "Rarely": "Raramente",
    "Sometimes": "Às vezes",
    "Often": "Frequentemente",
    "Daily": "Diariamente",
}

USAGE_ORDER = ["Baixo uso", "Uso moderado", "Alto uso", "Uso muito alto"]
NIGHT_ORDER = ["Nunca", "Raramente", "Às vezes", "Frequentemente", "Diariamente"]
FOMO_ORDER = ["Baixo (1–3)", "Médio (4–7)", "Alto (8–10)"]
AGE_ORDER = ["Até 24", "25–34", "35–44", "45–54", "55+"]

# Identidade visual: uma cor principal e tons auxiliares apenas quando necessário.
PRIMARY = "#0F6B6D"
PRIMARY_DARK = "#0B4F52"
PRIMARY_LIGHT = "#69A7A8"
PRIMARY_PALE = "#DDEEEE"

ACCENT = "#E76F00"
ACCENT_LIGHT = "#F4B477"

NEUTRAL = "#667085"
NEUTRAL_LIGHT = "#D9DDE3"
BACKGROUND = "#F8F9FA"
TEXT = "#202124"
GRID = "#E2E2E2"

PLOT_CONFIG = {
    "displaylogo": False,
    "responsive": True,
}

DISPLAY_LABELS = {
    USAGE: "Uso diário de redes sociais (h)",
    PRODUCTIVITY: "Produtividade autorrelatada (%)",
    NIGHT: "Uso noturno (escala 0–4)",
    SLEEP: "Qualidade do sono",
    FOCUS: "Dificuldade de foco",
    DOOM: "Frequência de doomscrolling",
    FOMO: "Pontuação de FOMO",
    AGE: "Idade",
}


# ==========================================
# CARREGAMENTO
# ==========================================

def find_files(root):
    if not root.exists():
        return []

    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".csv", ".xlsx", ".xls"}
        and not any(x in {".venv", ".git", "__pycache__"} for x in p.parts)
    )


def read_file(path):
    return pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)


@st.cache_data
def load_dataset():
    folders = [Path.cwd(), Path.cwd() / "data", Path.cwd() / "datasets"]

    try:
        folders.append(Path(kagglehub.dataset_download(DATASET_SLUG)))
    except Exception:
        pass

    for folder in folders:
        for path in find_files(folder):
            try:
                df = read_file(path)
                if not df.empty and all(col in df.columns for col in REQUIRED):
                    return df
            except Exception:
                continue

    return None


# ==========================================
# TRATAMENTO
# ==========================================

def prepare_dataframe(df):
    if df is None or df.empty:
        return None

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        friendly_missing = [DISPLAY_LABELS.get(c, c) for c in missing]
        st.error("Colunas ausentes: " + ", ".join(friendly_missing))
        return None

    has_age = AGE in df.columns
    cols = REQUIRED + ([AGE] if has_age else [])
    df = df[cols].copy()

    for col in [USAGE, PRODUCTIVITY, FOMO]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[SLEEP] = df[SLEEP].map(SLEEP_MAP)
    df[NIGHT] = df[NIGHT].map(NIGHT_MAP)
    df[FOCUS] = df[FOCUS].map(FOCUS_MAP)

    df = df.dropna(subset=REQUIRED)

    df["Uso noturno"] = pd.cut(
        df[NIGHT],
        [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5],
        labels=NIGHT_ORDER,
    )

    df["Intensidade de uso"] = pd.cut(
        df[USAGE],
        [float("-inf"), 2, 4, 6, float("inf")],
        labels=USAGE_ORDER,
    )

    df["Nível de FOMO"] = pd.cut(
        df[FOMO],
        [0, 3, 7, 10],
        labels=FOMO_ORDER,
        include_lowest=True,
    )

    if has_age:
        df[AGE] = pd.to_numeric(df[AGE], errors="coerce")
        df["Faixa etária"] = pd.cut(
            df[AGE],
            [0, 24, 34, 44, 54, 150],
            labels=AGE_ORDER,
        )

    return df


# ==========================================
# ESTILO
# ==========================================

def style_chart(fig, y_range=None, showlegend=False, height=340):
    fig.update_layout(
        showlegend=showlegend,
        plot_bgcolor=BACKGROUND,
        paper_bgcolor=BACKGROUND,
        font=dict(color=TEXT, size=12),
        height=height,
        margin=dict(l=8, r=8, t=20, b=8),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_color=TEXT,
        ),
        xaxis=dict(showgrid=False, title_font=dict(color=TEXT)),
        yaxis=dict(
            showgrid=True,
            gridcolor=GRID,
            range=y_range,
            title_font=dict(color=TEXT),
        ),
        legend=dict(
            title_text="",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )
    return fig


def section_title(text):
    st.markdown(
        f"""
        <h3 style="
            color:{PRIMARY};
            font-size:1.25rem;
            font-weight:700;
            margin:0 0 0.75rem 0;
            line-height:1.3;
        ">
            {text}
        </h3>
        """,
        unsafe_allow_html=True,
    )


def chart_caption(text):
    st.caption(text)


def page_description(text):
    st.markdown(
        f"""
        <p style="
            color:{NEUTRAL};
            font-size:0.98rem;
            line-height:1.55;
            margin:0.25rem 0 1.25rem 0;
            max-width:950px;
        ">
            {text}
        </p>
        """,
        unsafe_allow_html=True,
    )


def plot(fig):
    st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)


# ==========================================
# GRÁFICOS
# ==========================================

def build_charts(df):
    corr = df[[USAGE, PRODUCTIVITY]].corr().iloc[0, 1]

    scatter = px.scatter(
        df,
        x=USAGE,
        y=PRODUCTIVITY,
        trendline="ols",
        opacity=0.55,
        color_discrete_sequence=[PRIMARY],
        labels=DISPLAY_LABELS,
    )

    scatter.data[0].update(
        marker=dict(size=8, color=PRIMARY),
        hovertemplate=(
            "Uso diário: %{x:.1f} h"
            "<br>Produtividade: %{y:.1f}%"
            "<extra></extra>"
        ),
    )

    if len(scatter.data) > 1:
        scatter.data[1].update(
            line=dict(color=PRIMARY_DARK, width=3),
            hovertemplate=(
                "Linha de tendência"
                "<br>Uso diário: %{x:.1f} h"
                "<br>Produtividade estimada: %{y:.1f}%"
                "<extra></extra>"
            ),
        )

    style_chart(scatter)

    night_colors = {
        "Nunca": "#DDEEEE",
        "Raramente": "#A9CECF",
        "Às vezes": "#69A7A8",
        "Frequentemente": PRIMARY,
        "Diariamente": PRIMARY_DARK,
    }

    box = px.box(
        df,
        x="Uso noturno",
        y=SLEEP,
        color="Uso noturno",
        category_orders={"Uso noturno": NIGHT_ORDER},
        color_discrete_map=night_colors,
        labels={
            "Uso noturno": "Frequência de uso noturno",
            SLEEP: "Qualidade do sono (1–5)",
        },
        points="outliers",
    )

    box.update_traces(
        line=dict(width=2),
        hovertemplate=(
            "Uso noturno: %{x}"
            "<br>Qualidade do sono: %{y:.1f}"
            "<extra></extra>"
        ),
    )
    style_chart(box, [0.5, 5.5])
    box.update_layout(showlegend=False)

    focus_df = (
        df.groupby("Intensidade de uso", observed=True)[FOCUS]
        .mean()
        .reindex(USAGE_ORDER)
        .reset_index(name="Dificuldade média de foco")
    )

    focus_bar = px.bar(
        focus_df,
        x="Intensidade de uso",
        y="Dificuldade média de foco",
        text_auto=".2f",
        category_orders={"Intensidade de uso": USAGE_ORDER},
        labels={
            "Intensidade de uso": "Intensidade de uso",
            "Dificuldade média de foco": "Dificuldade média de foco (0–3)",
        },
    )
    focus_bar.update_traces(
        marker_color=PRIMARY,
        width=0.55,
        hovertemplate=(
            "Intensidade: %{x}"
            "<br>Dificuldade média de foco: %{y:.2f}"
            "<extra></extra>"
        ),
    )
    style_chart(focus_bar, [0, 3])

    heatmap_df = df[[DOOM, "Nível de FOMO"]].copy()
    heatmap_df["Frequência de doomscrolling"] = heatmap_df[DOOM].replace(DOOM_MAP)

    heatmap_data = pd.crosstab(
        heatmap_df["Frequência de doomscrolling"],
        heatmap_df["Nível de FOMO"],
    ).reindex(
        index=NIGHT_ORDER,
        columns=FOMO_ORDER,
        fill_value=0,
    )

    heatmap = px.imshow(
        heatmap_data,
        text_auto=True,
        aspect="auto",
        color_continuous_scale=[
            [0.00, "#F2F8F8"],
            [0.25, "#DDEEEE"],
            [0.50, "#A9CECF"],
            [0.75, "#69A7A8"],
            [1.00, PRIMARY_DARK],
        ],
        labels={
            "x": "Nível de FOMO",
            "y": "Frequência de doomscrolling",
            "color": "Participantes",
        },
    )
    heatmap.update_traces(
        hovertemplate=(
            "FOMO: %{x}"
            "<br>Doomscrolling: %{y}"
            "<br>Participantes: %{z}"
            "<extra></extra>"
        )
    )
    heatmap.update_coloraxes(
        colorbar=dict(title="Participantes"),
    )
    style_chart(heatmap)

    return scatter, box, focus_bar, heatmap, corr


def build_age_charts(df):
    if AGE not in df.columns or "Faixa etária" not in df.columns:
        return None, None

    age_df = df.dropna(subset=["Faixa etária"])
    if age_df.empty:
        return None, None

    usage_by_age = (
        age_df.groupby("Faixa etária", observed=True)[USAGE]
        .mean()
        .reindex(AGE_ORDER)
        .reset_index(name="Uso médio diário (h)")
    )

    fomo_by_age = (
        age_df.groupby("Faixa etária", observed=True)[FOMO]
        .mean()
        .reindex(AGE_ORDER)
        .reset_index(name="FOMO médio")
    )

    age_bar = px.bar(
        usage_by_age,
        x="Faixa etária",
        y="Uso médio diário (h)",
        text_auto=".1f",
        category_orders={"Faixa etária": AGE_ORDER},
    )
    age_bar.update_traces(
        marker_color=PRIMARY,
        width=0.55,
        hovertemplate=(
            "Faixa etária: %{x}"
            "<br>Uso médio diário: %{y:.1f} h"
            "<extra></extra>"
        ),
    )
    style_chart(age_bar)

    fomo_bar = px.bar(
        fomo_by_age,
        x="Faixa etária",
        y="FOMO médio",
        text_auto=".1f",
        category_orders={"Faixa etária": AGE_ORDER},
    )
    fomo_bar.update_traces(
        marker_color=PRIMARY,
        width=0.55,
        hovertemplate=(
            "Faixa etária: %{x}"
            "<br>FOMO médio: %{y:.1f}"
            "<extra></extra>"
        ),
    )
    style_chart(fomo_bar, [0, 10])

    return age_bar, fomo_bar


def build_dependency_charts(df):
    dep_df = df.dropna(subset=["Nível de FOMO"]).copy()
    if dep_df.empty:
        return None, None

    dep_df["Doom_num"] = dep_df[DOOM].map(NIGHT_MAP)

    profile = (
        dep_df.groupby("Nível de FOMO", observed=True)[[NIGHT, "Doom_num", FOCUS]]
        .mean()
        .reindex(FOMO_ORDER)
    )

    profile["Uso noturno"] = profile[NIGHT] / 4
    profile["Doomscrolling"] = profile["Doom_num"] / 4
    profile["Dificuldade de foco"] = profile[FOCUS] / 3

    profile_long = (
        profile[["Uso noturno", "Doomscrolling", "Dificuldade de foco"]]
        .reset_index()
        .melt(
            id_vars="Nível de FOMO",
            var_name="Indicador",
            value_name="Intensidade média",
        )
    )

    profile_bar = px.bar(
        profile_long,
        x="Nível de FOMO",
        y="Intensidade média",
        color="Indicador",
        barmode="group",
        category_orders={
            "Nível de FOMO": FOMO_ORDER,
            "Indicador": ["Uso noturno", "Doomscrolling", "Dificuldade de foco"],
        },
        color_discrete_map={
            "Uso noturno": PRIMARY_DARK,
            "Doomscrolling": PRIMARY,
            "Dificuldade de foco": PRIMARY_LIGHT,
        },
        labels={
            "Nível de FOMO": "Nível de FOMO",
            "Intensidade média": "Intensidade média normalizada",
        },
    )
    profile_bar.update_traces(
        hovertemplate=(
            "Nível de FOMO: %{x}"
            "<br>%{fullData.name}: %{y:.2f}"
            "<extra></extra>"
        )
    )
    style_chart(profile_bar, [0, 1], showlegend=True)

    sleep_by_fomo = (
        dep_df.groupby("Nível de FOMO", observed=True)[SLEEP]
        .mean()
        .reindex(FOMO_ORDER)
        .reset_index(name="Qualidade média do sono")
    )

    sleep_bar = px.bar(
        sleep_by_fomo,
        x="Nível de FOMO",
        y="Qualidade média do sono",
        text_auto=".2f",
        category_orders={"Nível de FOMO": FOMO_ORDER},
    )
    sleep_bar.update_traces(
        marker_color=PRIMARY,
        width=0.55,
        hovertemplate=(
            "Nível de FOMO: %{x}"
            "<br>Qualidade média do sono: %{y:.2f}"
            "<extra></extra>"
        ),
    )
    style_chart(sleep_bar, [0, 5])

    return profile_bar, sleep_bar


def build_additional_charts(df):
    """Duas análises adicionais que complementam os temas já existentes."""

    # 1) Perfil da amostra por intensidade de uso.
    usage_profile = (
        df["Intensidade de uso"]
        .value_counts(sort=False)
        .reindex(USAGE_ORDER, fill_value=0)
        .rename_axis("Intensidade de uso")
        .reset_index(name="Participantes")
    )

    total = usage_profile["Participantes"].sum()
    usage_profile["Percentual"] = (
        usage_profile["Participantes"] / total * 100 if total else 0
    )
    usage_profile["Rótulo"] = usage_profile["Percentual"].map(lambda v: f"{v:.1f}%")

    usage_colors = {
        "Baixo uso": "#DDEEEE",
        "Uso moderado": "#A9CECF",
        "Alto uso": PRIMARY,
        "Uso muito alto": PRIMARY_DARK,
    }

    usage_profile_bar = px.bar(
        usage_profile,
        x="Intensidade de uso",
        y="Participantes",
        color="Intensidade de uso",
        text="Rótulo",
        category_orders={"Intensidade de uso": USAGE_ORDER},
        color_discrete_map=usage_colors,
        labels={
            "Intensidade de uso": "Intensidade de uso",
            "Participantes": "Participantes",
        },
    )
    usage_profile_bar.update_traces(
        width=0.58,
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            "Intensidade: %{x}"
            "<br>Participantes: %{y}"
            "<extra></extra>"
        ),
    )
    style_chart(usage_profile_bar, height=330)
    usage_profile_bar.update_layout(showlegend=False)

    # 2) Relação ainda não exibida: produtividade média por nível de FOMO.
    productivity_fomo = (
        df.dropna(subset=["Nível de FOMO"])
        .groupby("Nível de FOMO", observed=True)[PRODUCTIVITY]
        .mean()
        .reindex(FOMO_ORDER)
        .reset_index(name="Produtividade média")
    )

    productivity_fomo_bar = px.bar(
        productivity_fomo,
        x="Nível de FOMO",
        y="Produtividade média",
        text_auto=".1f",
        category_orders={"Nível de FOMO": FOMO_ORDER},
        labels={
            "Nível de FOMO": "Nível de FOMO",
            "Produtividade média": "Produtividade média (%)",
        },
    )
    productivity_fomo_bar.update_traces(
        marker_color=PRIMARY,
        width=0.55,
        hovertemplate=(
            "Nível de FOMO: %{x}"
            "<br>Produtividade média: %{y:.1f}%"
            "<extra></extra>"
        ),
    )
    style_chart(productivity_fomo_bar, height=330)

    return usage_profile_bar, productivity_fomo_bar


# ==========================================
# INTERFACE
# ==========================================

def main():
    st.set_page_config(
        page_title="Dashboard Social Media & Produtividade",
        page_icon="📊",
        layout="wide",
    )

    st.markdown(
        f"""
        <style>
            .block-container {{
                padding-top: 2rem;
                padding-bottom: 3rem;
            }}

            h1 {{
                color: {TEXT};
            }}

            [data-testid="stMetricValue"] {{
                color: {PRIMARY};
            }}

            div[data-testid="stMetric"] {{
                background-color: white;
                border: 1px solid #EAEAEA;
                padding: 0.9rem 1rem;
                border-radius: 0.75rem;
            }}

            /* Controles de filtro: mesma identidade azul-petróleo */
            div[data-testid="stSlider"] [role="slider"] {{
                background-color: {PRIMARY} !important;
                border-color: {PRIMARY} !important;
            }}

            div[data-testid="stSlider"] [data-baseweb="slider"] > div > div {{
                background-color: {PRIMARY} !important;
            }}

            div[data-testid="stNumberInput"] input {{
                color: {TEXT};
            }}

            div[data-testid="stNumberInput"] input:focus {{
                border-color: {PRIMARY} !important;
                box-shadow: 0 0 0 1px {PRIMARY} !important;
            }}

            /* Abas do painel */
            button[data-baseweb="tab"] {{
                font-weight: 600;
            }}

            button[data-baseweb="tab"][aria-selected="true"] {{
                color: {PRIMARY} !important;
            }}

            div[data-baseweb="tab-highlight"] {{
                background-color: {PRIMARY} !important;
            }}

            /* Slider - reforço da cor principal */
            div[data-baseweb="slider"] [role="slider"] {{
                background-color: {PRIMARY} !important;
                border-color: {PRIMARY} !important;
            }}

            div[data-baseweb="slider"] > div > div {{
                background-color: {PRIMARY} !important;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    df = prepare_dataframe(load_dataset())

    if df is None:
        st.error("Dataset não encontrado ou inválido.")
        return

    st.sidebar.header("Filtro de análise")

    min_use = float(df[USAGE].min())
    max_use = float(df[USAGE].max())

    # Mantém slider e campos numéricos sincronizados.
    if "usage_range" not in st.session_state:
        st.session_state.usage_range = (min_use, max_use)
    if "usage_min_input" not in st.session_state:
        st.session_state.usage_min_input = min_use
    if "usage_max_input" not in st.session_state:
        st.session_state.usage_max_input = max_use

    def sync_from_slider():
        lower, upper = st.session_state.usage_range
        st.session_state.usage_min_input = float(lower)
        st.session_state.usage_max_input = float(upper)

    def sync_from_min_input():
        lower = max(min_use, min(float(st.session_state.usage_min_input), max_use))
        upper = max(min_use, min(float(st.session_state.usage_max_input), max_use))

        if lower > upper:
            upper = lower
            st.session_state.usage_max_input = upper

        st.session_state.usage_min_input = lower
        st.session_state.usage_range = (lower, upper)

    def sync_from_max_input():
        lower = max(min_use, min(float(st.session_state.usage_min_input), max_use))
        upper = max(min_use, min(float(st.session_state.usage_max_input), max_use))

        if upper < lower:
            lower = upper
            st.session_state.usage_min_input = lower

        st.session_state.usage_max_input = upper
        st.session_state.usage_range = (lower, upper)

    st.sidebar.slider(
        "Horas diárias de uso das redes sociais",
        min_value=min_use,
        max_value=max_use,
        value=st.session_state.usage_range,
        step=0.1,
        key="usage_range",
        on_change=sync_from_slider,
        help="Arraste os marcadores ou digite abaixo os valores exatos do intervalo.",
    )

    input_col1, input_col2 = st.sidebar.columns(2)

    with input_col1:
        st.number_input(
            "Mínimo",
            min_value=min_use,
            max_value=max_use,
            step=0.1,
            format="%.2f",
            key="usage_min_input",
            on_change=sync_from_min_input,
        )

    with input_col2:
        st.number_input(
            "Máximo",
            min_value=min_use,
            max_value=max_use,
            step=0.1,
            format="%.2f",
            key="usage_max_input",
            on_change=sync_from_max_input,
        )

    filtro = st.session_state.usage_range
    filtered = df[df[USAGE].between(*filtro)]

    # Navegação para consulta dos dados.
    if "show_raw_data" not in st.session_state:
        st.session_state.show_raw_data = False

    st.sidebar.divider()
    st.sidebar.subheader("Consulta")

    if st.sidebar.button(
        "Consultar dados brutos",
        use_container_width=True,
        help="Abre a tabela com os dados considerados pelo filtro atual.",
    ):
        st.session_state.show_raw_data = True

    st.title("Painel de Mídias Sociais e Produtividade")
    st.caption(
        "Análise das relações entre uso de redes sociais, foco, sono, FOMO e produtividade."
    )

    with st.container(border=True):
        section_title("Visão geral da amostra")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Participantes", f"{len(filtered):,}".replace(",", "."))
        c2.metric("Tempo médio diário", f"{filtered[USAGE].mean():.1f} h")
        c3.metric(
            "Produtividade média",
            f"{filtered[PRODUCTIVITY].mean():.1f}%",
        )
        c4.metric(
            "FOMO médio",
            f"{filtered[FOMO].mean():.1f}",
        )

    # Tela dedicada de consulta, acessada pelo menu lateral.
    if st.session_state.show_raw_data:
        with st.container(border=True):
            section_title("Consulta dos dados")
            page_description(
                "Visualize os registros considerados no intervalo selecionado nos filtros. "
                "Os nomes das variáveis foram adaptados para facilitar a leitura."
            )

            display_df = filtered.rename(columns=DISPLAY_LABELS)

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                height=520,
            )

            st.caption(
                f"Exibindo {len(display_df):,} registros do filtro atual."
                .replace(",", ".")
            )

            if st.button("Voltar às análises"):
                st.session_state.show_raw_data = False
                st.rerun()

        return

    if len(filtered) <= 5:
        st.warning("A amostra filtrada possui poucos dados para uma análise confiável.")
        return

    scatter, box, focus_bar, heatmap, corr = build_charts(filtered)
    age_bar, fomo_bar = build_age_charts(filtered)
    profile_bar, sleep_bar = build_dependency_charts(filtered)
    usage_profile_bar, productivity_fomo_bar = build_additional_charts(filtered)

    tab_perfil, tab_foco, tab_prod, tab_sono, tab_bem_estar = st.tabs(
        [
            "Perfil e uso",
            "Foco e concentração",
            "Produtividade",
            "Sono e hábitos digitais",
            "Bem-estar e dependência digital",
        ]
    )

    # ==================================================
    # PERFIL DOS PARTICIPANTES E USO DAS REDES
    # ==================================================
    with tab_perfil:
        page_description(
            "Apresenta o perfil da amostra e como o uso das redes sociais se distribui "
            "entre os participantes, considerando intensidade de uso e diferenças entre "
            "faixas etárias."
        )

        with st.container(border=True):
            section_title("Distribuição dos participantes por intensidade de uso")
            chart_caption(
                "Mostra como a amostra se distribui entre diferentes níveis "
                "de tempo diário nas redes sociais."
            )
            plot(usage_profile_bar)

        if age_bar is not None and fomo_bar is not None:
            col_perfil_1, col_perfil_2 = st.columns(2, gap="large")

            with col_perfil_1:
                with st.container(border=True):
                    section_title("Uso de redes sociais por faixa etária")
                    chart_caption(
                        "Tempo médio diário de uso das redes sociais entre as diferentes faixas etárias."
                    )
                    plot(age_bar)

            with col_perfil_2:
                with st.container(border=True):
                    section_title("FOMO por faixa etária")
                    chart_caption(
                        "Nível médio de FOMO entre as diferentes faixas etárias."
                    )
                    plot(fomo_bar)
        else:
            st.info(
                "A coluna de idade não está disponível no conjunto de dados carregado; "
                "as análises por faixa etária não podem ser exibidas."
            )

    # ==================================================
    # REDES SOCIAIS, FOCO E CONCENTRAÇÃO
    # ==================================================
    with tab_foco:
        page_description(
            "Investiga a relação entre a intensidade de uso das redes sociais e a "
            "dificuldade de manter o foco e a concentração nas atividades do dia a dia."
        )

        with st.container(border=True):
            section_title("Intensidade de uso e dificuldade de foco")
            chart_caption(
                "Comparação da dificuldade média de foco entre diferentes intensidades de uso."
            )
            plot(focus_bar)
            st.info(
                "Escala utilizada: Nunca = 0, Raramente = 1, Às vezes = 2 e Sim = 3."
            )

    # ==================================================
    # REDES SOCIAIS E PRODUTIVIDADE
    # ==================================================
    with tab_prod:
        page_description(
            "Analisa como o tempo diário nas redes sociais e os níveis de FOMO se "
            "relacionam com a produtividade autorrelatada pelos participantes."
        )

        col_prod_1, col_prod_2 = st.columns(2, gap="large")

        with col_prod_1:
            with st.container(border=True):
                section_title("Uso diário de redes sociais e produtividade")
                chart_caption(
                    "Relação entre o tempo médio diário nas redes sociais e a produtividade autorrelatada."
                )
                plot(scatter)

                if corr <= -0.20:
                    st.info(
                        "A tendência geral indica que, conforme o tempo diário de uso das "
                        "redes sociais aumenta, a produtividade relatada tende a diminuir."
                    )
                elif corr >= 0.20:
                    st.info(
                        "A tendência geral indica que, conforme o tempo diário de uso das "
                        "redes sociais aumenta, a produtividade relatada tende a aumentar."
                    )
                else:
                    st.info(
                        "Neste recorte, não aparece uma tendência clara entre o tempo diário "
                        "de uso das redes sociais e a produtividade relatada."
                    )

        with col_prod_2:
            with st.container(border=True):
                section_title("Produtividade por nível de FOMO")
                chart_caption(
                    "Compara a produtividade média entre participantes com "
                    "níveis baixo, médio e alto de FOMO."
                )
                plot(productivity_fomo_bar)
                st.info(
                    "A comparação ajuda a observar se níveis diferentes de FOMO aparecem "
                    "associados a diferenças na produtividade autorrelatada."
                )

    # ==================================================
    # REDES SOCIAIS, SONO E HÁBITOS DIGITAIS
    # ==================================================
    with tab_sono:
        page_description(
            "Explora hábitos digitais ligados ao período noturno, relacionando frequência "
            "de uso, doomscrolling, qualidade do sono e níveis de FOMO."
        )

        col_sono_1, col_sono_2 = st.columns(2, gap="large")

        with col_sono_1:
            with st.container(border=True):
                section_title("Uso noturno e qualidade do sono")
                chart_caption(
                    "Distribuição da qualidade do sono conforme a frequência de uso noturno."
                )
                plot(box)
                st.info(
                    "Os tons ficam mais escuros conforme aumenta a frequência de uso noturno."
                )

        with col_sono_2:
            with st.container(border=True):
                section_title("Doomscrolling e níveis de FOMO")
                chart_caption(
                    "Concentração de participantes por frequência de doomscrolling e nível de FOMO."
                )
                plot(heatmap)
                st.info(
                    "Quanto mais intensa a cor, maior a concentração de participantes naquela combinação."
                )

    # ==================================================
    # BEM-ESTAR PSICOLÓGICO E DEPENDÊNCIA DIGITAL
    # ==================================================
    with tab_bem_estar:
        page_description(
            "Reúne indicadores de comportamento digital para observar como FOMO, "
            "doomscrolling, uso noturno, dificuldade de foco e qualidade do sono se "
            "relacionam dentro da amostra analisada."
        )

        if profile_bar is not None and sleep_bar is not None:
            col_bem_1, col_bem_2 = st.columns([1.25, 1], gap="large")

            with col_bem_1:
                with st.container(border=True):
                    section_title("Comportamentos associados aos níveis de FOMO")
                    chart_caption(
                        "Comparação entre uso noturno, doomscrolling e dificuldade de foco."
                    )
                    plot(profile_bar)
                    st.info(
                        "Os indicadores foram normalizados de 0 a 1 para permitir a "
                        "comparação entre escalas diferentes."
                    )

            with col_bem_2:
                with st.container(border=True):
                    section_title("Qualidade do sono por nível de FOMO")
                    chart_caption(
                        "Qualidade média do sono entre os diferentes níveis de FOMO."
                    )
                    plot(sleep_bar)
        else:
            st.info("Não há dados suficientes para exibir as análises de FOMO.")

if __name__ == "__main__":
    main()
