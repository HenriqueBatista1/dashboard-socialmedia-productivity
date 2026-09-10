from pathlib import Path

import kagglehub
import pandas as pd
import plotly.express as px
import streamlit as st


DATASET_SLUG = "manaswinsripatnala/social-media-dopamine-and-productivity-dataset"

USAGE = "avg_daily_sm_hours"
PRODUCTIVITY = "productivity_self_rating"
NIGHT = "late_night_scrolling"
SLEEP = "sleep_quality"
FOCUS = "difficulty_maintaining_focus"
DOOM = "doomscrolling_frequency"
FOMO = "fomo_score"

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


# ==========================================
# CARREGAMENTO
# ==========================================

def find_files(root):
    if not root.exists():
        return []

    return sorted(
        p for p in root.rglob("*")
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
                if not df.empty:
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
        st.error("Colunas ausentes: " + ", ".join(missing))
        return None

    df = df[REQUIRED].copy()

    for col in [USAGE, PRODUCTIVITY, FOMO]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[SLEEP] = df[SLEEP].map(SLEEP_MAP)
    df[NIGHT] = df[NIGHT].map(NIGHT_MAP)
    df[FOCUS] = df[FOCUS].map(FOCUS_MAP)

    df = df.dropna()

    df["Uso noturno"] = pd.cut(
        df[NIGHT],
        [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5],
        labels=NIGHT_ORDER,
    )

    df["Destaque"] = df["Uso noturno"].apply(
        lambda x: "Foco" if x == "Diariamente" else "Neutro"
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

    return df


# ==========================================
# ESTILO
# ==========================================

def style_chart(fig, y_range=None):
    fig.update_layout(
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="#F8F9FA",
        font=dict(color="#333333", size=12),
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#DDDDDD", range=y_range),
    )


def question(text, orange=False):
    color = "#FF3300" if orange else "#333333"

    st.markdown(
        f"""
        <p style="color:{color}; font-size:1.2em; font-weight:bold; margin-bottom:10px;">
            {text}
        </p>
        """,
        unsafe_allow_html=True,
    )


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
        opacity=0.5,
        color_discrete_sequence=["#666666"],
        labels={
            USAGE: "Uso diário (horas)",
            PRODUCTIVITY: "Produtividade (%)",
        },
    )

    if len(scatter.data) > 1:
        scatter.data[1].line.color = "#FF3300"
        scatter.data[1].line.width = 4

    style_chart(scatter)

    box = px.box(
        df,
        x="Uso noturno",
        y=SLEEP,
        color="Destaque",
        color_discrete_map={"Foco": "#FF3300", "Neutro": "#666666"},
        category_orders={"Uso noturno": NIGHT_ORDER},
        labels={
            "Uso noturno": "Frequência de uso noturno",
            SLEEP: "Qualidade do sono (1-5)",
        },
    )

    style_chart(box)

    focus_df = (
        df.groupby("Intensidade de uso", observed=True)[FOCUS]
        .mean()
        .reset_index(name="Dificuldade média de foco")
    )

    focus_bar = px.bar(
        focus_df,
        x="Intensidade de uso",
        y="Dificuldade média de foco",
        color="Intensidade de uso",
        text_auto=".2f",
        color_discrete_map={
            "Baixo uso": "#FFD6AD",
            "Uso moderado": "#FFAD66",
            "Alto uso": "#FF7A21",
            "Uso muito alto": "#E94F00",
        },
        category_orders={"Intensidade de uso": USAGE_ORDER},
        labels={
            "Intensidade de uso": "Intensidade de uso das redes sociais",
            "Dificuldade média de foco": "Dificuldade média de foco (0-3)",
        },
    )

    focus_bar.update_traces(width=0.55)
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
            "#F5F3FF",
            "#DDD6FE",
            "#A78BFA",
            "#7C3AED",
            "#4C1D95",
        ],
        labels={
            "x": "Nível de FOMO",
            "y": "Frequência de doomscrolling",
            "color": "Participantes",
        },
    )

    heatmap.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="#F8F9FA",
        font=dict(color="#333333", size=12),
        margin=dict(l=0, r=0, t=30, b=0),
    )

    return scatter, box, focus_bar, heatmap, corr


# ==========================================
# INTERFACE
# ==========================================

def main():
    st.set_page_config(
        page_title="Dashboard Social Media & Produtividade",
        layout="wide",
    )

    df = prepare_dataframe(load_dataset())

    if df is None:
        st.error("Dataset não encontrado ou inválido.")
        return

    st.sidebar.header("🔍 Filtros de Análise")
    st.sidebar.markdown("*(Zoom/Filter)*")

    min_use = float(df[USAGE].min())
    max_use = float(df[USAGE].max())

    filtro = st.sidebar.slider(
        "Filtrar por Horas de Uso:",
        min_use,
        max_use,
        (min_use, max_use),
    )

    filtered = df[df[USAGE].between(*filtro)]

    st.title("📊 Dashboard: Mídias Sociais, Dopamina e Produtividade")
    st.caption(
        "Análise de associações entre uso de redes sociais, "
        "foco, sono e produtividade."
    )

    with st.container(border=True):
        st.subheader("Visão Geral da Amostra Filtrada")

        c1, c2, c3 = st.columns(3)

        c1.metric("Participantes", len(filtered))
        c2.metric("Tempo Médio Diário", f"{filtered[USAGE].mean():.1f} h")
        c3.metric(
            "Produtividade Média",
            f"{filtered[PRODUCTIVITY].mean():.1f}%",
        )

    if len(filtered) <= 5:
        st.warning("Dados insuficientes.")
        return

    scatter, box, focus_bar, heatmap, corr = build_charts(filtered)

    with st.container(border=True):
        question(
            "Existe associação entre o tempo de uso diário de redes sociais "
            "e a queda na taxa de produtividade autorrelatada?",
            True,
        )

        st.plotly_chart(scatter, use_container_width=True)
        st.info(f"**Correlação de Pearson:** {corr:.2f}")

    with st.container(border=True):
        question(
            "Existe associação entre a frequência de uso noturno "
            "e a qualidade do sono percebida?"
        )

        st.plotly_chart(box, use_container_width=True)

    with st.container(border=True):
        question(
            "Existe diferença na dificuldade de foco entre usuários "
            "com diferentes intensidades de uso?",
            True,
        )

        st.plotly_chart(focus_bar, use_container_width=True)

        st.info(
            "**Escala de dificuldade de foco:** "
            "Nunca = 0, Raramente = 1, Às vezes = 2 e Sim = 3."
        )

    with st.container(border=True):
        question(
            "Como a frequência de doomscrolling se distribui "
            "entre diferentes níveis de FOMO?",
            True,
        )

        st.plotly_chart(heatmap, use_container_width=True)

        st.info(
            "**Como interpretar:** cada célula mostra a quantidade "
            "de participantes naquela combinação. Tons de roxo mais "
            "escuros indicam maior concentração de participantes."
        )

    with st.expander("Ver Dados Brutos (Details on Demand)"):
        st.dataframe(
            filtered.drop(columns=["Destaque"], errors="ignore"),
            use_container_width=True,
        )


if __name__ == "__main__":
    main()