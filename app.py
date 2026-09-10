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
AGE = "age"  # opcional: confira no df.columns se é esse o nome real

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

def style_chart(fig, y_range=None):
    fig.update_layout(
        showlegend=False,
        plot_bgcolor="#F8F9FA",
        paper_bgcolor="#F8F9FA",
        font=dict(color="#000000", size=12),
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
        plot_bgcolor="#F8F9FA",
        paper_bgcolor="#F8F9FA",
        font=dict(color="#000000", size=12),
        margin=dict(l=0, r=0, t=30, b=0),
    )

    return scatter, box, focus_bar, heatmap, corr


def build_age_charts(df):
    """Par de gráficos: uso médio e FOMO médio por faixa etária, lado a
    lado, com a mesma ordem de faixas no eixo X (small multiples)."""
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
    )
    age_bar.update_traces(marker_color="#666666", width=0.55)
    style_chart(age_bar)

    max_val = fomo_by_age["FOMO médio"].max()
    colors = [
        "#FF3300" if v == max_val else "#FFAD66"
        for v in fomo_by_age["FOMO médio"]
    ]

    fomo_bar = px.bar(
        fomo_by_age,
        x="Faixa etária",
        y="FOMO médio",
        text_auto=".1f",
    )
    fomo_bar.update_traces(marker_color=colors, width=0.55)
    style_chart(fomo_bar, [0, 10])

    return age_bar, fomo_bar

def build_dependency_charts(df):
    """Par de gráficos: perfil de comportamento compulsivo (uso noturno,
    doomscrolling e dificuldade de foco) e qualidade do sono, comparando
    diferentes níveis de FOMO. Complementa a pergunta de doomscrolling/FOMO,
    não conta como uma pergunta nova."""
    dep_df = df.dropna(subset=["Nível de FOMO"]).copy()
    if dep_df.empty:
        return None, None

    dep_df["Doom_num"] = dep_df[DOOM].map(NIGHT_MAP)

    profile = (
        dep_df.groupby("Nível de FOMO", observed=True)[[NIGHT, "Doom_num", FOCUS]]
        .mean()
        .reindex(FOMO_ORDER)
    )
    profile["Uso noturno (0-1)"] = profile[NIGHT] / 4
    profile["Doomscrolling (0-1)"] = profile["Doom_num"] / 4
    profile["Dificuldade de foco (0-1)"] = profile[FOCUS] / 3

    profile_long = (
        profile[["Uso noturno (0-1)", "Doomscrolling (0-1)", "Dificuldade de foco (0-1)"]]
        .reset_index()
        .melt(id_vars="Nível de FOMO", var_name="Indicador", value_name="Intensidade média")
    )

    profile_bar = px.bar(
        profile_long,
        x="Nível de FOMO",
        y="Intensidade média",
        color="Indicador",
        barmode="group",
        category_orders={"Nível de FOMO": FOMO_ORDER},
        color_discrete_sequence=["#666666", "#FFAD66", "#FF3300"],
    )
    style_chart(profile_bar, [0, 1])
    profile_bar.update_layout(showlegend=True)  # aqui a legenda ajuda a distinguir os 3 indicadores

    sleep_by_fomo = (
        dep_df.groupby("Nível de FOMO", observed=True)[SLEEP]
        .mean()
        .reindex(FOMO_ORDER)
        .reset_index(name="Qualidade média do sono")
    )
    min_val = sleep_by_fomo["Qualidade média do sono"].min()
    colors = [
        "#FF3300" if v == min_val else "#FFAD66"
        for v in sleep_by_fomo["Qualidade média do sono"]
    ]

    sleep_bar = px.bar(
        sleep_by_fomo,
        x="Nível de FOMO",
        y="Qualidade média do sono",
        text_auto=".2f",
    )
    sleep_bar.update_traces(marker_color=colors, width=0.55)
    style_chart(sleep_bar, [0, 5])

    return profile_bar, sleep_bar


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
    age_bar, fomo_bar = build_age_charts(filtered)
    profile_bar, sleep_bar = build_dependency_charts(filtered)

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

        if profile_bar is not None:
            st.divider()
            st.caption(
                "Aprofundando: níveis maiores de FOMO estão associados a outros comportamentos e à qualidade do sono?"
            )

            col_x, col_y = st.columns(2)

            with col_x:
                st.caption("Comportamentos relacionados ao uso de redes sociais por nível de FOMO")
                st.plotly_chart(profile_bar, use_container_width=True)

            with col_y:
                st.caption("Qualidade média do sono por nível de FOMO")
                st.plotly_chart(sleep_bar, use_container_width=True)

            st.info(
                "**Como comparar:** à esquerda, cada indicador é normalizado "
                "de 0 a 1 — quanto mais alta a barra, mais intenso o "
                "comportamento naquele grupo de FOMO. À direita, a barra em "
                "vermelho marca o grupo com pior sono médio."
            )

    if age_bar is not None:
        with st.container(border=True):
            question(
                "Como o uso diário de redes sociais e o nível de FOMO "
                "variam entre as diferentes faixas etárias?",
                True,
            )

            col_a, col_b = st.columns(2)

            with col_a:
                st.caption("Uso médio diário por faixa etária")
                st.plotly_chart(age_bar, use_container_width=True)

            with col_b:
                st.caption("Nível médio de FOMO por faixa etária")
                st.plotly_chart(fomo_bar, use_container_width=True)

            st.info(
                "**Como comparar:** os dois gráficos compartilham o mesmo "
                "eixo de faixas etárias, observe se os picos de uso "
                "coincidem com os picos de FOMO (faixa destacada em "
                "vermelho)."
            )
    else:
        st.warning(
            f"Coluna '{AGE}' não encontrada no dataset — ajuste a "
            "constante AGE, no topo do arquivo, para o nome real da "
            "coluna de idade."
        )

    with st.expander("Ver Dados Brutos (Details on Demand)"):
        st.dataframe(
            filtered.drop(columns=["Destaque"], errors="ignore"),
            use_container_width=True,
        )


if __name__ == "__main__":
    main()