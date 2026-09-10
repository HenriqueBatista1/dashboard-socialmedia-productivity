import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import kagglehub

DATASET_SLUG = "manaswinsripatnala/social-media-dopamine-and-productivity-dataset"

# ==========================================
# 1. CARREGAMENTO E TRATAMENTO DE DADOS
# ==========================================
def iter_data_files(root: Path):
    if not root.exists():
        return []
    files = []
    for item in root.rglob("*"):
        if not item.is_file():
            continue
        if any(part in {".venv", ".git", "__pycache__"} for part in item.parts):
            continue
        if item.suffix.lower() in {".csv", ".xlsx", ".xls"}:
            files.append(item)
    return sorted(files)

@st.cache_data
def load_dataset():
    local_candidates = []
    for base in [Path.cwd(), Path.cwd() / "data", Path.cwd() / "datasets"]:
        local_candidates.extend(iter_data_files(base))

    if local_candidates:
        for path in local_candidates:
            try:
                df = _read_file(path)
                if df is not None and not df.empty:
                    return df
            except Exception:
                continue

    try:
        dataset_path = kagglehub.dataset_download(DATASET_SLUG)
        dataset_root = Path(dataset_path)
        files = []
        for ext in ("*.csv", "*.xlsx", "*.xls"):
            files.extend(dataset_root.rglob(ext))

        for path in sorted(files):
            try:
                df = _read_file(path)
                if df is not None and not df.empty:
                    return df
            except Exception:
                continue
    except Exception as exc:
        st.warning("Não foi possível baixar o dataset automaticamente.")
        st.code(str(exc), language="text")

    return None

def _read_file(path: Path):
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return None

def prepare_dataframe(df: pd.DataFrame):
    if df is None or df.empty:
        return None, None, None, None

    usage_candidates = ["avg_daily_sm_hours", "avg_daily_screen_time_hours", "daily_social_media_usage_hours"]
    productivity_candidates = ["productivity_self_rating", "productivity_rating", "productivity_score"]
    night_candidates = ["late_night_scrolling", "nightly_social_media_usage", "night_time_social_media_usage"]
    sleep_candidates = ["sleep_quality", "sleep_quality_score", "avg_sleep_hours"]

    usage_col = next((c for c in usage_candidates if c in df.columns), None)
    productivity_col = next((c for c in productivity_candidates if c in df.columns), None)
    night_col = next((c for c in night_candidates if c in df.columns), None)
    sleep_col = next((c for c in sleep_candidates if c in df.columns), None)

    if not all([usage_col, productivity_col, night_col, sleep_col]):
        return None, None, None, None

    cleaned = df[[usage_col, productivity_col, night_col, sleep_col]].copy()
    cleaned[usage_col] = pd.to_numeric(cleaned[usage_col], errors="coerce")
    cleaned[productivity_col] = pd.to_numeric(cleaned[productivity_col], errors="coerce")

    sleep_order = {"Very poor": 1, "Poor": 2, "Fair": 3, "Good": 4, "Very good": 5}
    night_order = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3, "Daily": 4}

    cleaned[sleep_col] = cleaned[sleep_col].replace(sleep_order)
    cleaned[night_col] = cleaned[night_col].replace(night_order)
    cleaned = cleaned.dropna()

    cleaned["Uso noturno"] = pd.cut(
        cleaned[night_col],
        bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5],
        labels=["Nunca", "Raramente", "Às vezes", "Frequentemente", "Diariamente"],
        include_lowest=True,
    )

    # Criação de variável para destaque pré-atentivo no Boxplot
    cleaned["Destaque"] = cleaned["Uso noturno"].apply(lambda x: "Foco" if x == "Diariamente" else "Neutro")

    return cleaned, usage_col, productivity_col, sleep_col


# ==========================================
# 2. CONSTRUÇÃO VISUAL COM GESTALT E PRÉ-ATENÇÃO
# ==========================================
def build_charts(df: pd.DataFrame, usage_col: str, productivity_col: str, sleep_col: str):
    corr = df[[usage_col, productivity_col]].corr().iloc[0, 1]

    # SCATTER PLOT - Destaque Pré-Atentivo na Linha de Tendência
    scatter = px.scatter(
        df, x=usage_col, y=productivity_col, trendline="ols",
        opacity=0.4, 
        color_discrete_sequence=["#A9A9A9"], # Fundo neutro (Chunking/Figura-fundo)
        labels={usage_col: "Uso diário (horas)", productivity_col: "Produtividade (%)"}
    )
    
    # Destacando a resposta central com cor quente
    if len(scatter.data) > 1:
        scatter.data[1].line.color = "#FF3300" 
        scatter.data[1].line.width = 4

    scatter.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, title_font=dict(size=14)),
        yaxis=dict(showgrid=True, gridcolor="#E5E5E5", title_font=dict(size=14)),
        margin=dict(l=0, r=0, t=30, b=0)
    )

    # BOX PLOT - Destaque Pré-Atentivo na categoria mais extrema
    box = px.box(
        df, x="Uso noturno", y=sleep_col, color="Destaque",
        color_discrete_map={"Foco": "#FF3300", "Neutro": "#A9A9A9"},
        category_orders={"Uso noturno": ["Nunca", "Raramente", "Às vezes", "Frequentemente", "Diariamente"]},
        labels={"Uso noturno": "Frequência de uso noturno", sleep_col: "Qualidade do sono (1-5)"}
    )
    
    box.update_layout(
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, title_font=dict(size=14)),
        yaxis=dict(showgrid=True, gridcolor="#E5E5E5", title_font=dict(size=14)),
        margin=dict(l=0, r=0, t=30, b=0)
    )

    return scatter, box, corr


# ==========================================
# 3. INTERFACE E APLICAÇÃO DO MANTRA
# ==========================================
def main():
    st.set_page_config(page_title="Dashboard Social Media & Produtividade", layout="wide")
    
    df = load_dataset()
    if df is None:
        st.error("Dataset não encontrado.")
        return

    cleaned, usage_col, productivity_col, sleep_col = prepare_dataframe(df)
    if cleaned is None:
        st.error("Falha ao processar colunas.")
        return

    # MANTRA: ZOOM & FILTER (Painel Lateral)
    st.sidebar.header("🔍 Filtros de Análise")
    st.sidebar.markdown("*(Zoom/Filter)*")
    
    filtro_uso = st.sidebar.slider(
        "Filtrar por Horas de Uso:", 
        min_value=float(cleaned[usage_col].min()), 
        max_value=float(cleaned[usage_col].max()), 
        value=(0.0, float(cleaned[usage_col].max()))
    )
    
    df_filtrado = cleaned[
        (cleaned[usage_col] >= filtro_uso[0]) & 
        (cleaned[usage_col] <= filtro_uso[1])
    ]

    st.title("📊 Dashboard: Mídias Sociais, Dopamina e Produtividade")
    st.caption("Análise do impacto do uso de redes sociais no foco e produtividade (Dados autorrelatados).")

    # MANTRA: OVERVIEW (Visão Geral - Aplicando Gestalt de Fechamento)
    with st.container(border=True):
        st.subheader("Visão Geral da Amostra Filtrada")
        c1, c2, c3 = st.columns(3)
        c1.metric("Participantes", len(df_filtrado))
        c2.metric("Tempo Médio Diário", f"{df_filtrado[usage_col].mean():.1f} h")
        c3.metric("Produtividade Média", f"{df_filtrado[productivity_col].mean():.1f}%")

    if len(df_filtrado) > 5:
        scatter, box, corr = build_charts(df_filtrado, usage_col, productivity_col, sleep_col)
        
        # PERGUNTA CENTRAL (Gestalt: Proximidade e Fechamento)
        with st.container(border=True):
            st.markdown(
                """
                <h3 style='color: #FF3300; margin-bottom: 0px;'>Pergunta Central</h3>
                <p style='font-size: 1.1em; font-weight: bold;'>Existe associação entre o tempo de uso diário de redes sociais e a queda na taxa de produtividade autorrelatada?</p>
                """, unsafe_allow_html=True
            )
            st.plotly_chart(scatter, use_container_width=True)
            st.info(f"**Correlação de Pearson:** {corr:.2f} *(Valores negativos indicam queda de produtividade conforme o uso aumenta)*")

        # PERGUNTA SECUNDÁRIA
        with st.container(border=True):
            st.markdown(
                """
                <h3 style='margin-bottom: 0px;'>Segunda Pergunta Principal</h3>
                <p style='font-size: 1.1em;'>O hábito de uso de mídias sociais no período noturno afeta diretamente a qualidade do sono percebida?</p>
                """, unsafe_allow_html=True
            )
            st.plotly_chart(box, use_container_width=True)

    else:
        st.warning("Dados insuficientes para os filtros selecionados.")

    # MANTRA: DETAILS ON DEMAND (Detalhes sob Demanda)
    with st.expander("Ver Dados Brutos (Details on Demand)"):
        st.dataframe(df_filtrado.drop(columns=["Destaque"]), use_container_width=True)

if __name__ == "__main__":
    main()