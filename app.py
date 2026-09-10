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
        return None, None

    usage_candidates = ["avg_daily_sm_hours", "avg_daily_screen_time_hours", "daily_social_media_usage_hours"]
    productivity_candidates = ["productivity_self_rating", "productivity_rating", "productivity_score"]
    night_candidates = ["late_night_scrolling", "nightly_social_media_usage", "night_time_social_media_usage"]
    sleep_candidates = ["sleep_quality", "sleep_quality_score", "avg_sleep_hours"]
    age_candidates = ["age", "user_age", "participant_age"]
    fomo_candidates = ["fomo_score", "fomo", "fomo_level"]

    usage_col = next((c for c in usage_candidates if c in df.columns), None)
    productivity_col = next((c for c in productivity_candidates if c in df.columns), None)
    night_col = next((c for c in night_candidates if c in df.columns), None)
    sleep_col = next((c for c in sleep_candidates if c in df.columns), None)
    age_col = next((c for c in age_candidates if c in df.columns), None)
    fomo_col = next((c for c in fomo_candidates if c in df.columns), None)

    if not all([usage_col, productivity_col, night_col, sleep_col]):
        return None, None

    base_cols = [usage_col, productivity_col, night_col, sleep_col]
    extra_cols = [c for c in [age_col, fomo_col] if c is not None]

    cleaned = df[base_cols + extra_cols].copy()
    cleaned[usage_col] = pd.to_numeric(cleaned[usage_col], errors="coerce")
    cleaned[productivity_col] = pd.to_numeric(cleaned[productivity_col], errors="coerce")

    if fomo_col:
        cleaned[fomo_col] = pd.to_numeric(cleaned[fomo_col], errors="coerce")
    if age_col:
        cleaned[age_col] = pd.to_numeric(cleaned[age_col], errors="coerce")

    sleep_order = {"Very poor": 1, "Poor": 2, "Fair": 3, "Good": 4, "Very good": 5}
    night_order = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3, "Daily": 4}

    cleaned[sleep_col] = cleaned[sleep_col].replace(sleep_order)
    cleaned[night_col] = cleaned[night_col].replace(night_order)

    cleaned = cleaned.dropna(subset=base_cols)

    cleaned["Uso noturno"] = pd.cut(
        cleaned[night_col],
        bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5],
        labels=["Nunca", "Raramente", "Às vezes", "Frequentemente", "Diariamente"],
        include_lowest=True,
    )

    cleaned["Destaque"] = cleaned["Uso noturno"].apply(lambda x: "Foco" if x == "Diariamente" else "Neutro")

    if age_col:
        cleaned["Faixa etária"] = pd.cut(
            cleaned[age_col],
            bins=[0, 24, 34, 44, 54, 150],
            labels=["Até 24", "25–34", "35–44", "45–54", "55+"],
        )

    cols = {
        "usage": usage_col,
        "productivity": productivity_col,
        "night": night_col,
        "sleep": sleep_col,
        "age": age_col,
        "fomo": fomo_col,
    }

    return cleaned, cols


# ==========================================
# 2. CONSTRUÇÃO VISUAL COM GESTALT E PRÉ-ATENÇÃO
# ==========================================
def build_charts(df: pd.DataFrame, usage_col: str, productivity_col: str, sleep_col: str):
    corr = df[[usage_col, productivity_col]].corr().iloc[0, 1]

    # SCATTER PLOT
    scatter = px.scatter(
        df, x=usage_col, y=productivity_col, trendline="ols",
        opacity=0.5, 
        color_discrete_sequence=["#666666"], 
        labels={usage_col: "Uso diário (horas)", productivity_col: "Produtividade (%)"}
    )
    
    if len(scatter.data) > 1:
        scatter.data[1].line.color = "#FF3300" 
        scatter.data[1].line.width = 4

    scatter.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="#F8F9FA", # Cria um contraste elegante com o branco do container
        font=dict(color="#333333", size=12),
        xaxis=dict(showgrid=False, title_font=dict(color="#333333"), tickfont=dict(color="#333333"), fixedrange=False),
        yaxis=dict(showgrid=True, gridcolor="#DDDDDD", title_font=dict(color="#333333"), tickfont=dict(color="#333333"), fixedrange=False),
        margin=dict(l=0, r=0, t=30, b=0)
    )

    # BOX PLOT
    box = px.box(
        df, x="Uso noturno", y=sleep_col, color="Destaque",
        color_discrete_map={"Foco": "#FF3300", "Neutro": "#666666"},
        category_orders={"Uso noturno": ["Nunca", "Raramente", "Às vezes", "Frequentemente", "Diariamente"]},
        labels={"Uso noturno": "Frequência de uso noturno", sleep_col: "Qualidade do sono (1-5)"}
    )
    
    box.update_layout(
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="#F8F9FA", # Cria um contraste elegante com o branco do container
        font=dict(color="#333333", size=12),
        xaxis=dict(showgrid=False, title_font=dict(color="#333333"), tickfont=dict(color="#333333"), fixedrange=False),
        yaxis=dict(showgrid=True, gridcolor="#DDDDDD", title_font=dict(color="#333333"), tickfont=dict(color="#333333"), fixedrange=False),
        margin=dict(l=0, r=0, t=30, b=0)
    )

    return scatter, box, corr


def build_age_charts(df: pd.DataFrame, cols: dict):
    """Par de gráficos novo: uso médio e FOMO médio por faixa etária,
    lado a lado, com a mesma ordem de faixas no eixo X (small multiples)."""
    age_col = cols.get("age")
    fomo_col = cols.get("fomo")

    if not age_col or not fomo_col or "Faixa etária" not in df.columns:
        return None, None

    age_order = ["Até 24", "25–34", "35–44", "45–54", "55+"]
    age_df = df.dropna(subset=["Faixa etária", fomo_col])
    if age_df.empty:
        return None, None

    usage_by_age = (
        age_df.groupby("Faixa etária", observed=True)[cols["usage"]]
        .mean()
        .reindex(age_order)
        .reset_index(name="Uso médio diário (h)")
    )

    fomo_by_age = (
        age_df.groupby("Faixa etária", observed=True)[fomo_col]
        .mean()
        .reindex(age_order)
        .reset_index(name="FOMO médio")
    )

    age_bar = px.bar(
        usage_by_age,
        x="Faixa etária",
        y="Uso médio diário (h)",
        text_auto=".1f",
    )
    age_bar.update_traces(marker_color="#666666", width=0.55)
    age_bar.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="#F8F9FA",
        font=dict(color="#333333", size=12),
        xaxis=dict(showgrid=False, title_font=dict(color="#333333"), tickfont=dict(color="#333333")),
        yaxis=dict(showgrid=True, gridcolor="#DDDDDD", title_font=dict(color="#333333"), tickfont=dict(color="#333333")),
        margin=dict(l=0, r=0, t=30, b=0),
    )

    # Destaca em vermelho a faixa etária com maior FOMO médio; as demais
    # ficam em laranja claro, mesma lógica de destaque do resto do dashboard
    max_val = fomo_by_age["FOMO médio"].max()
    colors = ["#FF3300" if v == max_val else "#FFAD66" for v in fomo_by_age["FOMO médio"]]

    fomo_bar = px.bar(
        fomo_by_age,
        x="Faixa etária",
        y="FOMO médio",
        text_auto=".1f",
    )
    fomo_bar.update_traces(marker_color=colors, width=0.55)
    fomo_bar.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="#F8F9FA",
        font=dict(color="#333333", size=12),
        xaxis=dict(showgrid=False, title_font=dict(color="#333333"), tickfont=dict(color="#333333")),
        yaxis=dict(showgrid=True, gridcolor="#DDDDDD", range=[0, 10], title_font=dict(color="#333333"), tickfont=dict(color="#333333")),
        margin=dict(l=0, r=0, t=30, b=0),
    )

    return age_bar, fomo_bar


# ==========================================
# 3. INTERFACE E APLICAÇÃO DO MANTRA
# ==========================================
def main():
    st.set_page_config(page_title="Dashboard Social Media & Produtividade", layout="wide")
    
    df = load_dataset()
    if df is None:
        st.error("Dataset não encontrado.")
        return

    cleaned, cols = prepare_dataframe(df)
    if cleaned is None:
        st.error("Falha ao processar colunas.")
        return

    usage_col = cols["usage"]
    productivity_col = cols["productivity"]
    sleep_col = cols["sleep"]

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

    with st.container(border=True):
        st.subheader("Visão Geral da Amostra Filtrada")
        c1, c2, c3 = st.columns(3)
        c1.metric("Participantes", len(df_filtrado))
        c2.metric("Tempo Médio Diário", f"{df_filtrado[usage_col].mean():.1f} h")
        c3.metric("Produtividade Média", f"{df_filtrado[productivity_col].mean():.1f}%")

    if len(df_filtrado) > 5:
        scatter, box, corr = build_charts(df_filtrado, usage_col, productivity_col, sleep_col)
        
    # PERGUNTA CENTRAL (Limpa, direta e com atributo pré-atentivo de cor)
        with st.container(border=True):
            st.markdown(
                """
                <p style='color: #FF3300; font-size: 1.2em; font-weight: bold; margin-bottom: 10px;'>
                Existe associação entre o tempo de uso diário de redes sociais e a queda na taxa de produtividade autorrelatada?
                </p>
                """, unsafe_allow_html=True
            )
            st.plotly_chart(scatter, use_container_width=True)
            st.info(f"**Correlação de Pearson:** {corr:.2f} *(Valores negativos indicam queda de produtividade conforme o uso aumenta)*")

        # SEGUNDA PERGUNTA
        with st.container(border=True):
            st.markdown(
                """
                <p style='font-size: 1.2em; font-weight: bold; margin-bottom: 10px;'>
                O hábito de uso de mídias sociais no período noturno afeta diretamente a qualidade do sono percebida?
                </p>
                """, unsafe_allow_html=True
            )
            st.plotly_chart(box, use_container_width=True)

        # TERCEIRA PERGUNTA (novos gráficos: uso e FOMO por faixa etária)
        age_bar, fomo_bar = build_age_charts(df_filtrado, cols)

        if age_bar is not None:
            with st.container(border=True):
                st.markdown(
                    """
                    <p style='color: #FF3300; font-size: 1.2em; font-weight: bold; margin-bottom: 10px;'>
                    Como o uso diário de redes sociais e o nível de FOMO variam entre as diferentes faixas etárias?
                    </p>
                    """, unsafe_allow_html=True
                )

                col_a, col_b = st.columns(2)
                with col_a:
                    st.caption("Uso médio diário por faixa etária")
                    st.plotly_chart(age_bar, use_container_width=True)
                with col_b:
                    st.caption("Nível médio de FOMO por faixa etária")
                    st.plotly_chart(fomo_bar, use_container_width=True)

                st.info(
                    "**Como comparar:** os dois gráficos compartilham o mesmo eixo de faixas "
                    "etárias, observe se os picos de uso coincidem com os picos de FOMO "
                    "(faixa destacada em vermelho)."
                )
        else:
            st.warning(
                "Colunas de idade e/ou FOMO não encontradas no dataset"
            )

    else:
        st.warning("Dados insuficientes para os filtros selecionados.")

    with st.expander("Ver Dados Brutos (Details on Demand)"):
        st.dataframe(df_filtrado.drop(columns=["Destaque"], errors="ignore"), use_container_width=True)

if __name__ == "__main__":
    main()