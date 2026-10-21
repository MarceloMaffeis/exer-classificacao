import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Controle de Qualidade - Usinagem",
    page_icon="⚙️",
    layout="wide"
)

# Mapeamento numérico para rótulos legíveis
CLASS_NAMES = {
    0: "Refugo",
    1: "Retrabalho",
    2: "Aprovada"
}

# --- GERAÇÃO DE DADOS E TREINAMENTO DO MODELO (COM CACHE) ---
@st.cache_resource
def carregar_dados_e_treinar_modelo():
    # Centros nominais definidos para [Peso_gramas, Espessura_mm]
    # 0: Refugo, 1: Retrabalho, 2: Aprovada
    centers = [
        [120.0, 8.5],   # Refugo (muito leve / fino)
        [145.0, 10.2],  # Retrabalho (intermediário / ajustes)
        [150.0, 12.0]   # Aprovada (especificação ideal)
    ]
    
    X, y = make_blobs(
        n_samples=600,
        centers=centers,
        cluster_std=[4.0, 3.0, 2.5],
        random_state=42
    )

    # Criação do DataFrame completo para visualização
    df = pd.DataFrame(X, columns=["Peso_gramas", "Espessura_mm"])
    df["Status"] = [CLASS_NAMES[label] for label in y]
    df["Peso_gramas"] = df["Peso_gramas"].round(2)
    df["Espessura_mm"] = df["Espessura_mm"].round(2)

    # Divisão em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Treinamento do SVM com kernel RBF
    model = SVC(kernel="rbf", C=1.0, probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Matriz de confusão no conjunto de validação
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)

    return model, scaler, cm, df

model, scaler, cm, df_pecas = carregar_dados_e_treinar_modelo()

# --- INTERFACE STREAMLIT ---
st.title("⚙️ Sistema de Controle de Qualidade de Peças")
st.markdown("Inspeção dimensional e de massa para classificação automática de peças usinadas via **Support Vector Machine (SVM)**.")

st.divider()

# Layout em duas colunas: Entrada do Inspetor e Matriz de Confusão
col_input, col_metrics = st.columns([1, 1], gap="large")

# --- COLUNA 1: ENTRADA DO INSPETOR ---
with col_input:
    st.subheader("📋 Painel do Inspetor")
    st.write("Ajuste os parâmetros medidos da peça:")

    peso = st.slider(
        label="Peso da Peça (gramas):",
        min_value=100.0,
        max_value=170.0,
        value=148.0,
        step=0.1
    )

    espessura = st.slider(
        label="Espessura da Peça (mm):",
        min_value=5.0,
        max_value=16.0,
        value=11.5,
        step=0.05
    )

    st.write("")
    btn_avaliar = st.button("🔍 Avaliar Peça", type="primary", use_container_width=True)

    if btn_avaliar:
        entrada = np.array([[peso, espessura]])
        entrada_scaled = scaler.transform(entrada)
        predicao = model.predict(entrada_scaled)[0]
        resultado = CLASS_NAMES[predicao]

        st.markdown("### Resultado da Inspeção:")
        if resultado == "Aprovada":
            st.success(f"✅ **Peça Aprovada!** (Dentro das tolerâncias nominais)")
        elif resultado == "Retrabalho":
            st.warning(f"⚠️ **Peça para Retrabalho!** (Necessita de ajustes dimensionais)")
        else:
            st.error(f"❌ **Peça de Refugo!** (Fora dos limites recuperáveis)")

# --- COLUNA 2: MÉTRICAS E MATRIZ DE CONFUSÃO ---
with col_metrics:
    st.subheader("📊 Matriz de Confusão do Modelo")
    st.caption("Desempenho do classificador SVM no conjunto de validação:")

    labels = ["Refugo", "Retrabalho", "Aprovada"]
    
    fig, ax = plt.subplots(figsize=(5, 3.8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar=False
    )
    ax.set_xlabel("Predição do Modelo", fontsize=11, fontweight="bold")
    ax.set_ylabel("Classe Real", fontsize=11, fontweight="bold")
    plt.tight_layout()

    st.pyplot(fig)

# --- SEÇÃO INFERIOR: VISUALIZAÇÃO DO DATAFRAME ---
st.divider()
st.subheader("📁 Base de Dados Sintética de Inspeção")
st.caption("Amostras geradas com features de processo (`Peso_gramas`, `Espessura_mm`) e o respectivo status de qualidade.")

# Filtro interativo para explorar o DataFrame
col_filtro, col_total = st.columns([1, 2])
with col_filtro:
    filtro_status = st.multiselect(
        "Filtrar por Status:",
        options=list(CLASS_NAMES.values()),
        default=list(CLASS_NAMES.values())
    )

df_filtrado = df_pecas[df_pecas["Status"].isin(filtro_status)]

with col_total:
    st.metric("Total de Peças Exibidas", len(df_filtrado))

# Renderização do DataFrame interativo
st.dataframe(
    df_filtrado,
    use_container_width=True,
    hide_index=True
)
