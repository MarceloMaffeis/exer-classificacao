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

# --- GERAÇÃO E TREINAMENTO DO MODELO (COM CACHE) ---
@st.cache_resource
def treinar_modelo():
    # Classes: 0: Refugo, 1: Retrabalho, 2: Aprovada
    # Centros definidos para [Peso_gramas, Espessura_mm]
    centers = [
        [120.0, 8.5],   # 0: Refugo (muito leve / fino)
        [145.0, 10.2],  # 1: Retrabalho (intermediário / ajustes)
        [150.0, 12.0]   # 2: Aprovada (especificação ideal)
    ]
    
    X, y = make_blobs(
        n_samples=600,
        centers=centers,
        cluster_std=[4.0, 3.0, 2.5],
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Treinamento do SVM com kernel RBF
    model = SVC(kernel="rbf", C=1.0, probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Avaliação no conjunto de teste para a matriz de confusão
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)

    return model, scaler, cm

model, scaler, cm = treinar_modelo()

# Mapeamento das classes
class_names = {
    0: "Refugo",
    1: "Retrabalho",
    2: "Aprovada"
}

# --- INTERFACE STREAMLIT ---
st.title("⚙️ Sistema de Controle de Qualidade de Peças")
st.markdown("Inspeção dimensional e de massa para classificação automática de peças usinadas via **Support Vector Machine (SVM)**.")

st.divider()

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
        # Prepara os dados de entrada
        entrada = np.array([[peso, espessura]])
        entrada_scaled = scaler.transform(entrada)
        
        predicao = model.predict(entrada_scaled)[0]
        resultado = class_names[predicao]

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
    
    fig, ax = plt.subplots(figsize=(5, 4))
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
