import streamlit as st
import joblib
import pandas as pd

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Sentiment Analyzer AI",
    page_icon="🧠",
    layout="centered"
)

# --- CARREGAMENTO DO MODELO (A MÁGICA DE TROCA) ---
@st.cache_resource
def carregar_modelo():
    # -----------------------------------------------------------
    # 👇 AQUI É ONDE VOCÊ DÁ PARA TESTAR OUTRO MODELO 👇
    # Basta mudar o nome 'sentiment_model.pkl' pelo nome de arquivo do novo modelo a ser testado.
    # -----------------------------------------------------------
    try:
        model = joblib.load('sentiment_model.pkl')
        return model
    except Exception as e:
        return None

pipeline = carregar_modelo()

# --- INTERFACE VISUAL ---
st.title("🧠 Análise de Sentimentos com IA")
st.write("Digite um comentário sobre um produto e a IA descobrirá a emoção.")

# Área de Texto
texto_usuario = st.text_area("Digite o comentário aqui:", height=150)

# Botão de Ação
if st.button("Analisar Sentimento"):
    if not pipeline:
        st.error("❌ Erro: O arquivo do modelo não foi encontrado na pasta.")
    elif not texto_usuario:
        st.warning("⚠️ Por favor, digite algum texto.")
    else:
        # Fazer a previsão
        try:
            # Pega a classe (Pos/Neg)
            predicao = pipeline.predict([texto_usuario])[0]
            # Pega a probabilidade (0.0 a 1.0)
            probs = pipeline.predict_proba([texto_usuario])[0]
            confianca = max(probs)
            
            # --- MOSTRAR RESULTADO COM CORES ---
            st.divider()
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Mostra o "Carimbo" Grande
                if predicao == 'Positivo':
                    st.success(f"## 😄 {predicao}")
                elif predicao == 'Negativo':
                    st.error(f"## 😡 {predicao}")
                else:
                    st.warning(f"## 😐 {predicao}")
            
            with col2:
                # Mostra a Barra de Confiança
                st.write("### Nível de Certeza da IA:")
                st.progress(confianca)
                st.caption(f"A IA tem {confianca*100:.1f}% de certeza.")

        except Exception as e:
            st.error(f"Ocorreu um erro ao processar: {e}")

# --- BARRA LATERAL (SIDEBAR) ---
st.sidebar.header("Sobre o Projeto")
st.sidebar.info(
    """
    Este dashboard valida o modelo de NLP treinado
    com dados de E-commerce.
    
    **Equipe SentimentAPI**
    
    """
)