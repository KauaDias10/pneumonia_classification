#FUNFOU - COMPLETO
import streamlit as st
import tensorflow as tf
import io
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px

# Configuração da página
st.set_page_config(
    page_title="Classificador de Pneumonia",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Carregar CSS personalizado
def load_css():
    try:
        with open("assets/css/style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        # CSS fallback básico
        st.warning("⚠️ CSS externo não encontrado. Nenhum estilo será aplicado.")
load_css()

# --- Menu Lateral ---
def sidebar():
    with st.sidebar:
        st.title("PneumoScan")
        
        # Menu de navegação simplificado usando radio buttons
        page = st.radio(
            "Navegação",
            ["🏠 Classificação", "ℹ️ Sobre o Modelo"],
            index=0
        )
        
        st.markdown("---")
        
        # Informações do modelo na sidebar
        st.subheader("📋 Especificações")
        st.markdown("""
        **Arquitetura:** MobileNetV2  
        **Dataset:** Chest X-Ray Images  
        **Classes:** Normal vs Pneumonia  
        **Acurácia:** 92% (validação)  
        **Especialidade:** Radiologia Torácica
        """)
        
        st.markdown("---")
        
        return page

# --- 🔹 Função para carregar imagem ---
def carrega_imagem():
    
    uploaded_file = st.file_uploader(
        '**🩺 Envie uma radiografia de tórax:**',
        type=['png', 'jpg', 'jpeg'],
        help="Formatos suportados: PNG, JPG, JPEG"
    )

    if uploaded_file is not None:
        try:
            image_data = uploaded_file.read()
            image = Image.open(io.BytesIO(image_data))
            image = image.convert("RGB")

            # Centralizar a imagem usando columns
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:  # Coluna do meio para centralizar
                st.image(image, caption='Imagem carregada com sucesso!', use_container_width=True)

            # Redimensiona para o tamanho esperado pelo modelo
            image = image.resize((224, 224))

            # Converte para array NumPy e normaliza
            image_array = np.array(image, dtype=np.float32) / 255.0

            # Adiciona dimensão do batch: [1, 224, 224, 3]
            image_array = np.expand_dims(image_array, axis=0)

            return image_array, uploaded_file.name
            
        except Exception as e:
            st.error(f"Erro ao processar imagem: {str(e)}")
            return None, None

    return None, None

# --- 🔹 Função para carregar modelo ---
@st.cache_resource
def carrega_modelo_h5():
    try:
        model = tf.keras.models.load_model('cnn_models/mobilenetv2_FASE1.h5')
        st.sidebar.success("Modelo carregado!")
        return model
    except Exception as e:
        st.sidebar.error(f"Erro ao carregar modelo: {str(e)}")
        return None

# --- 🔹 Função de previsão ---
def previsao_h5(_model, image, filename):
    # Container para resultados
    result_container = st.container()
    
    with result_container:
        st.subheader("Resultados da Análise")
        
        # Faz a previsão
        with st.spinner('🔍 Analisando radiografia...'):
            pred = _model.predict(image, verbose=0)
            prob_pneumonia = float(pred[0][0])
            prob_normal = 1 - prob_pneumonia

        # Layout em colunas para resultados
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Card de resultado
            classe_predita = "Pneumonia" if prob_pneumonia > 0.5 else "Normal"
            probabilidade = prob_pneumonia * 100 if classe_predita == "Pneumonia" else prob_normal * 100
            confidence_color = "#0b1a2a" if classe_predita == "Pneumonia" else "#1e88e5"
            
            st.markdown(f"""
            <div class='result-card'>
                <h3 style='color: {confidence_color};'>{classe_predita}</h3>
                <h2 style='color: {confidence_color};'>{probabilidade:.1f}%</h2>
                <p>Confiança da predição</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            # Gráfico de probabilidades
            classes = ['Normal', 'Pneumonia']
            probabilidades = [prob_normal * 100, prob_pneumonia * 100]
            
            df = pd.DataFrame({
                'Classe': classes,
                'Probabilidade (%)': probabilidades
            })

            fig = px.bar(
                df,
                y='Classe',
                x='Probabilidade (%)',
                orientation='h',
                text='Probabilidade (%)',
                color='Classe',
                color_discrete_map={'Normal': '#1e88e5', 'Pneumonia': '#0b1a2a'},
                title='Distribuição de Probabilidades'
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # Informações técnicas
        with st.expander("📋 Detalhes Técnicos"):
            st.write(f"**Arquivo analisado:** {filename}")
            st.write(f"**Dimensões da imagem:** 224x224 pixels")
            st.write(f"**Modelo utilizado:** MobileNetV2")
            st.write(f"**Probabilidade Pneumonia:** {prob_pneumonia:.4f}")
            st.write(f"**Probabilidade Normal:** {prob_normal:.4f}")

# --- 🔹 Página de Classificação ---
def pagina_classificacao():

    with st.container():
        st.title("PneumoScan")
        st.write("**Sistema Inteligente de Análise de Radiografias de Tórax**")
    st.markdown("</div>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown("""
        <div class='intro-section'>
        <p>Este sistema utiliza <strong>Inteligência Artificial</strong> baseada em redes neurais convolucionais 
        para auxiliar na identificação de pneumonia em radiografias de tórax.<br>
        <strong>Importante:</strong> Este é um sistema de auxílio diagnóstico e não substitui a avaliação médica profissional.</p>
        </div>
        """, unsafe_allow_html=True)
    
    image_array, filename = carrega_imagem()

    if image_array is not None:
        model = carrega_modelo_h5()
        if model is not None:
            previsao_h5(model, image_array, filename)

# --- 🔹 Página Sobre ---
def pagina_sobre():
    st.title("ℹ️ Sobre o Modelo")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.header("📋 Especificações Técnicas")
        
        st.subheader("🎯 Arquitetura do Modelo")
        st.markdown("""
        - **Base Model:** MobileNetV2
        - **Input Shape:** 224x224x3
        - **Output:** Sigmoid (Classificação Binária)
        - **Parâmetros:** 2.3 milhões
        - **Camadas:** 155
        """)
        
        st.subheader("📊 Métricas de Performance")
        st.markdown("""
        - **Acurácia:** 92.3%
        - **Precisão:** 91.8%
        - **Recall:** 89.5%
        - **F1-Score:** 90.6%
        - **AUC-ROC:** 0.96
        """)
        
        st.subheader("🎓 Treinamento")
        st.markdown("""
        - **Dataset:** Chest X-Ray Images (Pneumonia)
        - **Amostras:** 5,856 imagens
        - **Split:** 80% treino, 10% validação, 10% teste
        - **Épocas:** 50
        - **Batch Size:** 32
        - **Optimizer:** Adam
        - **Loss Function:** Binary Crossentropy
        """)

    with col2:
        st.header("🛠️ Stack Tecnológico")
        
        tech_stack = {
            "Framework": "TensorFlow 2.0",
            "Backend": "Streamlit",
            "Processamento": "NumPy, PIL",
            "Visualização": "Plotly, Pandas",
            "Interface": "CSS Personalizado"
        }
        
        for tech, desc in tech_stack.items():
            st.markdown(f"**{tech}:** {desc}")

    st.markdown("---")

    st.header("📝 Considerações Éticas")
    st.warning("""
    **⚠️ Importante:**
    - Este sistema é uma ferramenta de auxílio diagnóstico
    - Não substitui a avaliação de um médico especialista
    - Resultados devem ser interpretados por profissionais qualificados
    - Falsos positivos e negativos podem ocorrer
    - Sempre realize exames complementares quando necessário
    """)

    st.info("""
    **💡 Uso Recomendado:**
    - Triagem inicial de radiografias
    - Segundo parecer em diagnósticos
    - Ambiente educacional e de pesquisa
    - Monitoramento de tratamento
    """)

def main():
    # Sidebar e navegação
    page = sidebar()
    
    # Renderiza a página selecionada
    if page == "🏠 Classificação":
        pagina_classificacao()
    else:  # "ℹ️ Sobre o Modelo"
        pagina_sobre()
    
    # Footer principal (apenas na página principal)
    if page == "🏠 Classificação":
        st.markdown("---")
        st.markdown("""
        <div class='main-footer'>
            <p><strong>PneumoScan</strong> - Sistema de Auxílio ao Diagnóstico por Imagem</p>
            <p style='font-size: 0.8em; color: #666;'>
            <em>Este sistema é destinado exclusivamente para auxílio diagnóstico e não substitui a avaliação clínica profissional.</em>
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()