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
        st.warning("CSS externo não encontrado.")
load_css()

#SIDE BAR
def sidebar():
    with st.sidebar:
        st.title("PneumoScan")
        
        page = st.radio(
            "Navegação",
            ["Classificação", "Sobre o Modelo"],
            index=0
        )
        
        st.markdown("---")
        
        # Infos sidebar
        st.subheader("Especificações")
        st.markdown("""
        **Arquitetura:** MobileNetV2  
        **Dataset:** Chest X-Ray Images  
        **Classes:** Normal vs Pneumonia  
        **Acurácia:** 86,70% (validação)  
        **Especialidade:** Radiologia Torácica
        """)
        
        st.markdown("---")
        
        return page

#Função para carregar imagem
def carrega_imagem():
    
    uploaded_file = st.file_uploader(
        '**Envie uma radiografia de tórax:**',
        type=['png', 'jpg', 'jpeg'],
        help="Formatos suportados: PNG, JPG, JPEG"
    )

    if uploaded_file is not None:
        try:
            image_data = uploaded_file.read()
            image = Image.open(io.BytesIO(image_data))
            image = image.convert("RGB")

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

#Função para carregar modelo
@st.cache_resource
def carrega_modelo_h5():
    try:
        model = tf.keras.models.load_model('cnn_models/mobilenetv2_FASE1.h5')
        st.sidebar.success("Modelo carregado!")
        return model
    except Exception as e:
        st.sidebar.error(f"Erro ao carregar modelo: {str(e)}")
        return None

#Função de previsão
def previsao_h5(_model, image, filename):
    # Container para resultados
    result_container = st.container()
    
    with result_container:
        st.subheader("Resultados da Análise")
        
        # Faz a previsão
        with st.spinner('Analisando radiografia...'):
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

        # Infos Técnicas
        with st.expander("📋 Detalhes Técnicos"):
            st.write(f"**Arquivo analisado:** {filename}")
            st.write(f"**Dimensões da imagem:** 224x224 pixels")
            st.write(f"**Modelo utilizado:** MobileNetV2")
            st.write(f"**Probabilidade Pneumonia:** {prob_pneumonia:.4f}")
            st.write(f"**Probabilidade Normal:** {prob_normal:.4f}")

#Página de Classificação
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

#Página Sobre
def pagina_sobre():
    st.title("Sobre o Modelo")
    st.markdown("---")

    # --- Introdução ---
    with st.container():
        st.markdown("""
        <div class='intro-section'>
        <h3>Desenvolvimento e Finalidade</h3>
        <p>O <strong>PneumoScan</strong> foi desenvolvido por <strong>Kauã Christian</strong> como parte de um 
        Trabalho de Conclusão de Curso (TCC), com o objetivo de aplicar técnicas de 
        Inteligência Artificial e aprendizado profundo no auxílio ao diagnóstico médico
        de pneumonia através da análise automatizada de radiografias de tórax.</p>

        <p>O sistema busca apoiar profissionais da saúde em processos de triagem e análise inicial de exames, 
        fornecendo previsões com base em redes neurais convolucionais treinadas em imagens reais.</p>

        <p>Apesar de apresentar resultados expressivos, este projeto possui caráter <strong>educacional e experimental</strong>, 
        e não substitui a avaliação médica profissional. As previsões geradas devem ser interpretadas com responsabilidade 
        e sempre em conjunto com parecer clínico.</p>

        <h4>Uso Recomendado</h4>
        <ul style='text-align: left; display: inline-block;'>
            <li>Triagem inicial de radiografias torácicas</li>
            <li>Ambientes de ensino e pesquisa</li>
            <li>Estudos sobre aplicações de IA na saúde</li>
            <li>Monitoramento de progresso em tratamentos</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    #detalhes técnicos
    col1, col2 = st.columns(2)

    with col1:
        st.header("Métricas de Performance")
        st.markdown("""
        - **Acurácia:** 86,70%  
        - **Precisão:** NORMAL: 89% | PNEUMONIA: 92%  
        - **Recall:** NORMAL: 86% | PNEUMONIA: 94%  
        - **F1-Score:** NORMAL: 88% | PNEUMONIA: 93%  
        - **AUC-ROC:** 0.9665  
        """)


    with col2:
        st.header("Treinamento")
        st.markdown("""
        - **Dataset:** Chest X-Ray Images (Pneumonia)  
        - **Total de Imagens:** 5.856  
        - **Divisão:** 80% treino, 10% teste, 5% validação  
        - **Épocas:** 50 + (EarlyStoping) 
        - **Batch Size:** 32  
        - **Optimizer:** Adam  
        - **Função de Perda:** Binary Crossentropy  
        """)

    st.markdown("---")

    #Encerramento
    with st.container():
        st.markdown("""
        <div class='intro-section'>
        <h3>Conclusões e Possíveis Melhorias</h3>
        <p>O projeto demonstrou um desempenho satisfatório, alcançando boas métricas de predição e 
        validando a eficiência da arquitetura MobileNetV2 em aplicações médicas de visão computacional.</p>

        <p>Como perspectivas futuras, o sistema poderá ser aprimorado com:</p>
        <ul style='text-align: left; display: inline-block;'>
            <li>Expansão do dataset com mais imagens de diferentes origens;</li>
            <li>Incremento da acurácia por meio de técnicas de fine-tuning e aumento de dados;</li>
            <li>Treinamento para detecção de outras doenças pulmonares, como COVID-19, tuberculose e enfisema;</li>
            <li>Otimização para execução em dispositivos móveis e ambientes clínicos reais.</li>
        </ul>

        <p>Este estudo reforça o potencial da Inteligência Artificial como ferramenta de apoio 
        no diagnóstico por imagem, contribuindo para o avanço da saúde digital e da pesquisa aplicada.</p>
        </div>
        """, unsafe_allow_html=True)

#chamando funções das paginas, sidebar e footer
def main():
    page = sidebar()
    
    # Renderiza a página selecionada
    if page == "Classificação":
        pagina_classificacao()
    else:  # "Sobre o Modelo"
        pagina_sobre()
    
    # Footer principal (apenas na página principal)
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