import streamlit as st
import tensorflow as tf
import io
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px

def carrega_imagem():
    uploaded_file = st.file_uploader(
        '🩺 Envie uma radiografia de tórax (PNG, JPG ou JPEG):',
        type=['png', 'jpg', 'jpeg']
    )

    if uploaded_file is not None:
        image_data = uploaded_file.read()
        image = Image.open(io.BytesIO(image_data))

        # Converte para RGB (3 canais)
        image = image.convert("RGB")

        st.image(image, caption='Imagem carregada com sucesso!', use_container_width=True)

        # Redimensiona para o tamanho esperado pelo modelo
        image = image.resize((224, 224))

        # Converte para array NumPy e normaliza
        image_array = np.array(image, dtype=np.float32) / 255.0

        # Adiciona dimensão do batch: [1, 224, 224, 3]
        image_array = np.expand_dims(image_array, axis=0)

        st.write("🧾 Formato final da imagem:", image_array.shape)  # debug
        return image_array

    return None

# --- 🔹 Função para carregar modelo ---
@st.cache_resource
def carrega_modelo_h5():
    model = tf.keras.models.load_model('cnn_models/mobilenetv2_FASE1.h5')
    return model

# --- 🔹 Função de previsão ---
def previsao_h5(model, image):
    # Faz a previsão (sigmoid)
    pred = model.predict(image)  # shape: (1, 1)
    prob_pneumonia = float(pred[0][0])
    prob_normal = 1 - prob_pneumonia

    # Cria DataFrame para visualização
    classes = ['Normal', 'Pneumonia']
    probabilidades = [prob_normal, prob_pneumonia]
    
    df = pd.DataFrame({
        'Classe': classes,
        'Probabilidade (%)': 100 * np.array(probabilidades)
    })

    # Gráfico de barras horizontal
    fig = px.bar(
        df,
        y='Classe',
        x='Probabilidade (%)',
        orientation='h',
        text='Probabilidade (%)',
        color='Classe',
        title='Resultado da Classificação'
    )
    st.plotly_chart(fig, use_container_width=True)

    classe_predita = classes[np.argmax(probabilidades)]
    probabilidade = np.max(probabilidades) * 100
    st.success(f'📊 Resultado: **{classe_predita}** com {probabilidade:.2f}% de confiança.')

def main():
    st.set_page_config(page_title="Classificação de Pneumonia", page_icon="🫁", layout="centered")
    st.title('🫁 Classificação de Radiografias de Tórax')
    st.write('Este aplicativo usa um modelo de **Deep Learning (CNN)** para identificar se a radiografia é **Normal** ou apresenta **Pneumonia**.')

    image = carrega_imagem()

    if image is not None:
        model = carrega_modelo_h5()
        previsao_h5(model, image)

if __name__ == "__main__":
    main()
