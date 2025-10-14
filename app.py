import streamlit as st
import tensorflow as tf
import io
from PIL import Image
import numpy as np #converter a imagem para um formato que o TensorFlow consiga entender
import pandas as pd
import plotly.express as px


@st.cache_resource #armazena o modelo em cache evitando downloads repetidos
def carrega_modelo():
    
    interpreter = tf.lite.Interpreter(model_path='cnn_models/modelo_quantizado16bits.tflite')
    interpreter.allocate_tensors()
    return interpreter

def carrega_imagem():
    uploaded_file = st.file_uploader(
        '🩺 Envie uma radiografia de tórax (formato PNG, JPG ou JPEG):',
        type=['png', 'jpg', 'jpeg']
    )

    if uploaded_file is not None:
        image_data = uploaded_file.read()
        image = Image.open(io.BytesIO(image_data))

        # Converte para RGB (3 canais) — necessário!
        image = image.convert("RGB")

        st.image(image, caption='Imagem carregada com sucesso!', use_container_width=True)

        # Redimensiona para o tamanho que o modelo espera (256x256)
        image = image.resize((256, 256))

        # Converte para array NumPy e normaliza
        image = np.array(image, dtype=np.float32)
        image = image / 255.0

        # Adiciona dimensão do batch: [1, 256, 256, 3]
        image = np.expand_dims(image, axis=0)

        st.write("🧾 Formato final da imagem:", image.shape)  # só pra conferência
        return image

    return None

def previsao(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Envia imagem ao modelo
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    # Obtém a saída (probabilidade da classe "Pneumonia")
    output_data = interpreter.get_tensor(output_details[0]['index'])[0][0]  # único valor

    # Calcula probabilidades para ambas as classes
    prob_pneumonia = float(output_data)
    prob_normal = 1 - prob_pneumonia

    # Organiza os dados
    classes = ['Normal', 'Pneumonia']
    probabilidades = [prob_normal, prob_pneumonia]

    df = pd.DataFrame({
        'Classe': classes,
        'Probabilidade (%)': 100 * np.array(probabilidades)
    })

    # Gráfico
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

    # Resultado final
    classe_predita = classes[np.argmax(probabilidades)]
    probabilidade = np.max(probabilidades) * 100
    st.success(f'📊 Resultado: **{classe_predita}** com {probabilidade:.2f}% de confiança.')


def main():
    
    st.set_page_config(page_title="Classificação de Pneumonia", page_icon="🫁", layout="centered")
    st.title('🫁 Classificação de Radiografias de Tórax')
    st.write('Este aplicativo usa um modelo de **Deep Learning (CNN)** para identificar se a radiografia é **Normal** ou possui **Pneumonia**.')

    #carrega modelo
    interpreter = carrega_modelo()
    
    #carrega imagem
    image = carrega_imagem()

    #classifica
    if image is not None:
        previsao(interpreter,image)

if __name__=="__main__":
    main()