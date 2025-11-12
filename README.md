Classificação de Pneumonia com Inteligência Artificial

Projeto teve como objetivo cientifico e prático e aplicado para Trabalho de conclusão de curso em Sistemas de Informação
Este projeto tem como objetivo desenvolver um sistema capaz de classificar imagens de raio-X de tórax para identificar casos de pneumonia utilizando redes neurais convolucionais (CNNs).

A aplicação utiliza Python, TensorFlow, Keras e NumPy para o treinamento e inferência do modelo, além de uma interface web interativa desenvolvida em Streamlit, onde o usuário pode realizar o upload de imagens e visualizar os resultados de classificação de forma simples e intuitiva.

Especificações

Arquitetura: MobileNetV2

Dataset: Chest X-Ray Images

Classes: Normal vs Pneumonia

Acurácia: 91% (validação)

Especialidade: Radiologia Torácica

    Matriz de confusão
    [[202  32]
    [ 24 366]]

Métricas:

                  precision    recall  f1-score   support
    NORMAL             0.89      0.86      0.88       234
    PNEUMONIA          0.92      0.94      0.93       390

    accuracy                               0.91       624
    macro avg          0.91      0.90      0.90       624
    weighted avg       0.91      0.91      0.91       624

Em 634 imagens o modelo acertou uma taxa de 86,70% classificando
Estatísticas por classe:

    PNEUMONIA: 335/390 corretas (85.90%)
    NORMAL: 206/234 corretas (88.03%)

#rodar projeto: streamlit run app.py
