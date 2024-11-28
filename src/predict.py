import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from src.preprocess import detect_and_preprocess

def predict_image(image_path, model_path, class_map):
    # Carregar o modelo treinado
    model = load_model(model_path)
    
    # Detectar e pré-processar as moedas na imagem
    coins = detect_and_preprocess(image_path)

    total_value = 0
    preds = []
    for coin in coins:
        # Expandir a dimensão para batch único
        coin = np.expand_dims(coin, axis=0)
        
        # Fazer a predição
        prediction = model.predict(coin)
        preds.append(prediction)
        
        # Determinar a classe predita e o valor
        class_idx = np.argmax(prediction)
        value = class_map[class_idx]
        total_value += value
    
    # Visualizar as moedas e predições
    n = len(coins)  # Número de moedas detectadas
    plt.figure(figsize=(15, 5))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(coins[i])
        plt.title(f'Predito: {class_map[np.argmax(preds[i])]} R$')
    plt.show()

    return total_value
