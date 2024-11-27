import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def detect_and_preprocess(image_path, target_size=(224, 224)):

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Falha ao carregar a imagem: {image_path}")

    # Converter para escala de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplicar suavização
    blurred = cv2.GaussianBlur(gray, (17, 17), 0)

    # Detectar bordas
    edged = cv2.Canny(blurred, 25, 125)

    #exit()

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    coins = []
    for contour in contours:
        if cv2.contourArea(contour) < 100:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        coin = image[y:y+h, x:x+w]
        coin = cv2.resize(coin, target_size)
        coin = coin / 255.0  # Normalizar
        coins.append(coin)
    
    return np.array(coins)


def create_generators(dataset_path, img_size=(224, 224), batch_size=32):
    datagen = ImageDataGenerator(
        rotation_range=90,       
        shear_range=0.2,              
        horizontal_flip=True,  
        brightness_range=[0.5, 1.5],
        validation_split=0.2,
        fill_mode='nearest'
    )
    train_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'  # 80% para treino
    )

    val_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'  # 20% para validação
    )
    return train_generator, val_generator
