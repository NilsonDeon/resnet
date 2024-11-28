import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def detect_and_preprocess(image_path, target_size=(224, 224), debug=False):
    """
    Detecta moedas circulares em uma imagem usando HoughCircles e retorna as imagens recortadas e normalizadas.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

    # Carregar a imagem
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Falha ao carregar a imagem: {image_path}")

    # Converter para escala de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplicar suavização para melhorar a detecção de círculos
    blurred = cv2.medianBlur(gray, 5)

    # Detectar círculos usando HoughCircles
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,          # Resolução do acumulador (ajustável)
        minDist=30,      # Distância mínima entre os centros dos círculos
        param1=50,       # Limite superior para Canny
        param2=30,       # Limite de acumulação para centro de círculos
        minRadius=10,    # Raio mínimo do círculo
        maxRadius=100    # Raio máximo do círculo
    )

    coins = []
    if circles is not None:
        # Converter para inteiros
        circles = np.round(circles[0, :]).astype("int")

        for (x, y, r) in circles:
            # Recortar a região circular da imagem
            x1, y1, x2, y2 = x - r, y - r, x + r, y + r
            coin = image[y1:y2, x1:x2]

            if coin.size == 0:  # Ignorar regiões inválidas
                continue

            # Redimensionar e normalizar
            coin = cv2.resize(coin, target_size)
            coin = coin / 255.0
            coins.append(coin)

            # Opcional: Desenhar círculo detectado para depuração
            if debug:
                debug_image = image.copy()

                # Obter dimensões da imagem
                height, width = image.shape[:2]
                
                # Calcular escala para ajustar a imagem
                scale = min(800 / width, 800 / height, 1.0)
                new_width = int(width * scale)
                new_height = int(height * scale)

                # Redimensionar a imagem
                resized_image = cv2.resize(debug_image, (new_width, new_height))

                cv2.circle(resized_image, (x, y), r, (0, 255, 0), 2)
                cv2.rectangle(resized_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.imshow("Detected Circle", resized_image)
                cv2.namedWindow("Detected Circle", cv2.WINDOW_NORMAL)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    return np.array(coins)


def create_generators(dataset_path, img_size=(224, 224), batch_size=32):
    """
    Cria geradores de dados para treinamento e validação com aumentação de dados.
    """
    datagen = ImageDataGenerator(
        rotation_range=90,
        shear_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.5, 1.5],
        validation_split=0.2,
        fill_mode='nearest',
    )
    train_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    val_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    return train_generator, val_generator
