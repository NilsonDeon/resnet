import os
from src.preprocess import create_generators
from src.model import build_model
from src.predict import predict_image

if __name__ == "__main__":
    dataset_path = "data/train"
    model_path = "models/coin_classifier_resnet50_v2.keras"

    """ # Criar geradores para treino e validação
    train_generator, validation_generator = create_generators(dataset_path)

    # Criar o modelo
    model = build_model(num_classes=train_generator.num_classes)

    # Calcular steps
    steps_per_epoch = (train_generator.samples // train_generator.batch_size) - 1
    validation_steps = (validation_generator.samples // validation_generator.batch_size) - 1

    # Ajustar para no mínimo 1 step
    steps_per_epoch = max(1, steps_per_epoch)
    validation_steps = max(1, validation_steps)

    # Treinar o modelo
    model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=10,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )
 
    # Salvar o modelo
    model.save(model_path)
    print(f"Modelo salvo em: {model_path}")  """

    # Mapeamento de classes para valores
    class_map = {
        0: 0.05,
        1: 0.10,
        2: 0.25,
        3: 0.50,
        4: 1.00
    }

    # Diretório com as imagens para teste
    test_dir = "data/resto/umas"

    # Iterar sobre todas as imagens na pasta
    for filename in os.listdir(test_dir):
        image_path = os.path.join(test_dir, filename)
        if not os.path.isfile(image_path):
            continue  # Ignorar se não for um arquivo

        try:
            # Fazer a predição
            total_value = predict_image(image_path, model_path, class_map)
            print(f"Imagem: {filename} - Valor total detectado: R$ {total_value:.2f}")
        except Exception as e:
            print(f"Erro ao processar {filename}: {e}") 
