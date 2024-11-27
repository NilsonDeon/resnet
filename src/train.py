from src.preprocess import create_generators
from src.model import build_model

def train_model(dataset_path, output_model_path):
    train_gen, val_gen = create_generators(dataset_path)
    
    # Verificar o número de amostras
    print(f"Número de imagens no treino: {train_gen.samples}")
    print(f"Número de imagens na validação: {val_gen.samples}")
    
    # Calcular steps
    steps_per_epoch = train_gen.samples // train_gen.batch_size
    validation_steps = val_gen.samples // val_gen.batch_size

    # Ajustar para no mínimo 1 step
    steps_per_epoch = max(steps_per_epoch, 1)
    validation_steps = max(validation_steps, 1)
    
    # Criar o modelo
    model = build_model(num_classes=train_gen.num_classes)
    
    # Treinar o modelo
    model.fit(
        train_gen,
        epochs=20,
        validation_data=val_gen,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )
    
    # Salvar o modelo
    model.save(output_model_path)
    print(f"Modelo salvo em: {output_model_path}")
