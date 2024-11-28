from src.preprocess import create_generators
from src.model import build_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping

def train_model(dataset_path, output_model_path):
    train_gen, val_gen = create_generators(dataset_path, img_size=(128, 128), batch_size=16)

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
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=2,
        restore_best_weights=True
    )
    
    # Salvar o modelo
    model.save(output_model_path)
    print(f"Modelo salvo em: {output_model_path}")

    # Prever no conjunto de validação
    print("\nAvaliando no conjunto de validação...")
    val_gen.reset()
    y_true = val_gen.classes
    y_pred = model.predict(val_gen, steps=validation_steps)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Exibir relatório de classificação
    class_labels = list(val_gen.class_indices.keys())
    print("\nRelatório de Classificação:")
    print(classification_report(y_true, y_pred_classes, target_names=class_labels))

    # Exibir matriz de confusão
    print("\nMatriz de Confusão:")
    print(confusion_matrix(y_true, y_pred_classes))
