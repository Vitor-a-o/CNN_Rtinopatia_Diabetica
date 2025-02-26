import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
import sklearn

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, cohen_kappa_score, classification_report, ConfusionMatrixDisplay
from sklearn.utils import class_weight

print(sklearn.__version__)

# Configurações e caminhos
IMG_PATH = "/app/resized_train"
CSV_PATH = "/app/CNN_Rtinopatia_Diabetica/trainLabels3.csv"  # Certifique-se de ajustar o caminho se necessário
CHECKPOINT_PATH = '/app/CNN_Rtinopatia_Diabetica/model_checkpoint.weights.h5'
SAMPLE_FRAC = 1
BATCH_SIZE = 16
TARGET_SIZE = (224, 224)
EPOCHS = 5
FINE_TUNE_EPOCHS = 5
OUTPUT_DIR = 'img'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_split_data(csv_path, sample_frac=SAMPLE_FRAC, random_state=43):
    
    """Carrega o CSV, renomeia colunas, converte tipos e divide os dados."""
    
    df = pd.read_csv(csv_path)
    df.rename(columns={'image': 'image_id', 'level': 'dr_grade'}, inplace=True)
    df['dr_grade'] = df['dr_grade'].astype(str)
    
    df_sampled = df.sample(frac=sample_frac, random_state=random_state)
    train_df, test_df = train_test_split(df_sampled, test_size=0.15, random_state=random_state)
    train_df, val_df = train_test_split(train_df, test_size=0.15/(1-0.15), random_state=random_state)
    
    print("Train:", train_df.shape, "Validation:", val_df.shape, "Test:", test_df.shape)
    return train_df, val_df, test_df

def create_generators(train_df, val_df, test_df, img_path, target_size, batch_size):
    
    """Cria os geradores de dados com normalização."""
    
    datagen = ImageDataGenerator(rescale=1./255)
    
    train_gen = datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=img_path,
        x_col="image_id",
        y_col="dr_grade",
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical"
    )
    
    val_gen = datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=img_path,
        x_col="image_id",
        y_col="dr_grade",
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical"
    )
    
    test_gen = datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=img_path,
        x_col="image_id",
        y_col="dr_grade",
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False  # Mantém a ordem para previsões
    )
    
    # Exibe as dimensões de uma amostra do batch
    images, labels = next(iter(train_gen))
    print(f"Dimensões das imagens: {images.shape}, Dimensões dos rótulos: {labels.shape}")
    
    return train_gen, val_gen, test_gen

def build_model(input_shape, num_classes):
    
    """Constrói o modelo com EfficientNetB3 como base."""
    
    base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=input_shape)
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    
    # Congela as camadas da base para preservar o aprendizado prévio
    for layer in base_model.layers:
        layer.trainable = False
        
    return model, base_model

def focal_loss(gamma=2.0, alpha=0.25):
    
    """Função de perda focal para lidar com classes desbalanceadas."""
    
    def focal_loss_fixed(y_true, y_pred):
        eps = 1e-7
        y_pred = K.clip(y_pred, eps, 1-eps)
        
        cross_entropy = -y_true * K.log(y_pred)
        loss = alpha * K.pow(1-y_pred, gamma) * cross_entropy
        
        return K.sum(loss, axis=1)
    
    return focal_loss_fixed

def compile_and_load(model, checkpoint_path, lr=1e-4):
    
    """Compila o modelo e tenta carregar um checkpoint se existir."""
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss=focal_loss(),
                  metrics=['accuracy'])
    
    if os.path.exists(checkpoint_path):
        try:
            model.load_weights(checkpoint_path)
            print("Checkpoint carregado com sucesso.")
        except Exception as e:
            print(f"Erro ao carregar checkpoint: {e}")
    else:
        print("Checkpoint não encontrado. Iniciando o treinamento do zero.")

def plot_metrics(history, filename, title_prefix=''):
    """Plota e salva os gráficos de loss e acurácia."""
    plt.figure(figsize=(12, 5))
    
    # Treinamento
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.title(f'{title_prefix} Treinamento')
    plt.xlabel('Épocas')
    plt.ylabel('Loss / Accuracy')
    plt.legend()
    
    # Validação
    plt.subplot(1, 2, 2)
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{title_prefix} Validação')
    plt.xlabel('Épocas')
    plt.ylabel('Loss / Accuracy')
    plt.legend()
    
    plt.savefig(filename)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, filename):
    """Plota e salva a matriz de confusão."""
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.savefig(filename)
    plt.close()

def main():
    # Carrega e divide os dados
    train_df, val_df, test_df = load_and_split_data(CSV_PATH)
    
    # Cria os geradores de dados
    train_gen, val_gen, test_gen = create_generators(train_df, val_df, test_df, IMG_PATH, TARGET_SIZE, BATCH_SIZE)
    
    # Define o número de classes e constrói o modelo
    num_classes = len(train_gen.class_indices)
    model, base_model = build_model(input_shape=TARGET_SIZE + (3,), num_classes=num_classes)
    
    # Compila o modelo e tenta carregar um checkpoint
    compile_and_load(model, CHECKPOINT_PATH, lr=1e-4)
    
    # Callbacks para o treinamento inicial
    checkpoint_callback = ModelCheckpoint(filepath=CHECKPOINT_PATH,
                                          save_best_only=True,
                                          monitor='val_loss',
                                          mode='min',
                                          save_weights_only=True)
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    # Aplica pesos de classe para lidar com o desbalanceamento
    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(train_gen.classes),
                                                      train_gen.classes,)
    
    class_weights_dict = dict(enumerate(class_weights))
    
    # Treinamento inicial
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        class_weight=class_weights_dict,
        epochs=EPOCHS,
        callbacks=[checkpoint_callback, early_stopping_callback]
    )
    
    # Avalia o conjunto de teste
    test_loss, test_accuracy = model.evaluate(test_gen)
    print(f"Test Loss: {test_loss} | Test Accuracy: {test_accuracy}")
    
    # Previsões e métricas
    test_gen.reset()
    Y_pred = model.predict(test_gen, steps=math.ceil(test_gen.n / test_gen.batch_size))
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = test_gen.classes
    kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    print(f"Cohen's Kappa: {kappa}")
    print(history.history)
    
    # Salva os gráficos de matriz de confusão e métricas de treinamento
    plot_confusion_matrix(y_true, y_pred, f'{OUTPUT_DIR}/confusion_matrix.png')
    plot_metrics(history, f'{OUTPUT_DIR}/training_metrics.png', title_prefix='Inicial')
    
    # Fine-tuning: descongela a base e treina novamente com lr menor
    for layer in base_model.layers:
        layer.trainable = True
        
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss=focal_loss(),
                  metrics=['accuracy'])
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-7)
    
    history_fine = model.fit(
        train_gen,
        validation_data=val_gen,
        class_weight=class_weights_dict,
        epochs=EPOCHS + FINE_TUNE_EPOCHS,
        initial_epoch=EPOCHS,
        callbacks=[checkpoint_callback, reduce_lr]
    )
    
    # Combina os históricos para plotar métricas completas
    total_epochs = EPOCHS + FINE_TUNE_EPOCHS
    acc = history.history['accuracy'] + history_fine.history['accuracy']
    val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
    loss = history.history['loss'] + history_fine.history['loss']
    val_loss = history.history['val_loss'] + history_fine.history['val_loss']
    
    # Avaliação após fine-tuning
    test_loss_fine, test_accuracy_fine = model.evaluate(test_gen)
    print(f"Test Loss after Fine-Tuning: {test_loss_fine} | Test Accuracy after Fine-Tuning: {test_accuracy_fine}")
    
    test_gen.reset()
    Y_pred_fine = model.predict(test_gen, steps=math.ceil(test_gen.n / test_gen.batch_size))
    y_pred_fine = np.argmax(Y_pred_fine, axis=1)
    kappa_fine = cohen_kappa_score(y_true, y_pred_fine, weights='quadratic')
    print(f"Cohen's Kappa after Fine-Tuning: {kappa_fine}")
    print(history_fine.history)
    
    plot_confusion_matrix(y_true, y_pred_fine, f'{OUTPUT_DIR}/confusion_matrix_fine_tuning.png')
    
    # Plota os gráficos combinados de loss e acurácia
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(total_epochs), loss, label='Train Loss')
    plt.plot(range(total_epochs), acc, label='Train Accuracy')
    plt.title('Treinamento Completo')
    plt.xlabel('Épocas')
    plt.ylabel('Loss / Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(total_epochs), val_loss, label='Validation Loss')
    plt.plot(range(total_epochs), val_acc, label='Validation Accuracy')
    plt.title('Validação Completa')
    plt.xlabel('Épocas')
    plt.ylabel('Loss / Accuracy')
    plt.legend()
    
    plt.savefig(f'{OUTPUT_DIR}/fine_tuning_metrics.png')
    plt.close()

if __name__ == '__main__':
    main()
