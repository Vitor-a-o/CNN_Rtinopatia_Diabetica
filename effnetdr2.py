import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, cohen_kappa_score, classification_report, ConfusionMatrixDisplay
from sklearn.utils import class_weight

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
    
    train_df, val_df, test_df = None, None, None
    train_gen, val_gen, test_gen = None, None, None
    model, base_model = None, None
    
    # Carrega e divide os dados
    try:
        train_df, val_df, test_df = load_and_split_data(CSV_PATH)
    except FileNotFoundError as e:
        print(f"[ERRO] Arquivo CSV não encontrado: {e}")
        return
    except Exception as e:
        print(f"[ERRO] Ocorreu um erro ao carregar e dividir os dados: {e}")
        return
    
    # Cria os geradores de dados
    try:
        train_gen, val_gen, test_gen = create_generators(train_df, val_df, test_df, IMG_PATH, TARGET_SIZE, BATCH_SIZE)
    except Exception as e:
        print(f"[ERRO] Ocorreu um erro ao criar os geradores de dados: {e}")
        return
    
    
    # Constrói e compila o modelo
    try:
        num_classes = len(train_gen.class_indices)
        model, base_model = build_model(input_shape=TARGET_SIZE + (3,), num_classes=num_classes)
        compile_and_load(model, CHECKPOINT_PATH, lr=1e-4)
    except Exception as e:
        print(f"[ERRO] Ocorreu um erro ao construir e compilar o modelo: {e}")
        return
        
    # Callbacks para o treinamento inicial
    checkpoint_callback = ModelCheckpoint(filepath=CHECKPOINT_PATH,
                                          save_best_only=True,
                                          monitor='val_loss',
                                          mode='min',
                                          save_weights_only=True)
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    # Aplica pesos de classe para lidar com o desbalanceamento
    try:
        class_weights = class_weight.compute_class_weight(
                                                        class_weight='balanced',
                                                        classes= np.unique(train_gen.classes),
                                                        y=train_gen.classes)
        
        class_weights_dict = dict(enumerate(class_weights))
    except Exception as e:
        print(f"[ERRO] Ocorreu um erro ao calcular os pesos de classe: {e}")
        class_weights_dict = None
    
    kappa_history = []
    
    # Treinamento inicial
    try:
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            class_weight=class_weights_dict,
            epochs=EPOCHS,
            callbacks=[checkpoint_callback, early_stopping_callback]
        )
    except Exception as e:
        print(f"[ERRO] Ocorreu um erro durante o treinamento: {e}")
        return
    
    # Avalia o conjunto de teste
    try:
        test_loss, test_accuracy = model.evaluate(test_gen)
        print(f"Test Loss: {test_loss} | Test Accuracy: {test_accuracy}")
    except Exception as e:
        print(f"[ERRO] Ocorreu um erro ao avaliar o conjunto de teste: {e}")
    
    # Previsões e métricas
    y_pred, y_true = None, None
    
    try:
        test_gen.reset()
        Y_pred = model.predict(test_gen, steps=math.ceil(test_gen.n / test_gen.batch_size))
        y_pred = np.argmax(Y_pred, axis=1)
        y_true = test_gen.classes
        kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        kappa_history.append({"stage": "initial_training", "kappa": kappa})
        print(f"Cohen's Kappa: {kappa}")
    except Exception as e:
        print(f"[ERRO] Ocorreu um erro ao fazer previsões: {e}")
    
    # Salva os gráficos de matriz de confusão e métricas de treinamento
    try:
        if y_true is not None and y_pred is not None:
            plot_confusion_matrix(y_true, y_pred, f'{OUTPUT_DIR}/confusion_matrix.png')
        plot_metrics(history, f'{OUTPUT_DIR}/training_metrics.png', title_prefix='Inicial')
    except Exception as e:
        print(f"[ERRO] Ocorreu um erro ao salvar os gráficos: {e}")
    
    # Fine-tuning: descongela a base e treina novamente com lr menor
    try:
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
    except Exception as e:
        print(f"[ERRO] Ocorreu um erro durante o fine-tuning: {e}")
        return
    
    # Combina os históricos para plotar métricas completas
    total_epochs = EPOCHS + FINE_TUNE_EPOCHS
    
    try:
        acc = history.history['accuracy'] + history_fine.history['accuracy']
        val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
        loss = history.history['loss'] + history_fine.history['loss']
        val_loss = history.history['val_loss'] + history_fine.history['val_loss']
    except Exception as e:
        print(f"[ERRO] Ocorreu um erro ao combinar os históricos: {e}")
        acc, val_acc, loss, val_loss = [], [], [], []
    
    # Avaliação após o fine-tuning
    y_pred_fine = None
    
    try:
        test_loss_fine, test_accuracy_fine = model.evaluate(test_gen)
        print(f"Test Loss after Fine-Tuning: {test_loss_fine} | Test Accuracy after Fine-Tuning: {test_accuracy_fine}")
        test_gen.reset()
        Y_pred_fine = model.predict(test_gen, steps=math.ceil(test_gen.n / test_gen.batch_size))
        y_pred_fine = np.argmax(Y_pred_fine, axis=1)
    except Exception as e:
        print(f"[ERRO] Ocorreu um erro ao avaliar o conjunto de teste após o fine-tuning: {e}")
    
    # Métricas após o fine-tuning
    try:
        if y_true is not None and y_pred_fine is not None:
            kappa_fine = cohen_kappa_score(y_true, y_pred_fine, weights='quadratic')
            kappa_history.append({"stage": "fine_tuning", "kappa": kappa_fine})
            print(f"Cohen's Kappa after Fine-Tuning: {kappa_fine}")
            plot_confusion_matrix(y_true, y_pred_fine, f'{OUTPUT_DIR}/confusion_matrix_fine_tuning.png')
    except Exception as e:
        print(f"[ERRO] Ocorreu um erro ao calcular as métricas após o fine-tuning: {e}")
    
    
    # Plota os gráficos combinados de loss e acurácia
    try:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(len(loss)), loss, label='Train Loss')
        plt.plot(range(len(acc)), acc, label='Train Accuracy')
        plt.title('Treinamento Completo')
        plt.xlabel('Épocas')
        plt.ylabel('Loss / Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(range(len(val_loss)), val_loss, label='Validation Loss')
        plt.plot(range(len(val_acc)), val_acc, label='Validation Accuracy')
        plt.title('Validação Completa')
        plt.xlabel('Épocas')
        plt.ylabel('Loss / Accuracy')
        plt.legend()
        
        plt.savefig(f'{OUTPUT_DIR}/fine_tuning_metrics.png')
        plt.close()
    except Exception as e:
        print(f"[ERRO] Ocorreu um erro ao salvar os gráficos de métricas após o fine-tuning: {e}")
    
    # Salva o histórico de Kappa
    try:
        df_kappa = pd.DataFrame(kappa_history)
        df_kappa.to_csv(f'{OUTPUT_DIR}/kappa_history.csv', index=False)
        print("Histórico de Kappa salvo com sucesso.")
        print("--------------------------------------")
        print("Histórico de Kappa:")
        print(df_kappa)
        print("--------------------------------------")
    except Exception as e:
        print(f"[ERRO] Ocorreu um erro ao salvar o histórico de Kappa: {e}")

if __name__ == '__main__':
    main()
