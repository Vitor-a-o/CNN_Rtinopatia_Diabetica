# -*- coding: utf-8 -*-
"""EffNetdr.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1iRngt5yokyrNFjcEZOGvtDqXcDW5S9Fb
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix, cohen_kappa_score, classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#from google.colab import drive
#drive.mount('/content/drive')

img_path = "C:\\Users\\Vitor\\Desktop\\resized_train\\resized_train"
file_path = ""

#file_path = "/home/vitoroliveira/CNN_Rtinopatia_Diabetica/"
#img_path = "/home/vitoroliveira/resized_train/"
#csv_name = "trainLabels3.csv"

# Carregar o CSV com os detalhes das imagens
df = pd.read_csv(file_path + csv_name)
df = df.rename(columns={'image': 'image_id', 'level': 'dr_grade'})  # Ajuste conforme as colunas do seu CSV

# Converter a coluna 'dr_grade' para string, se ainda não estiver
df['dr_grade'] = df['dr_grade'].astype(str)

# Amostrar uma fração dos dados
df_sampled = df.sample(frac=0.05, random_state=43)

# Dividir o conjunto amostrado em treino, validação e teste
train_df, test_df = train_test_split(df_sampled, test_size=0.15, random_state=43)
train_df, val_df = train_test_split(train_df, test_size=0.15/(1-0.15), random_state=43)

# Verifique as primeiras linhas e as dimensões de cada conjunto
print("Train DataFrame Head:")
print(train_df.head())
print("Train DataFrame Shape:", train_df.shape)

print("\nValidation DataFrame Head:")
print(val_df.head())
print("Validation DataFrame Shape:", val_df.shape)

print("\nTest DataFrame Head:")
print(test_df.head())
print("Test DataFrame Shape:", test_df.shape)

batch_size = 16

# Configurar o ImageDataGenerator para treino e validação com normalização
datagen_train_val = ImageDataGenerator(rescale=1./255)
datagen_test = ImageDataGenerator(rescale=1./255)

# Gerador de treino
train_gen = datagen_train_val.flow_from_dataframe(
    dataframe=train_df,
    directory=img_path,
    x_col="image_id",
    y_col="dr_grade",
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode="categorical"
)

# Gerador de validação
val_gen = datagen_train_val.flow_from_dataframe(
    dataframe=val_df,
    directory=img_path,
    x_col="image_id",
    y_col="dr_grade",
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode="categorical"
)

# Gerador de teste
test_gen = datagen_test.flow_from_dataframe(
    dataframe=test_df,
    directory=img_path,
    x_col="image_id",
    y_col="dr_grade",
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False  # Para manter a ordem nas previsões
)

# Verificar as dimensões de uma amostra do batch
for images, labels in train_gen:
    print(f"Dimensões das imagens: {images.shape}")
    print(f"Dimensões dos rótulos: {labels.shape}")
    break

# Obter o número de classes
num_classes = len(train_gen.class_indices)  # Número de classes no dataset

# Carregar EfficientNetB3 com pesos pré-treinados no ImageNet
base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Adicionar camadas de pooling e fully connected para classificação
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Redução de dimensionalidade
x = Dense(128, activation='relu')(x)
output = Dense(num_classes, activation='softmax')(x)  # Ajuste para o número de classes no dataset

# Construir o modelo completo
model = Model(inputs=base_model.input, outputs=output)

# Congelar as camadas do modelo base para preservar o aprendizado prévio
for layer in base_model.layers:
    layer.trainable = False

from tensorflow.keras.mixed_precision import set_global_policy

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

import os
# Caminho para o checkpoint salvo (ajuste conforme necessário)
checkpoint_path = 'model_checkpoint.weights.h5'

# Verificar se o checkpoint existe e carregar os pesos se existir
if os.path.exists(checkpoint_path):
    try:
        model.load_weights(checkpoint_path)
        print("Checkpoint carregado com sucesso.")
    except Exception as e:
        print(f"Erro ao carregar checkpoint: {e}")
else:
    print("Checkpoint não encontrado. Iniciando o treinamento do zero.")

# Definir o callback para salvar o modelo atualizado
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_best_only=True,
    monitor='val_loss',
    mode='min',
    save_weights_only=True
)

# Definir o callback para early stopping
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

import math
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow.keras.callbacks import ReduceLROnPlateau

steps_per_epoch = math.ceil(train_gen.n / train_gen.batch_size)
validation_steps = math.ceil(val_gen.n / val_gen.batch_size)

epochs = 1  # Ajuste o número de épocas conforme necessário

history = model.fit(
    train_gen,
    #steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    #validation_steps=validation_steps,
    epochs=epochs,
    callbacks=[checkpoint_callback, early_stopping_callback]
)

# Avaliação no conjunto de teste
test_loss, test_accuracy = model.evaluate(test_gen)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Obter as previsões e as classes reais para o cálculo do coeficiente kappa
test_gen.reset()
Y_pred = model.predict(test_gen, steps=test_gen.n // test_gen.batch_size + 1)
y_pred = np.argmax(Y_pred, axis=1)
y_true = test_gen.classes

# Calcular o coeficiente kappa
kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
print(f"Cohen's Kappa: {kappa}")

# Gerar relatório de classificação
print(classification_report(y_true, y_pred))

#Diretório para salvar as imagens
output_dir = 'img'

# Plotar a matriz de confusão
cm_display = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
plt.savefig(f'{output_dir}/confusion_matrix.png')
#plt.show()

# Plotar Loss e Acurácia para o treinamento inicial
plt.figure(figsize=(12, 5))

# Treinamento
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.title('Treinamento Inicial')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()

# Validação
plt.subplot(1, 2, 2)
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Validação Inicial')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()

plt.savefig(f'{output_dir}/training_metrics.png')

#plt.show()

# Descongelar todas as camadas do modelo base para fine-tuning
for layer in base_model.layers:
    layer.trainable = True

# Compilar novamente com uma taxa de aprendizado menor e otimizador adequado
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Adicionar callback para reduzir a taxa de aprendizado se necessário
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-7)

# Treinamento de ajuste fino
fine_tune_epochs = 1
total_epochs = epochs + fine_tune_epochs

history_fine = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1] + 1,
    callbacks=[checkpoint_callback, reduce_lr]
)

# Combinar histórico de treinamento
acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
loss = history.history['loss'] + history_fine.history['loss']
val_loss = history.history['val_loss'] + history_fine.history['val_loss']

# Avaliação no conjunto de teste após fine-tuning
test_loss_fine, test_accuracy_fine = model.evaluate(test_gen)
print(f"Test Loss after Fine-Tuning: {test_loss_fine}")
print(f"Test Accuracy after Fine-Tuning: {test_accuracy_fine}")

# Obter as previsões e as classes reais para o cálculo do coeficiente kappa após fine-tuning
test_gen.reset()
Y_pred_fine = model.predict(test_gen, steps=test_gen.n // test_gen.batch_size + 1)
y_pred_fine = np.argmax(Y_pred_fine, axis=1)

# Calcular o coeficiente kappa após fine-tuning
kappa_fine = cohen_kappa_score(y_true, y_pred_fine, weights='quadratic')
print(f"Cohen's Kappa after Fine-Tuning: {kappa_fine}")

# Gerar relatório de classificação após fine-tuning
print(classification_report(y_true, y_pred_fine))

# Plotar a matriz de confusão após fine-tuning
cm_display_fine = ConfusionMatrixDisplay.from_predictions(y_true, y_pred_fine)
plt.savefig(f'{output_dir}/confusion_matrix_fine_tuning.png')
#plt.show()

# Plotar Loss e Acurácia para o treinamento completo
plt.figure(figsize=(12, 5))

# Treinamento completo
plt.subplot(1, 2, 1)
plt.plot(range(total_epochs), loss, label='Train Loss')
plt.plot(range(total_epochs), acc, label='Train Accuracy')
plt.title('Treinamento Completo')
plt.xlabel('Épocas')
plt.ylabel('Loss e Acurácia')
plt.legend()

# Validação completa
plt.subplot(1, 2, 2)
plt.plot(range(total_epochs), val_loss, label='Validation Loss')
plt.plot(range(total_epochs), val_acc, label='Validation Accuracy')
plt.title('Validação Completa')
plt.xlabel('Épocas')
plt.ylabel('Loss e Acurácia')
plt.legend()

plt.savefig(f'{output_dir}/fine_tuning_metrics.png')

#plt.show()