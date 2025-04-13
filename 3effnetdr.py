# %%
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3, InceptionV3
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_eff
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inc
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, cohen_kappa_score, classification_report, ConfusionMatrixDisplay
from sklearn.utils import class_weight

# %%
# Libera a sessão atual
K.clear_session()

# Evita alocação total da memória da GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# %%
print("GPUs visíveis:", tf.config.list_physical_devices('GPU'))

# %% [markdown]
# Configurações e caminhos

# %%
#IMG_PATH = "/home/vitoroliveira/resized_train"
IMG_PATH = "/app/resized_train"
#CSV_PATH = "/home/vitoroliveira/CNN_Rtinopatia_Diabetica/trainLabels3.csv"  # Certifique-se de ajustar o caminho se necessário
CSV_PATH = "/app/CNN_Rtinopatia_Diabetica/trainLabels3.csv"
SAMPLE_FRAC = 1
BATCH_SIZE = 48
EPOCHS = 10
FINE_TUNE_EPOCHS = 20
OUTPUT_DIR = 'output2'
EFF_WEIGHTS_PATH = os.path.join(OUTPUT_DIR, "efficientnetb3_weights_only.h5")
INC_WEIGHTS_PATH = os.path.join(OUTPUT_DIR, "inceptionv3_weights_only.h5")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# %%
# ========================================================================
#Carregando o CSV, amostrando e preparando os dados
# =======================================================================
df = pd.read_csv(CSV_PATH)
df = df.sample(frac=SAMPLE_FRAC, random_state=42)
df['Patient_ID'] = df['image'].apply(lambda x: x.split('_')[0])
df['path'] = df['image'].apply(lambda x: os.path.join(IMG_PATH, x))
df['exists'] = df['path'].map(lambda x: os.path.exists(x))

print(df['exists'].sum(), 'imagens encontradas de ', len(df), 'no total')
df['eye'] = df['image'].apply(lambda x: 1 if x.split('_')[-1].split('.')[0] == 'left' else 0)
from keras.utils import to_categorical
df['level_cat'] = df['level'].map(lambda x: to_categorical(x, 1+df['level'].max()))

df.dropna(inplace=True)
df = df[df['exists']]
df.sample(5)

# %%
df[['level', 'eye']].hist(figsize = (10, 5))

# %% [markdown]
# Dividindo os dados em treino e teste

# %%
rr_df = df[['Patient_ID', 'level']].drop_duplicates()
train_ids, valid_ids = train_test_split(rr_df['Patient_ID'], test_size=0.25, random_state=42, stratify=rr_df['level'])
train_df = df[df['Patient_ID'].isin(train_ids)]
valid_df = df[df['Patient_ID'].isin(valid_ids)]
print('Train:', len(train_df), 'Valid:', len(valid_df))

# %%
# ========================================================================
# Gerando os dados de treino e validação
# ========================================================================

train_df['level'] = train_df['level'].astype(str)
valid_df['level'] = valid_df['level'].astype(str)

#EfficientNetB3

train_datagen_eff = ImageDataGenerator(
    preprocessing_function=preprocess_eff,
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    
)

valid_datagen_eff = ImageDataGenerator(preprocessing_function=preprocess_eff, rescale=1./255)

train_generator_eff = train_datagen_eff.flow_from_dataframe(
    dataframe=train_df,
    x_col='path',
    y_col='level',
    target_size=(300, 300),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
)

valid_generator_eff = valid_datagen_eff.flow_from_dataframe(
    dataframe=valid_df,
    x_col='path',
    y_col='level',
    target_size=(300, 300),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
)

# InceptionV3

train_datagen_inc = ImageDataGenerator(
    preprocessing_function=preprocess_inc,
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
)

valid_datagen_inc = ImageDataGenerator(preprocessing_function=preprocess_inc, rescale=1./255)

train_generator_inc = train_datagen_inc.flow_from_dataframe(
    dataframe=train_df,
    x_col='path',
    y_col='level',
    target_size=(299, 299),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
)

valid_generator_inc = valid_datagen_inc.flow_from_dataframe(
    dataframe=valid_df,
    x_col='path',
    y_col='level',
    target_size=(299, 299),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
)


# %% [markdown]
# Função de perda focal_loss para lidar com o desbalanceamento de classes

# %%
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        eps =  1e-7
        y_pred = K.clip(y_pred, eps, 1 - eps)
        cross_entropy = -y_true * K.log(y_pred)
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        return K.sum(loss, axis=1)
    return focal_loss_fixed
# %% 
# =============================================================================
# Função auxiliar para construir o modelo do zero (caso o load_model falhe)
# =============================================================================
def build_efficientnet_b3():
    input_b3 = tf.keras.Input(shape=(300, 300, 3))
    base_model_b3 = EfficientNetB3(include_top=False, weights='imagenet', input_tensor=input_b3)

    x_b3 = base_model_b3.output
    x_b3 = GlobalAveragePooling2D()(x_b3)
    x_b3 = Dropout(0.2)(x_b3)

    predictions_b3 = Dense(5, activation='softmax')(x_b3)
    model_b3_ = Model(inputs=base_model_b3.input, outputs=predictions_b3)

    for layer in base_model_b3.layers:
        layer.trainable = False

    model_b3_.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=focal_loss(),
        metrics=['accuracy']
    )
    return model_b3_


def build_inception_v3():
    input_inc = tf.keras.Input(shape=(299, 299, 3))
    base_model_inc = InceptionV3(include_top=False, weights='imagenet', input_tensor=input_inc)

    x_inc = base_model_inc.output
    x_inc = GlobalAveragePooling2D()(x_inc)
    x_inc = Dropout(0.2)(x_inc)

    predictions_inc = Dense(5, activation='softmax')(x_inc)
    model_inc_ = Model(inputs=base_model_inc.input, outputs=predictions_inc)

    for layer in base_model_inc.layers:
        layer.trainable = False

    model_inc_.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=focal_loss(),
        metrics=['accuracy']
    )
    return model_inc_


# %%
# =============================================================================
# Funções para plot e avaliação
# =============================================================================

def plot_metrics(history, filename_prefix):
    """
    Plota e salva gráficos de loss e accuracy para treino e validação.
    """
    plt.figure(figsize=(12, 5))
    
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.legend()
    
    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{filename_prefix}_training_metrics.png")
    plt.close()

def evaluate_and_plot_confusion_matrix(model, valid_generator, model_name):
    """
    Faz predições, mostra métricas e salva a matriz de confusão.
    """
    # Avaliação (loss e accuracy) no conjunto de validação
    val_loss, val_accuracy = model.evaluate(valid_generator, verbose=0)
    print(f"{model_name} -> Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Predições para matriz de confusão e Kappa
    valid_generator.reset()
    y_pred_proba = model.predict(valid_generator, steps=math.ceil(valid_generator.n / valid_generator.batch_size))
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = valid_generator.classes

    # Kappa
    kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    print(f"{model_name} -> Cohen's Kappa: {kappa:.4f}")

    # Classification Report
    print(classification_report(y_true, y_pred, digits=4))

    # Plot da Matriz de Confusão
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Matriz de Confusão - {model_name}")
    plt.savefig(f"{OUTPUT_DIR}/{model_name}_confusion_matrix.png")
    plt.close()

    return kappa


# %%
# =============================================================================
# Treinamento inicial do modelo EfficientNetB3
# =============================================================================

# Carregando os pesos do checkpoint se existir
if os.path.exists(EFF_WEIGHTS_PATH):
    model_b3.load_weights(EFF_WEIGHTS_PATH)
    print("Pesos do EfficientNetB3 carregados do checkpoint.")
else:
    print("Checkpoint do EfficientNetB3 não encontrado. Treinando do zero.")
    
# Definindo callbacks
checkpoint_eff = ModelCheckpoint(EFF_WEIGHTS_PATH), monitor='val_loss', save_best_only=True, verbose=1, mode='min', save_weights_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr_eff = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-7, verbose=1)

# Cálculo de class weights para lidar com possível desbalanceamento
train_labels_eff = train_generator_eff.classes
class_weights_eff = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels_eff),
    y=train_labels_eff
)
class_weights_eff = dict(enumerate(class_weights_eff))

# Treinamento
history_b3 = model_b3.fit(
    train_generator_eff,
    validation_data=valid_generator_eff,
    epochs=EPOCHS,
    callbacks=[checkpoint_eff, early_stopping],
    class_weight= class_weights_eff
)
# %% [markdown]
# Plotando métricas e avaliando EfficientNetB3
# %%
plot_metrics(history_b3, 'efficientnetb3')
kappa_b3 = evaluate_and_plot_confusion_matrix(model_b3, valid_generator_eff, 'EfficientNetB3')

# %% [markdown]
# Treinamento inicial do modelo InceptionV3
# %%
# Carregando os pesos do checkpoint se existir
if os.path.exists(INC_WEIGHTS_PATH):
    model_inc.load_weights(INC_WEIGHTS_PATH):)
    print("Pesos do InceptionV3 carregados do checkpoint.")
else:
    print("Checkpoint do InceptionV3 não encontrado. Treinando do zero.")

# Definindo callbacks
checkpoint_inc = ModelCheckpoint(INC_WEIGHTS_PATH):, monitor='val_loss', save_best_only=True, verbose=1, mode='min', save_weights_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr_inc = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-7, verbose=1)

# Cálculo de class weights para lidar com possível desbalanceamento
train_labels_inc = train_generator_inc.classes
class_weights_inc = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels_inc),
    y=train_labels_inc
)
class_weights_inc = dict(enumerate(class_weights_inc))

# Treinamento
history_inc = model_inc.fit(
    train_generator_inc,
    validation_data=valid_generator_inc,
    epochs=EPOCHS,
    callbacks=[checkpoint_inc, early_stopping],
    class_weight= class_weights_inc
)
# %% [markdown]
# Plotando métricas e avaliando InceptionV3
# %%
plot_metrics(history_inc, 'inceptionv3')
kappa_inc = evaluate_and_plot_confusion_matrix(model_inc, valid_generator_inc, 'InceptionV3')

# %% [markdown]
# Treinamento de Fine Tuning do modelo EfficientNetB3
# %%

# Descongelando as últimas camadas do EfficientNetB3
for layer in model_b3.layers[-20:]:
    layer.trainable = True
    
# Recompilando o modelo
model_b3.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=focal_loss(),
    metrics=['accuracy']
)

reduce_lr_eff = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-7, verbose=1)

history_eff_fine = model_b3.fit(
    train_generator_eff,
    validation_data=valid_generator_eff,
    initial_epoch=EPOCHS,  # continua de onde parou
    epochs=EPOCHS + FINE_TUNE_EPOCHS,
    callbacks=[checkpoint_eff, reduce_lr_eff],
    class_weight=class_weights_eff
)
# %% [markdown]
# Plotando métricas e avaliando EfficientNetB3 após Fine Tuning
# %%
plot_metrics(history_eff_fine, 'efficientnetb3_finetune')
kappa_b3_fine = evaluate_and_plot_confusion_matrix(model_b3, valid_generator_eff, 'EfficientNetB3_FineTune')

# %% [markdown]
# Treinamento de Fine Tuning do modelo InceptionV3
# %%

# Descongelando as últimas camadas do InceptionV3
for layer in model_inc.layers[-20:]:
    layer.trainable = True
    
# Recompilando o modelo
model_inc.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=focal_loss(),
    metrics=['accuracy']
)

reduce_lr_inc = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-7, verbose=1)

history_inc_fine = model_inc.fit(
    train_generator_inc,
    validation_data=valid_generator_inc,
    initial_epoch=EPOCHS,  # continua de onde parou
    epochs=EPOCHS + FINE_TUNE_EPOCHS,
    callbacks=[checkpoint_inc, reduce_lr_inc],
    class_weight=class_weights_inc
)

# %% [markdown]
# Plotando métricas e avaliando InceptionV3 após Fine Tuning
# %%
plot_metrics(history_inc_fine, 'inceptionv3_finetune')
kappa_inc_fine = evaluate_and_plot_confusion_matrix(model_inc, valid_generator_inc, 'InceptionV3_FineTune')
# %% [markdown]
# Resultados Finais
# %%

print("\n======== RESULTADOS FINAIS ========")
print(f"EfficientNetB3   (inicial): {kappa_b3:.4f} | Fine-Tuning: {kappa_b3_fine:.4f}")
print(f"InceptionV3   (inicial): {kappa_inc:.4f} | Fine-Tuning: {kappa_inc_fine:.4f}")
print("====================================\n")

# Salvando resultados em train_history.csv
results = {
    'Model': ['EfficientNetB3', 'InceptionV3'],
    'Epochs': [EPOCHS, EPOCHS],
    'Fine_Tuning_Epochs': [FINE_TUNE_EPOCHS, FINE_TUNE_EPOCHS],
    'Initial_Kappa': [kappa_b3, kappa_inc],
    'Fine_Tuning_Kappa': [kappa_b3_fine, kappa_inc_fine]
}

path = os.path.join(OUTPUT_DIR, 'train_history.csv')
if os.path.exists(path) and os.path.getsize(path) > 0:
    results_df = pd.read_csv(path)
    results_df = pd.concat([results_df, pd.DataFrame(results)], ignore_index=True)
else:
    results_df = pd.DataFrame(results)
results_df.to_csv(path, index=False)

# %%
