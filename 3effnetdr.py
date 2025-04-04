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
CHECKPOINT_PATH = '/home/vitoroliveira/CNN_Rtinopatia_Diabetica/model_checkpoint.weights.h5'
SAMPLE_FRAC = 1
BATCH_SIZE = 48
EPOCHS = 10
FINE_TUNE_EPOCHS = 20
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# %%
#Carregando o CSV, amostrando e preparando os dados
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

# %% [markdown]
# Gerando os dados de treino e validação

# %%
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

# %% [markdown]
# Construindo e compilando modelos

# %%
# EfficientNetB3

input_b3 = tf.keras.Input(shape=(300, 300, 3))
base_model_b3 = EfficientNetB3(include_top=False, weights='imagenet', input_tensor=input_b3)

x_b3 = base_model_b3.output
x_b3 = GlobalAveragePooling2D()(x_b3)
x_b3 = Dropout(0.2)(x_b3)

predictions_b3 = Dense(5, activation='softmax')(x_b3)

model_b3 = Model(inputs=base_model_b3.input, outputs=predictions_b3)

for layer in base_model_b3.layers:
    layer.trainable = False
    
model_b3.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=focal_loss(),
    metrics=['accuracy']
)

model_b3.summary()
    

# %%
# InceptionV3
input_inc = tf.keras.Input(shape=(299, 299, 3))
base_model_inc = InceptionV3(include_top=False, weights='imagenet', input_tensor=input_inc)

x_inc = base_model_inc.output
x_inc = GlobalAveragePooling2D()(x_inc)
x_inc = Dropout(0.2)(x_inc)

predictions_inc = Dense(5, activation='softmax')(x_inc)

model_inc = Model(inputs=base_model_inc.input, outputs=predictions_inc)

for layer in base_model_inc.layers:
    layer.trainable = False
    
model_inc.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=focal_loss(),
    metrics=['accuracy']
)

model_inc.summary()


# %%




