import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3, DenseNet169
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_eff
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_dn
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, cohen_kappa_score, classification_report, ConfusionMatrixDisplay
from sklearn.utils import class_weight

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

print("GPUs visíveis:", tf.config.list_physical_devices('GPU'))

# Configurações e caminhos
IMG_PATH = "/app/resized_train"
CSV_PATH = "/app/CNN_Rtinopatia_Diabetica/trainLabels3.csv"
SAMPLE_FRAC = 1
BATCH_SIZE = 48
EPOCHS = 10
FINE_TUNE_EPOCHS = 10
OUTPUT_DIR = 'output2'
EFF_WEIGHTS_PATH = os.path.join(OUTPUT_DIR, "efficientnetb3_weights_only.h5")
DN_WEIGHTS_PATH = os.path.join(OUTPUT_DIR, "densenet169_weights_only.h5")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Carrega CSV e prepara DataFrame
df = pd.read_csv(CSV_PATH)
df = df.sample(frac=SAMPLE_FRAC, random_state=42)
df['Patient_ID'] = df['image'].apply(lambda x: x.split('_')[0])
df['path'] = df['image'].apply(lambda x: os.path.join(IMG_PATH, x))
df['exists'] = df['path'].map(lambda x: os.path.exists(x))

print(df['exists'].sum(), 'imagens encontradas de', len(df), 'no total')
df['eye'] = df['image'].apply(lambda x: 1 if x.split('_')[-1].split('.')[0] == 'left' else 0)

from keras.utils import to_categorical
df['level_cat'] = df['level'].map(lambda x: to_categorical(x, 1 + df['level'].max()))

df.dropna(inplace=True)
df = df[df['exists']]
df.sample(5)

df[['level', 'eye']].hist(figsize=(10, 5))

# Divide em treino/validação
rr_df = df[['Patient_ID', 'level']].drop_duplicates()
train_ids, valid_ids = train_test_split(rr_df['Patient_ID'], test_size=0.25, random_state=42, stratify=rr_df['level'])
train_df = df[df['Patient_ID'].isin(train_ids)]
valid_df = df[df['Patient_ID'].isin(valid_ids)]
print('Train:', len(train_df), 'Valid:', len(valid_df))

train_df['level'] = train_df['level'].astype(str)
valid_df['level'] = valid_df['level'].astype(str)

# --------------------------------------------------------------------------------
# DataGenerators para o EfficientNetB3
# --------------------------------------------------------------------------------
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

valid_datagen_eff = ImageDataGenerator(
    preprocessing_function=preprocess_eff,
    rescale=1./255
)

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

# --------------------------------------------------------------------------------
# DataGenerators para o DenseNet169 (substituindo InceptionV3)
# Ajustamos o target_size para 224 x 224 (padrão DenseNet).
# --------------------------------------------------------------------------------
train_datagen_dn = ImageDataGenerator(
    preprocessing_function=preprocess_dn,
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
)

valid_datagen_dn = ImageDataGenerator(
    preprocessing_function=preprocess_dn,
    rescale=1./255
)

train_generator_dn = train_datagen_dn.flow_from_dataframe(
    dataframe=train_df,
    x_col='path',
    y_col='level',
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
)

valid_generator_dn = valid_datagen_dn.flow_from_dataframe(
    dataframe=valid_df,
    x_col='path',
    y_col='level',
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
)

# Função de perda focal_loss
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        eps = 1e-7
        y_pred = K.clip(y_pred, eps, 1 - eps)
        cross_entropy = -y_true * K.log(y_pred)
        loss = alpha * K.pow((1 - y_pred), gamma) * cross_entropy
        return K.sum(loss, axis=1)
    return focal_loss_fixed

# =============================================================================
# Função para construir o modelo do EfficientNetB3
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

# =============================================================================
# Função para construir o modelo do DenseNet169 (substituindo InceptionV3)
# =============================================================================
def build_densenet169():
    input_dn = tf.keras.Input(shape=(224, 224, 3))
    base_model_dn = DenseNet169(include_top=False, weights='imagenet', input_tensor=input_dn)

    x_dn = base_model_dn.output
    x_dn = GlobalAveragePooling2D()(x_dn)
    x_dn = Dropout(0.2)(x_dn)

    predictions_dn = Dense(5, activation='softmax')(x_dn)
    model_dn_ = Model(inputs=base_model_dn.input, outputs=predictions_dn)

    for layer in base_model_dn.layers:
        layer.trainable = False

    model_dn_.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=focal_loss(),
        metrics=['accuracy']
    )
    return model_dn_

# =============================================================================
# Funções para plot e avaliação
# =============================================================================
def plot_metrics(history, filename_prefix):
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
    val_loss, val_accuracy = model.evaluate(valid_generator, verbose=0)
    print(f"{model_name} -> Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

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

# =============================================================================
# Construindo o modelo EfficientNetB3
# =============================================================================
print("\n--- Preparando EfficientNetB3 ---")
model_b3 = build_efficientnet_b3()

# Verifica se há checkpoint de pesos salvos
if os.path.exists(EFF_WEIGHTS_PATH) and os.path.getsize(EFF_WEIGHTS_PATH) > 0:
    model_b3.load_weights(EFF_WEIGHTS_PATH)
    print("Pesos do EfficientNetB3 carregados do checkpoint.")
else:
    print("Checkpoint do EfficientNetB3 não encontrado. Treinando do zero.")

checkpoint_eff = ModelCheckpoint(EFF_WEIGHTS_PATH, monitor='val_loss', save_best_only=True,
                                 verbose=1, mode='min', save_weights_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr_eff = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-7, verbose=1)

train_labels_eff = train_generator_eff.classes
class_weights_eff = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels_eff),
    y=train_labels_eff
)
class_weights_eff = dict(enumerate(class_weights_eff))

history_b3 = model_b3.fit(
    train_generator_eff,
    validation_data=valid_generator_eff,
    epochs=EPOCHS,
    callbacks=[checkpoint_eff, early_stopping],
    class_weight=class_weights_eff
)

plot_metrics(history_b3, 'efficientnetb3')
kappa_b3 = evaluate_and_plot_confusion_matrix(model_b3, valid_generator_eff, 'EfficientNetB3')

# =============================================================================
# Construindo o modelo DenseNet169
# =============================================================================
print("\n--- Preparando DenseNet169 ---")
model_dn = build_densenet169()

# Verifica se há checkpoint de pesos salvos
if os.path.exists(DN_WEIGHTS_PATH) and os.path.getsize(DN_WEIGHTS_PATH) > 0:
    model_dn.load_weights(DN_WEIGHTS_PATH)
    print("Pesos do DenseNet169 carregados do checkpoint.")
else:
    print("Checkpoint do DenseNet169 não encontrado. Treinando do zero.")

checkpoint_dn = ModelCheckpoint(DN_WEIGHTS_PATH, monitor='val_loss', save_best_only=True,
                                verbose=1, mode='min', save_weights_only=True)
reduce_lr_dn = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-7, verbose=1)

train_labels_dn = train_generator_dn.classes
class_weights_dn = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels_dn),
    y=train_labels_dn
)
class_weights_dn = dict(enumerate(class_weights_dn))

history_dn = model_dn.fit(
    train_generator_dn,
    validation_data=valid_generator_dn,
    epochs=EPOCHS,
    callbacks=[checkpoint_dn, early_stopping],
    class_weight=class_weights_dn
)

plot_metrics(history_dn, 'densenet169')
kappa_dn = evaluate_and_plot_confusion_matrix(model_dn, valid_generator_dn, 'DenseNet169')

# =============================================================================
# Fine Tuning EfficientNetB3
# =============================================================================
print("\n--- Fine Tuning EfficientNetB3 ---")
for layer in model_b3.layers[-20:]:
    layer.trainable = True

model_b3.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=focal_loss(),
    metrics=['accuracy']
)

history_eff_fine = model_b3.fit(
    train_generator_eff,
    validation_data=valid_generator_eff,
    initial_epoch=EPOCHS,
    epochs=EPOCHS + FINE_TUNE_EPOCHS,
    callbacks=[checkpoint_eff, reduce_lr_eff],
    class_weight=class_weights_eff
)

plot_metrics(history_eff_fine, 'efficientnetb3_finetune')
kappa_b3_fine = evaluate_and_plot_confusion_matrix(model_b3, valid_generator_eff, 'EfficientNetB3_FineTune')

# =============================================================================
# Fine Tuning DenseNet169
# =============================================================================
print("\n--- Fine Tuning DenseNet169 ---")
for layer in model_dn.layers[-20:]:
    layer.trainable = True

model_dn.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=focal_loss(),
    metrics=['accuracy']
)

history_dn_fine = model_dn.fit(
    train_generator_dn,
    validation_data=valid_generator_dn,
    initial_epoch=EPOCHS,
    epochs=EPOCHS + FINE_TUNE_EPOCHS,
    callbacks=[checkpoint_dn, reduce_lr_dn],
    class_weight=class_weights_dn
)

plot_metrics(history_dn_fine, 'densenet169_finetune')
kappa_dn_fine = evaluate_and_plot_confusion_matrix(model_dn, valid_generator_dn, 'DenseNet169_FineTune')

# =============================================================================
# Resultados Finais
# =============================================================================
print("\n======== RESULTADOS FINAIS ========")
print(f"EfficientNetB3 (inicial): {kappa_b3:.4f} | Fine-Tuning: {kappa_b3_fine:.4f}")
print(f"DenseNet169    (inicial): {kappa_dn:.4f} | Fine-Tuning: {kappa_dn_fine:.4f}")
print("====================================\n")

results = {
    'Model': ['EfficientNetB3', 'DenseNet169'],
    'Epochs': [EPOCHS, EPOCHS],
    'Fine_Tuning_Epochs': [FINE_TUNE_EPOCHS, FINE_TUNE_EPOCHS],
    'Initial_Kappa': [kappa_b3, kappa_dn],
    'Fine_Tuning_Kappa': [kappa_b3_fine, kappa_dn_fine]
}

path = os.path.join(OUTPUT_DIR, 'train_history.csv')
if os.path.exists(path) and os.path.getsize(path) > 0:
    results_df = pd.read_csv(path)
    results_df = pd.concat([results_df, pd.DataFrame(results)], ignore_index=True)
else:
    results_df = pd.DataFrame(results)
results_df.to_csv(path, index=False)
