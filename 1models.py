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
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_dense
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, cohen_kappa_score, classification_report, ConfusionMatrixDisplay
from sklearn.utils import class_weight

# -------------------------------------------------------------
# GPU CONFIGURATION
# -------------------------------------------------------------
K.clear_session()

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# -------------------------------------------------------------
# PATHS & HYPERPARAMETERS
# -------------------------------------------------------------
IMG_PATH = '/app/resized_train'
CSV_PATH = '/app/CNN_Rtinopatia_Diabetica/trainLabels3.csv'
SAMPLE_FRAC = 1
BATCH_SIZE = 48
EPOCHS = 5
FINE_TUNE_EPOCHS = 5
OUTPUT_DIR = 'output2'
EFF_WEIGHTS_PATH = os.path.join(OUTPUT_DIR, 'efficientnetb3_weights_only.h5')
DENSE_WEIGHTS_PATH = os.path.join(OUTPUT_DIR, 'densenet169_weights_only.h5')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------
# DATA PREPARATION
# -------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
df = df.sample(frac=SAMPLE_FRAC, random_state=42)
df['Patient_ID'] = df['image'].apply(lambda x: x.split('_')[0])
df['path'] = df['image'].apply(lambda x: os.path.join(IMG_PATH, x))
df = df[df['path'].apply(os.path.exists)]
df['eye'] = df['image'].apply(lambda x: 1 if x.split('_')[-1].split('.')[0] == 'left' else 0)

rr_df = df[['Patient_ID', 'level']].drop_duplicates()
train_ids, valid_ids = train_test_split(
    rr_df['Patient_ID'], test_size=0.25, random_state=42, stratify=rr_df['level'])
train_df = df[df['Patient_ID'].isin(train_ids)]
valid_df = df[df['Patient_ID'].isin(valid_ids)]
train_df['level'] = train_df['level'].astype(str)
valid_df['level'] = valid_df['level'].astype(str)

# -------------------------------------------------------------
# DATA GENERATORS
# -------------------------------------------------------------

def create_datagen(preprocess_func, augment=True):
    if augment:
        return ImageDataGenerator(
            preprocessing_function=preprocess_func,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
    else:
        return ImageDataGenerator(preprocessing_function=preprocess_func)

train_gen_eff = create_datagen(preprocess_eff, augment=True).flow_from_dataframe(
    train_df, x_col='path', y_col='level', target_size=(300, 300),
    batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True)

valid_gen_eff = create_datagen(preprocess_eff, augment=False).flow_from_dataframe(
    valid_df, x_col='path', y_col='level', target_size=(300, 300),
    batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)

train_gen_dense = create_datagen(preprocess_dense, augment=True).flow_from_dataframe(
    train_df, x_col='path', y_col='level', target_size=(224, 224),
    batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True)

valid_gen_dense = create_datagen(preprocess_dense, augment=False).flow_from_dataframe(
    valid_df, x_col='path', y_col='level', target_size=(224, 224),
    batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)

# -------------------------------------------------------------
# LOSS FUNCTION
# -------------------------------------------------------------

def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        eps = 1e-7
        y_pred = K.clip(y_pred, eps, 1 - eps)
        cross_entropy = -y_true * K.log(y_pred)
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        return K.sum(loss, axis=1)
    return focal_loss_fixed

# -------------------------------------------------------------
# MODEL BUILDERS
# -------------------------------------------------------------

def build_efficientnet_b3():
    inp = tf.keras.Input(shape=(300, 300, 3))
    base = EfficientNetB3(include_top=False, weights='imagenet', input_tensor=inp)
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.2)(x)
    out = Dense(5, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=out)
    for layer in base.layers:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=focal_loss(), metrics=['accuracy'])
    return model

def build_densenet169():
    inp = tf.keras.Input(shape=(224, 224, 3))
    base = DenseNet169(include_top=False, weights='imagenet', input_tensor=inp)
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.2)(x)
    out = Dense(5, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=out)
    for layer in base.layers:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=focal_loss(), metrics=['accuracy'])
    return model

# -------------------------------------------------------------
# TRAINING UTILITIES
# -------------------------------------------------------------

def plot_metrics(history, prefix):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{prefix}_metrics.png')
    plt.close()

def evaluate(model, generator, name):
    val_loss, val_acc = model.evaluate(generator, verbose=0)
    print(f'{name} -> Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
    generator.reset()
    y_pred = np.argmax(model.predict(generator, verbose=0), axis=1)
    y_true = generator.classes
    kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    print(f'{name} -> Cohen Kappa: {kappa:.4f}')
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f'Confusion Matrix - {name}')
    plt.savefig(f'{OUTPUT_DIR}/{name}_cm.png')
    plt.close()
    return kappa

def train_model(model, train_gen, valid_gen, ckpt_path):
    callbacks = [
        ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1, mode='min'),
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-7, verbose=1)
    ]
    labels = train_gen.classes
    cw = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    history = model.fit(train_gen, validation_data=valid_gen, epochs=EPOCHS, callbacks=callbacks, class_weight=dict(enumerate(cw)))
    return history

def fine_tune(model, train_gen, valid_gen, ckpt_path, layers_to_unfreeze=20):
    for layer in model.layers[-layers_to_unfreeze:]:
        layer.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss=focal_loss(), metrics=['accuracy'])
    callbacks = [
        ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1, mode='min'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-7, verbose=1)
    ]
    history = model.fit(train_gen, validation_data=valid_gen, initial_epoch=EPOCHS, epochs=EPOCHS + FINE_TUNE_EPOCHS, callbacks=callbacks)
    return history

# -------------------------------------------------------------
# MODEL TRAINING
# -------------------------------------------------------------
model_eff = build_efficientnet_b3()
model_dense = build_densenet169()

if os.path.exists(EFF_WEIGHTS_PATH) and os.path.getsize(EFF_WEIGHTS_PATH) > 0:
    print(f'Loading EfficientNetB3 weights from {EFF_WEIGHTS_PATH}')
    model_eff.load_weights(EFF_WEIGHTS_PATH)
else:
    print(f'No EfficientNetB3 weights found at {EFF_WEIGHTS_PATH}, training from scratch.')
if os.path.exists(DENSE_WEIGHTS_PATH) and os.path.getsize(DENSE_WEIGHTS_PATH) > 0:
    print(f'Loading DenseNet169 weights from {DENSE_WEIGHTS_PATH}')
    model_dense.load_weights(DENSE_WEIGHTS_PATH)
else:
    print(f'No DenseNet169 weights found at {DENSE_WEIGHTS_PATH}, training from scratch.')

# Initial training
hist_eff = train_model(model_eff, train_gen_eff, valid_gen_eff, EFF_WEIGHTS_PATH)
plot_metrics(hist_eff, 'efficientnetb3')
kappa_eff = evaluate(model_eff, valid_gen_eff, 'EfficientNetB3')

hist_dense = train_model(model_dense, train_gen_dense, valid_gen_dense, DENSE_WEIGHTS_PATH)
plot_metrics(hist_dense, 'densenet169')
kappa_dense = evaluate(model_dense, valid_gen_dense, 'DenseNet169')

# Fine tuning
hist_eff_ft = fine_tune(model_eff, train_gen_eff, valid_gen_eff, EFF_WEIGHTS_PATH)
plot_metrics(hist_eff_ft, 'efficientnetb3_finetune')
kappa_eff_ft = evaluate(model_eff, valid_gen_eff, 'EfficientNetB3_FT')

hist_dense_ft = fine_tune(model_dense, train_gen_dense, valid_gen_dense, DENSE_WEIGHTS_PATH)
plot_metrics(hist_dense_ft, 'densenet169_finetune')
kappa_dense_ft = evaluate(model_dense, valid_gen_dense, 'DenseNet169_FT')

# -------------------------------------------------------------
# RESULTS & LOGGING
# -------------------------------------------------------------
print('\n======== RESULTADOS FINAIS ========')
print(f'EfficientNetB3  (init): {kappa_eff:.4f} | FT: {kappa_eff_ft:.4f}')
print(f'DenseNet169     (init): {kappa_dense:.4f} | FT: {kappa_dense_ft:.4f}')
print('===================================\n')

results = {
    'Model': ['EfficientNetB3', 'DenseNet169'],
    'Epochs': [EPOCHS, EPOCHS],
    'Fine_Tuning_Epochs': [FINE_TUNE_EPOCHS, FINE_TUNE_EPOCHS],
    'Initial_Kappa': [kappa_eff, kappa_dense],
    'Fine_Tuning_Kappa': [kappa_eff_ft, kappa_dense_ft],
}

out_path = os.path.join(OUTPUT_DIR, 'train_history.csv')
pd.DataFrame(results).to_csv(
    out_path,
    mode='a' if os.path.exists(out_path) else 'w',
    header=not os.path.exists(out_path),
    index=False
)

print(f'Histórico salvo em {out_path}')
