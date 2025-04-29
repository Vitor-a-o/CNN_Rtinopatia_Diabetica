import os
import sys
import math
import logging
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras import mixed_precision
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3, DenseNet169
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_eff
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_dense
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, cohen_kappa_score, ConfusionMatrixDisplay
from sklearn.utils import class_weight

# -------------------------------------------------------------
# LOGGER CONFIGURATION
# -------------------------------------------------------------
OUTPUT_DIR = 'output2'
os.makedirs(OUTPUT_DIR, exist_ok=True)

log_path = os.path.join(OUTPUT_DIR, 'train_debug.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_path, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
logger.info('Logger initialized. Output directory: %s', OUTPUT_DIR)

# -------------------------------------------------------------
# GPU CONFIGURATION
# -------------------------------------------------------------
K.clear_session()
logger.info('Cleared previous Keras session')

#mixed_precision.set_global_policy('mixed_float16')
#logger.info('Mixed precision policy set to float16')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    logger.info('GPUs found: %s', gpus)
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info('Enabled memory growth for GPUs')
    except RuntimeError as e:
        logger.exception('RuntimeError during GPU config: %s', e)
else:
    logger.warning('No GPU detected — training will use CPU.')

# -------------------------------------------------------------
# PATHS & HYPERPARAMETERS
# -------------------------------------------------------------
IMG_PATH = '/app/resized_train'
CSV_PATH = '/app/CNN_Rtinopatia_Diabetica/trainLabels3.csv'
SAMPLE_FRAC = 1
BATCH_SIZE = 48
EPOCHS = 10
FINE_TUNE_EPOCHS = 10
EFF_WEIGHTS_PATH = os.path.join(OUTPUT_DIR, 'efficientnetb3_weights_only.h5')
DENSE_WEIGHTS_PATH = os.path.join(OUTPUT_DIR, 'densenet169_weights_only.h5')

# -------------------------------------------------------------
# DATA PREPARATION
# -------------------------------------------------------------
logger.info('Starting DATA PREPARATION stage')
df = pd.read_csv(CSV_PATH)
logger.info('CSV loaded: %d rows', len(df))

df = df.sample(frac=SAMPLE_FRAC, random_state=42)
df['Patient_ID'] = df['image'].apply(lambda x: x.split('_')[0])
df['path'] = df['image'].apply(lambda x: os.path.join(IMG_PATH, x))

before_filter = len(df)
df = df[df['path'].apply(os.path.exists)]
logger.info('Filtered non‑existing files: %d -> %d', before_filter, len(df))

df['eye'] = df['image'].apply(lambda x: 1 if x.split('_')[-1].split('.')[0] == 'left' else 0)

rr_df = df[['Patient_ID', 'level']].drop_duplicates()
train_ids, valid_ids = train_test_split(rr_df['Patient_ID'], test_size=0.25, random_state=42, stratify=rr_df['level'])

train_df = df[df['Patient_ID'].isin(train_ids)]
valid_df = df[df['Patient_ID'].isin(valid_ids)]
train_df['level'] = train_df['level'].astype(str)
valid_df['level'] = valid_df['level'].astype(str)
logger.info('Train size: %d, Valid size: %d', len(train_df), len(valid_df))

# -------------------------------------------------------------
# DATA GENERATORS
# -------------------------------------------------------------
logger.info('Creating DATA GENERATORS')

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

logger.info('Generators instantiated successfully')

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
logger.info('Building model architectures')

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
# CUSTOM CALLBACK FOR DETAILED LOGGING
# -------------------------------------------------------------
class LogCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        logger.info('Training started')

    def on_epoch_begin(self, epoch, logs=None):
        logger.info('Epoch %d start', epoch + 1)

    def on_epoch_end(self, epoch, logs=None):
        logger.info('Epoch %d end — loss: %.4f, val_loss: %.4f, acc: %.4f, val_acc: %.4f',
                    epoch + 1,
                    logs.get('loss', float('nan')),
                    logs.get('val_loss', float('nan')),
                    logs.get('accuracy', float('nan')),
                    logs.get('val_accuracy', float('nan')))

    def on_train_end(self, logs=None):
        logger.info('Training finished')

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
    fig_path = f'{OUTPUT_DIR}/{prefix}_metrics.png'
    plt.savefig(fig_path)
    plt.close()
    logger.info('Saved metric plot to %s', fig_path)


def evaluate(model, generator, name):
    """Evaluate model with low‑memory prediction to avoid freezes."""
    logger.info('Evaluating %s', name)
    val_loss, val_acc = model.evaluate(generator, verbose=0)
    logger.info('%s -> Val Loss: %.4f | Val Acc: %.4f', name, val_loss, val_acc)

    generator.reset()
    y_pred_batches = []
    total_samples = generator.samples
    batch_size = generator.batch_size

    for i in range(len(generator)):
        batch_x, _ = generator[i]
        preds = model.predict_on_batch(batch_x)
        y_pred_batches.append(np.argmax(preds, axis=1))
        # Free GPU/CPU RAM periodically
        del batch_x, preds
        gc.collect()
        if (i + 1) * batch_size >= total_samples:
            break

    y_pred = np.concatenate(y_pred_batches)[:total_samples]
    y_true = generator.classes

    kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    logger.info('%s -> Cohen Kappa: %.4f', name, kappa)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    cm_path = f'{OUTPUT_DIR}/{name}_cm.png'
    plt.title(f'Confusion Matrix - {name}')
    plt.savefig(cm_path)
    plt.close()
    logger.info('Saved confusion matrix to %s', cm_path)
    return kappa


def train_model(model, train_gen, valid_gen, ckpt_path):
    logger.info('Beginning initial training — saving best weights to %s', ckpt_path)
    callbacks = [
        ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1, mode='min'),
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-7, verbose=1),
        LogCallback()
    ]
    labels = train_gen.classes
    cw = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    logger.info('Class weights: %s', cw)
    history = model.fit(train_gen, validation_data=valid_gen, epochs=EPOCHS, callbacks=callbacks, class_weight=dict(enumerate(cw)))
    return history


def fine_tune(model, train_gen, valid_gen, ckpt_path, layers_to_unfreeze=20):
    logger.info('Starting fine‑tuning: unfreezing last %d layers', layers_to_unfreeze)
    for layer in model.layers[-layers_to_unfreeze:]:
        layer.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss=focal_loss(), metrics=['accuracy'])
    callbacks = [
        ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1, mode='min'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-7, verbose=1),
        LogCallback()
    ]
    history = model.fit(train_gen, validation_data=valid_gen, initial_epoch=EPOCHS, epochs=EPOCHS + FINE_TUNE_EPOCHS, callbacks=callbacks)
    return history

# -------------------------------------------------------------
# MODEL TRAINING
# -------------------------------------------------------------
try:
    logger.info('Building EfficientNetB3 model')
    model_eff = build_efficientnet_b3()
    logger.info('Building DenseNet169 model')
    model_dense = build_densenet169()
except Exception:
    logger.exception('Error while building models')
    raise

if os.path.exists(EFF_WEIGHTS_PATH) and os.path.getsize(EFF_WEIGHTS_PATH) > 0:
    logger.info('Loading EfficientNetB3 weights from %s', EFF_WEIGHTS_PATH)
    model_eff.load_weights(EFF_WEIGHTS_PATH)
else:
    logger.info('No EfficientNetB3 weights found — training from scratch')

if os.path.exists(DENSE_WEIGHTS_PATH) and os.path.getsize(DENSE_WEIGHTS_PATH) > 0:
    logger.info('Loading DenseNet169 weights from %s', DENSE_WEIGHTS_PATH)
    model_dense.load_weights(DENSE_WEIGHTS_PATH)
else:
    logger.info('No DenseNet169 weights found — training from scratch')

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
logger.info('\n======== FINAL RESULTS ========')
logger.info('EfficientNetB3  (init): %.4f | FT: %.4f', kappa_eff, kappa_eff_ft)
logger.info('DenseNet169     (init): %.4f | FT: %.4f', kappa_dense, kappa_dense_ft)
logger.info('===================================\n')

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

logger.info('Training history saved to %s', out_path)
logger.info('Execution finished — see %s for detailed logs', log_path)
