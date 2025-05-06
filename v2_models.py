"""
Diabetic Retinopathy – Modern Backbones Pipeline
-----------------------------------------------
Full training script (warm‑up + fine‑tune + TTA) agora suportando:
  • **EfficientNetV2‑S** (input 384 px)
  • **ConvNeXt‑Tiny**  (input 224 px)

Funcionalidades mantidas:
  • Crop + CLAHE, Albumentations, MixUp
  • Quadratic‑Weighted Kappa loss
  • Cosine LR, EarlyStopping, checkpoint por val_loss (pode trocar para val_kappa)
  • TTA + threshold sweep, CSV resumo

Execute: `python modern_models.py` (ou renomeie conforme desejar)
"""

import os, sys, math, logging, gc, random, json
from typing import List, Dict

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from sklearn.utils import class_weight
import albumentations as A

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
OUTPUT_DIR   = "output2"
IMG_DIR      = "/app/resized_train"
CSV_PATH     = "/app/CNN_Rtinopatia_Diabetica/trainLabels3.csv"
SEED         = 42
BATCH_SIZE   = 32
EPOCHS       = 15   # warm‑up
FINE_TUNE_EPOCHS = 15 # fine‑tune
NUM_CLASSES  = 5

MODELS: List[Dict] = [
    {  # EfficientNetV2‑S
        "name": "EfficientNetV2S",
        "input": 384,
        "weights": os.path.join(OUTPUT_DIR, "effv2s_qwk.h5"),
    },
    {  # ConvNeXt‑Tiny
        "name": "ConvNeXtTiny",
        "input": 224,
        "weights": os.path.join(OUTPUT_DIR, "convnext_tiny_qwk.h5"),
    },
]

# -------------------------------------------------------------
# LOGGER
# -------------------------------------------------------------
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
# PRE‑PROCESS & AUGS
# -------------------------------------------------------------

def crop_and_clahe(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        img = img[y:y+h, x:x+w]
    h, w = img.shape[:2]
    if h != w:
        diff = abs(h-w)
        if h>w:
            img = cv2.copyMakeBorder(img, 0, 0, diff//2, diff-diff//2, cv2.BORDER_CONSTANT, value=0)
        else:
            img = cv2.copyMakeBorder(img, diff//2, diff-diff//2, 0, 0, cv2.BORDER_CONSTANT, value=0)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(2.0, (8,8)).apply(l)
    return cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2BGR)


def build_aug(size):
    train_aug = A.Compose([
        A.RandomResizedCrop(size, size, scale=(0.9,1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=20, p=0.7),
        A.RandomBrightnessContrast(0.2,0.2, p=0.7),
        A.GaussNoise(p=0.4),
        A.CoarseDropout(max_holes=8, max_height=size//10, max_width=size//10, fill_value=0, p=0.5),
    ])
    val_aug = A.Compose([A.Resize(size,size)])
    return train_aug, val_aug

# -------------------------------------------------------------
# DATA SEQUENCE
# -------------------------------------------------------------
class DRSeq(Sequence):
    def __init__(self, df, batch, size, aug_t, aug_v, train):
        self.df = df.reset_index(drop=True)
        self.batch = batch; self.size=size; self.aug_t=aug_t; self.aug_v=aug_v; self.train=train
        self.idx = np.arange(len(df)); self.on_epoch_end()
    def __len__(self): return math.ceil(len(self.df)/self.batch)
    def on_epoch_end(self):
        if self.train:
            np.random.shuffle(self.idx)
    def __getitem__(self, i):
        ids = self.idx[i*self.batch:(i+1)*self.batch]
        imgs, labs = [], []
        for j in ids:
            p = self.df.at[j,'path']
            im = cv2.imread(p)
            if im is None:
                continue
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = crop_and_clahe(im)
            im = (self.aug_t if self.train else self.aug_v)(image=im)["image"]
            imgs.append(im.astype('float32')/255.)
            labs.append(self.df.at[j,'level'])
        x = np.stack(imgs)
        y = tf.keras.utils.to_categorical(labs, NUM_CLASSES)
        if self.train and random.random()<0.5 and len(x)>1:
            lam = np.clip(np.random.beta(0.4,0.4), 0.1,0.9)
            idx = np.random.permutation(x.shape[0])
            x = lam*x + (1-lam)*x[idx]
            y = lam*y + (1-lam)*y[idx]
        return x, y

# -------------------------------------------------------------
# LOSS
# -------------------------------------------------------------

def qwk_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.nn.softmax(y_pred)
    hist_t = tf.reduce_sum(y_true,0)
    hist_p = tf.reduce_sum(y_pred,0)
    conf = tf.matmul(y_true, y_pred, transpose_a=True)
    w = tf.square(tf.expand_dims(tf.range(NUM_CLASSES, dtype=tf.float32),1) - tf.range(NUM_CLASSES, dtype=tf.float32))/(NUM_CLASSES-1)**2
    num = tf.reduce_sum(w*conf)
    denom= tf.reduce_sum(w*tf.tensordot(hist_t, hist_p,0)/tf.cast(tf.shape(y_true)[0],tf.float32))
    return 1 - num/(denom+K.epsilon())

# -------------------------------------------------------------
# BACKBONE FACTORY
# -------------------------------------------------------------
from tensorflow.keras.applications import EfficientNetV2S, ConvNeXtTiny

def build_backbone(name, size):
    inp = Input((size,size,3))
    if name=='EfficientNetV2S':
        base = EfficientNetV2S(include_top=False, weights='imagenet', input_tensor=inp)
    elif name=='ConvNeXtTiny':
        base = ConvNeXtTiny(include_top=False, weights='imagenet', input_tensor=inp)
    else:
        raise ValueError(name)
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.3)(x)
    out = Dense(NUM_CLASSES)(x)
    model = Model(inp,out)
    return base,model

# -------------------------------------------------------------
# LR SCHEDULE
# -------------------------------------------------------------

def cosine(epoch):
    base_lr=1e-4; total=EPOCHS+FINE_TUNE_EPOCHS; return 0.5*base_lr*(1+math.cos(math.pi*epoch/total))

# -------------------------------------------------------------
# DATA
# -------------------------------------------------------------
logger.info('Loading CSV…')
df = pd.read_csv(CSV_PATH)
df['path'] = df['image'].apply(lambda x: os.path.join(IMG_DIR,x))
df = df[df['path'].apply(os.path.exists)].reset_index(drop=True)
df['Patient_ID'] = df['image'].apply(lambda x: x.split('_')[0])
rr = df[['Patient_ID','level']].drop_duplicates()
train_ids, val_ids = train_test_split(rr['Patient_ID'], test_size=0.25, random_state=SEED, stratify=rr['level'])
train_df = df[df['Patient_ID'].isin(train_ids)].reset_index(drop=True)
val_df   = df[df['Patient_ID'].isin(val_ids)].reset_index(drop=True)

# -------------------------------------------------------------
# TRAIN LOOP
# -------------------------------------------------------------
results=[]
for cfg in MODELS:
    name, size, wpath = cfg['name'], cfg['input'], cfg['weights']
    logger.info('\n===== %s (%d px) =====', name, size)
    logger.info('Initializing build_aug for %s', name)
    aug_t, aug_v = build_aug(size)
    logger.info('Augmentation successfully built')
    logger.info('Initializing DRSeq for %s', name)
    tr_seq = DRSeq(train_df, BATCH_SIZE, size, aug_t, aug_v, True)
    va_seq = DRSeq(val_df,   BATCH_SIZE, size, aug_t, aug_v, False)
    logger.info('DRSeq successfully built')
    base, model = build_backbone(name,size)
    for layer in base.layers: layer.trainable=False
    model.compile(tf.keras.optimizers.Adam(1e-4), qwk_loss, ['accuracy'])

    cb_warm=[ModelCheckpoint(wpath, monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1),
             LearningRateScheduler(cosine),
             EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)]
    cw = class_weight.compute_class_weight('balanced', classes=np.arange(NUM_CLASSES), y=train_df['level'])
    model.fit(tr_seq, epochs=EPOCHS, validation_data=va_seq, callbacks=cb_warm, class_weight=dict(enumerate(cw)))

    # fine‑tune
    for layer in model.layers: layer.trainable=True
    model.compile(tf.keras.optimizers.Adam(1e-5), qwk_loss, ['accuracy'])
    cb_ft=[ModelCheckpoint(wpath, monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1),
           LearningRateScheduler(cosine),
           EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)]
    model.fit(tr_seq, initial_epoch=EPOCHS, epochs=EPOCHS+FINE_TUNE_EPOCHS, validation_data=va_seq, callbacks=cb_ft, class_weight=dict(enumerate(cw)))

    # Evaluation with TTA
    def tta(seq, n=8):
        ps=[tf.nn.softmax(model.predict(seq, verbose=0)).numpy() for _ in range(n)]
        return np.mean(ps,0)
    probs=tta(va_seq)
    best,th=-1,None
    for t in np.linspace(0.3,0.7,25):
        preds=np.argmax(probs+t*np.eye(NUM_CLASSES),1)
        kappa=cohen_kappa_score(val_df['level'], preds, weights='quadratic')
        if kappa>best: best,th=kappa,t
    logger.info('%s best QWK %.4f (thr %.2f)', name,best,th)
    with open(os.path.join(OUTPUT_DIR,f'val_kappa_{name}.json'),'w') as fp:
        json.dump({'kappa':best,'threshold':th},fp)
    results.append({'Model':name,'Kappa':best})

pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR,'summary_qwk.csv'), index=False)
logger.info('\u2714 Finished all models. Results saved.')
