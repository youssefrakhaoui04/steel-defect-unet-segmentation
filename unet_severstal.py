import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# ==========================================
# 1. CONSTRUCTION DE L'ARCHITECTURE U-NET   (ton code, inchangé)
# ==========================================
def unet_model(input_shape=(256, 256, 1)):
    inputs = layers.Input(input_shape)

    # --- ENCODER (Descente / Extraction de caractéristiques) ---
    # Bloc 1
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    # Bloc 2
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Bloc 3 (Bottleneck / Passage au plus bas niveau)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # --- LATENT SPACE (Fond du U) ---
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)

    # --- DECODER (Remontée / Reconstruction spatiale) ---
    # Bloc 3 Décodeur + Skip Connection (c3)
    u3 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c4)
    u3 = layers.concatenate([u3, c3])
    c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u3)
    c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c5)

    # Bloc 2 Décodeur + Skip Connection (c2)
    u2 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u2 = layers.concatenate([u2, c2])
    c6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u2)
    c6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c6)

    # Bloc 1 Décodeur + Skip Connection (c1)
    u1 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u1 = layers.concatenate([u1, c1])
    c7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u1)
    c7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c7)

    # Sortie : Masque binaire (fissure ou non par pixel)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c7)

    model = models.Model(inputs=[inputs], outputs=[outputs], name="U-Net_Steel_Defect")
    return model

# ==========================================
# 2. INITIALISATION ET COMPILATION DU MODÈLE   (ton code)
# ==========================================
# Image d'entrée standardisée en 256x256 pixels en grayscale
model = unet_model(input_shape=(256, 256, 1))

# Seule modification : BinaryIoU à la place de IoU. Avec une sortie sigmoïde (valeurs entre 0 et 1),
# IoU(num_classes=2) tronque les probabilités vers 0 et affiche ~0 en permanence ; BinaryIoU applique
# le seuil de 0.5 et donne la vraie valeur.
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5, name='iou')]
)

model.summary()


# ==========================================================================================
# 3. IMPORTATION DE LA BASE DE DONNÉES SEVERSTAL  (ajout)
# ==========================================================================================
import pandas as pd
from PIL import Image

import sys, zipfile
ON_COLAB = "google.colab" in sys.modules
GPU = bool(tf.config.list_physical_devices("GPU"))
print("Environnement :", "Google Colab" if ON_COLAB else "ordinateur local", "| GPU :", "oui" if GPU else "non")

# --- Réglages à adapter -----------------------------------------------------------------
# SPYDER (ordinateur local) : laisse DATA_ROOT = None si le dossier des données est quelque part
# sous le dossier du script (il est trouvé tout seul) ; sinon écris son chemin complet.
DATA_ROOT = None      # ex. r"C:\Users\toi\Documents\severstal-steel-defect-detection"
# COLAB : zip des données (le dossier severstal-steel-defect-detection compressé) placé sur ton Google Drive.
DRIVE_ZIP = "/content/drive/MyDrive/severstal.zip"
OUT_DIR = "/content/drive/MyDrive/resultats_unet" if ON_COLAB else "resultats_unet"   # sur Colab : Drive, pour garder les sauvegardes

# Tailles : plus grandes si un GPU est disponible
MAX_TRAIN_TILES = 6000 if GPU else 1500   # tuiles 256x256 pour l'entraînement
MAX_VAL_TILES = 1500 if GPU else 400
MAX_TEST_IMAGES = 400 if GPU else 150     # images entières (6 tuiles chacune) pour l'évaluation finale
EPOCHS = 20              # nombre TOTAL d'époques visé (l'entraînement peut être arrêté et repris, voir section 4)
BATCH = 16 if GPU else 8
USE_BCE_DICE = True      # True : perte BCE + Dice (adaptée aux défauts très minoritaires) ; False : ta BCE d'origine
REPARTIR_DE_ZERO = False # True : ignore les sauvegardes précédentes et recommence l'entraînement à zéro
SEED = 42
# -----------------------------------------------------------------------------------------
np.random.seed(SEED); tf.random.set_seed(SEED)

if ON_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
os.makedirs(OUT_DIR, exist_ok=True)

try:
    BASE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE = os.getcwd()


def find_data_root(bases, max_depth=3):
    """Cherche un dossier qui contient train.csv ET train_images/ (sans parcourir Google Drive)."""
    for base in bases:
        if not os.path.isdir(base):
            continue
        base_depth = base.rstrip(os.sep).count(os.sep)
        for dirpath, dirnames, files in os.walk(base):
            if dirpath.count(os.sep) - base_depth >= max_depth:
                dirnames[:] = []
            dirnames[:] = [d for d in dirnames if d not in ("drive", "sample_data", ".git", "__pycache__", "resultats_unet")]
            if "train.csv" in files and os.path.isdir(os.path.join(dirpath, "train_images")):
                return dirpath
    return None


if DATA_ROOT and not os.path.exists(os.path.join(DATA_ROOT, "train.csv")):
    raise FileNotFoundError(f"train.csv introuvable dans DATA_ROOT = {DATA_ROOT}")
if DATA_ROOT is None:
    DATA_ROOT = find_data_root([BASE, "/content/severstal_data"])
if DATA_ROOT is None and os.path.exists(DRIVE_ZIP):      # Colab : décompression sur le disque local (bien plus rapide que Drive)
    print("Décompression des données depuis", DRIVE_ZIP, "(quelques minutes)...")
    zipfile.ZipFile(DRIVE_ZIP).extractall("/content/severstal_data")
    DATA_ROOT = find_data_root(["/content/severstal_data"])
if DATA_ROOT is None:
    raise FileNotFoundError(
        "Données introuvables. Spyder : mets le dossier des données sous celui du script ou renseigne DATA_ROOT. "
        f"Colab : place le zip des données à {DRIVE_ZIP} (ou change DRIVE_ZIP).")
IMG_DIR = os.path.join(DATA_ROOT, "train_images")
print("Données :", DATA_ROOT)

FULL_H, FULL_W, TILE = 256, 1600, 256
N_TILES = FULL_W // TILE          # 6 tuiles de 256x256 par image (les 64 derniers pixels de droite sont ignorés)

# train.csv : une ligne par (image, type de défaut) ; masque codé en RLE.
# Les images absentes du CSV n'ont aucun défaut.
df = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"))
rle = {}
for img_id, enc in zip(df.ImageId, df.EncodedPixels):
    rle.setdefault(img_id, []).append(enc)          # on fusionne les 4 types : masque binaire "défaut / pas défaut"

all_ids = sorted(f for f in os.listdir(IMG_DIR) if f.lower().endswith(".jpg"))
print(f"{len(all_ids)} images, dont {len(rle)} avec au moins un défaut")


def rle_to_mask(enc_list):
    """Décode les RLE (ordre colonne par colonne) en un masque binaire 256x1600."""
    mask = np.zeros(FULL_H * FULL_W, dtype=np.uint8)
    for enc in enc_list:
        a = np.asarray(enc.split(), dtype=np.int64)
        for s, l in zip(a[0::2] - 1, a[1::2]):
            mask[s:s + l] = 1
    return mask.reshape((FULL_H, FULL_W), order="F")


def load_image_and_mask(img_id):
    img = np.asarray(Image.open(os.path.join(IMG_DIR, img_id)).convert("L"), dtype=np.uint8)
    return img, rle_to_mask(rle.get(img_id, []))


def tiles_of(img, mask):
    return ([img[:, k * TILE:(k + 1) * TILE] for k in range(N_TILES)],
            [mask[:, k * TILE:(k + 1) * TILE] for k in range(N_TILES)])


# --- Découpage par IMAGE (pas par tuile) : aucune fuite entre entraînement / validation / test ---
ids = np.array(all_ids); np.random.shuffle(ids)
n = len(ids)
train_ids, val_ids, test_ids = ids[:int(.70 * n)], ids[int(.70 * n):int(.85 * n)], ids[int(.85 * n):]
print("images train / val / test :", len(train_ids), len(val_ids), len(test_ids))


def build_balanced_tiles(id_list, max_tiles):
    """Garde toutes les tuiles contenant un défaut + autant de tuiles sans défaut (sinon le
    modèle apprend à toujours répondre 'pas de défaut', car les défauts sont très minoritaires)."""
    pos, neg = [], []
    for f in id_list:
        img, mask = load_image_and_mask(f)
        for x, y in zip(*tiles_of(img, mask)):
            (pos if y.any() else neg).append((x, y))
        if len(pos) >= max_tiles // 2 and len(neg) >= max_tiles // 2:
            break
    k = min(len(pos), max_tiles // 2, len(neg))
    np.random.shuffle(pos); np.random.shuffle(neg)
    sel = pos[:k] + neg[:k]
    np.random.shuffle(sel)
    X = np.stack([s[0] for s in sel])[..., None]          # uint8 (N,256,256,1)
    Y = np.stack([s[1] for s in sel])[..., None]
    print(f"  {len(X)} tuiles ({k} avec défaut, {k} sans)")
    return X, Y


print("Construction des tuiles d'entraînement..."); X_train, Y_train = build_balanced_tiles(train_ids, MAX_TRAIN_TILES)
print("Construction des tuiles de validation...");  X_val, Y_val = build_balanced_tiles(val_ids, MAX_VAL_TILES)


class TileSequence(tf.keras.utils.Sequence):
    """Sert des lots normalisés (0-1) ; augmentation par symétries pour l'entraînement."""
    def __init__(self, X, Y, batch=BATCH, augment=False):
        super().__init__()
        self.X, self.Y, self.batch, self.augment = X, Y, batch, augment
        self.idx = np.arange(len(X))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch))

    def on_epoch_end(self):
        if self.augment:
            np.random.shuffle(self.idx)

    def __getitem__(self, i):
        b = self.idx[i * self.batch:(i + 1) * self.batch]
        x, y = self.X[b].astype(np.float32) / 255.0, self.Y[b].astype(np.float32)
        if self.augment:
            if np.random.rand() < 0.5: x, y = x[:, :, ::-1], y[:, :, ::-1]
            if np.random.rand() < 0.5: x, y = x[:, ::-1], y[:, ::-1]
        return np.ascontiguousarray(x), np.ascontiguousarray(y)


train_seq = TileSequence(X_train, Y_train, augment=True)
val_seq = TileSequence(X_val, Y_val)


# ==========================================================================================
# 4. ENTRAÎNEMENT  (remplace la "simulation" avec données factices)
# ==========================================================================================
print("GPU détecté :", tf.config.list_physical_devices("GPU") or "aucun (entraînement sur processeur : lent)")


def dice_coef(y_true, y_pred, smooth=1.0):
    y_true = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    inter = tf.reduce_sum(y_true * y_pred)
    return (2.0 * inter + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)


def bce_dice_loss(y_true, y_pred):
    # BCE seule : le modèle peut obtenir une bonne perte en prédisant "pas de défaut" partout.
    # Le terme Dice force le modèle à bien recouvrir les pixels de défaut, même s'ils sont rares.
    bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
    return bce + (1.0 - dice_coef(y_true, y_pred))


if USE_BCE_DICE:   # même modèle, même optimiseur : seule la perte change
    model.compile(
        optimizer='adam',
        loss=bce_dice_loss,
        metrics=['accuracy', tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5, name='iou'), dice_coef]
    )

# --- Sauvegardes : à chaque époque, et reprise automatique si l'entraînement a été interrompu ---
LAST_W = os.path.join(OUT_DIR, "unet_last.weights.h5")
BEST_W = os.path.join(OUT_DIR, "unet_best.weights.h5")
LOG_CSV = os.path.join(OUT_DIR, "historique.csv")
initial_epoch = 0
if REPARTIR_DE_ZERO:
    for p in (LAST_W, BEST_W, LOG_CSV):
        if os.path.exists(p): os.remove(p)
if os.path.exists(LAST_W) and os.path.exists(LOG_CSV):
    model.load_weights(LAST_W)
    initial_epoch = len(pd.read_csv(LOG_CSV))
    print(f"Reprise de l'entraînement après l'époque {initial_epoch} (sauvegarde trouvée).")

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(LAST_W, save_weights_only=True),                      # dernière époque
    tf.keras.callbacks.ModelCheckpoint(BEST_W, monitor="val_iou", mode="max",
                                       save_best_only=True, save_weights_only=True),         # meilleure époque
    tf.keras.callbacks.CSVLogger(LOG_CSV, append=True),                                      # historique complet
]
t0 = time.time()
if initial_epoch < EPOCHS:
    try:
        model.fit(train_seq, validation_data=val_seq, epochs=EPOCHS, initial_epoch=initial_epoch,
                  callbacks=callbacks, verbose=2)
    except KeyboardInterrupt:
        print("Entraînement interrompu à la main : les sauvegardes sont conservées, relance le script pour reprendre.")
    print(f"Durée de cette session : {(time.time() - t0) / 60:.1f} min")
else:
    print(f"Les {EPOCHS} époques sont déjà faites : passage direct aux résultats.")

if os.path.exists(BEST_W):
    model.load_weights(BEST_W)                      # on évalue le MEILLEUR modèle, pas le dernier
    print("Poids de la meilleure époque (val_iou) chargés pour l'évaluation.")


# ==========================================================================================
# 5. RÉSULTATS D'ENTRAÎNEMENT
# ==========================================================================================
hist = pd.read_csv(LOG_CSV)                         # toutes les sessions d'entraînement cumulées
keys = [k for k in ["loss", "accuracy", "iou", "dice_coef"] if k in hist.columns]
fig, ax = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4))
for a_, key in zip(np.atleast_1d(ax), keys):
    a_.plot(hist["epoch"] + 1, hist[key], label="entraînement")
    a_.plot(hist["epoch"] + 1, hist["val_" + key], label="validation")
    a_.set_title(key); a_.set_xlabel("Époque"); a_.legend()
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "courbes_entrainement.png"), dpi=120); plt.show()

print(f"\n--- Meilleure époque (validation, jeu équilibré) sur {len(hist)} époques ---")
best = int(hist["val_iou"].idxmax())
print(f"Époque {best + 1} : val_iou = {hist['val_iou'][best]:.3f} | "
      f"val_accuracy = {hist['val_accuracy'][best]:.3f} | val_loss = {hist['val_loss'][best]:.3f}")


# --- Évaluation finale sur le TEST : images jamais vues, avec la proportion NATURELLE de défauts ---
def evaluate_test(predict_fn, thr=0.5, n_images=MAX_TEST_IMAGES):
    inter = pred_sum = true_sum = 0
    tp = fp = fn = tn = 0                            # niveau tuile : y a-t-il un défaut dans la tuile ?
    for f in test_ids[:n_images]:
        img, mask = load_image_and_mask(f)
        xs, ys = tiles_of(img, mask)
        x = (np.stack(xs)[..., None].astype(np.float32) / 255.0)
        p = (predict_fn(x) >= thr).astype(np.uint8)
        y = np.stack(ys)[..., None]
        inter += (p * y).sum(); pred_sum += p.sum(); true_sum += y.sum()
        pt = p.reshape(len(p), -1).any(1); yt = y.reshape(len(y), -1).any(1)
        tp += (pt & yt).sum(); fp += (pt & ~yt).sum(); fn += (~pt & yt).sum(); tn += (~pt & ~yt).sum()
    union = pred_sum + true_sum - inter
    return dict(dice=(2 * inter + 1e-7) / (pred_sum + true_sum + 1e-7), iou=(inter + 1e-7) / (union + 1e-7),
                precision_tuile=tp / max(tp + fp, 1), rappel_tuile=tp / max(tp + fn, 1), fp=fp, fn=fn, tp=tp, tn=tn)


keras_predict = lambda x: model.predict(x, batch_size=BATCH, verbose=0)

# Seuil choisi sur la validation (pas sur le test)
vp = model.predict(X_val.astype(np.float32) / 255.0, batch_size=BATCH, verbose=0)
def val_dice(thr):
    p = (vp >= thr).astype(np.uint8); g = Y_val
    return (2 * (p * g).sum() + 1e-7) / (p.sum() + g.sum() + 1e-7)
THR = max([0.3, 0.4, 0.5, 0.6, 0.7], key=val_dice)
print(f"\nSeuil choisi sur la validation : {THR}")

r = evaluate_test(keras_predict, THR)
print(f"\n=== RÉSULTATS SUR LE TEST ({min(MAX_TEST_IMAGES, len(test_ids))} images jamais vues) ===")
print(f"Dice = {r['dice']:.3f} | IoU = {r['iou']:.3f}")
print(f"Niveau tuile : précision = {r['precision_tuile']:.2f} | rappel = {r['rappel_tuile']:.2f} "
      f"(vrais positifs {r['tp']}, faux positifs {r['fp']}, faux négatifs {r['fn']}, vrais négatifs {r['tn']})")


# ==========================================
# 6. MESURE DU TEMPS D'INFÉRENCE   (ton code, conservé à titre indicatif)
# ==========================================
dummy_image = np.random.rand(1, 256, 256, 1).astype(np.float32)
_ = model.predict(dummy_image, verbose=0)           # préchauffage
start_time = time.time()
_ = model.predict(dummy_image, verbose=0)
inference_time_ms = (time.time() - start_time) * 1000
print(f"\nTemps d'inférence Keras mesuré : {inference_time_ms:.2f} ms par tuile (sur cet ordinateur)")


# ==========================================
# 7. CONVERSION TFLITE ET LATENCE (la mesure qui compte pour le CV : à refaire sur le Raspberry Pi)
# ==========================================
def convert(mode):
    try:
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
    except Exception:
        model.export(os.path.join(OUT_DIR, "saved_model"))
        conv = tf.lite.TFLiteConverter.from_saved_model(os.path.join(OUT_DIR, "saved_model"))
    if mode == "float16":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]; conv.target_spec.supported_types = [tf.float16]
    elif mode == "int8":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        def rep():
            for i in range(min(100, len(X_train))):
                yield [(X_train[i:i + 1].astype(np.float32) / 255.0)]
        conv.representative_dataset = rep
    return conv.convert()


for mode in ["float32", "float16", "int8"]:
    try:
        blob = convert(mode)
        path = os.path.join(OUT_DIR, f"unet_{mode}.tflite")
        open(path, "wb").write(blob)
        interp = tf.lite.Interpreter(model_path=path, num_threads=4); interp.allocate_tensors()
        inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
        x0 = dummy_image.astype(inp["dtype"])
        for _ in range(3): interp.set_tensor(inp["index"], x0); interp.invoke()
        ts = []
        for _ in range(20):
            t = time.perf_counter(); interp.set_tensor(inp["index"], x0); interp.invoke()
            ts.append((time.perf_counter() - t) * 1000)

        def tfl_predict(x, interp=interp, inp=inp, out=out):
            res = []
            for im in x:
                interp.set_tensor(inp["index"], im[None].astype(inp["dtype"])); interp.invoke()
                res.append(interp.get_tensor(out["index"])[0])
            return np.stack(res)

        rr = evaluate_test(tfl_predict, THR, n_images=min(50, MAX_TEST_IMAGES))
        print(f"{mode:8s} {len(blob) / 1e6:7.1f} Mo | latence médiane {np.median(ts):7.1f} ms/tuile (cet ordinateur) | "
              f"Dice test (50 images) = {rr['dice']:.3f}")
    except Exception as e:
        print(f"{mode}: conversion impossible ({type(e).__name__}: {str(e)[:100]})")
print("\nCopie les fichiers .tflite sur le Raspberry Pi et lance bench_tflite_rpi.py pour la vraie latence.")
