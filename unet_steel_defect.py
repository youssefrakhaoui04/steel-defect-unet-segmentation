import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# ==========================================
# 1. CONSTRUCTION DE L'ARCHITECTURE U-NET
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
# 2. INITIALISATION ET COMPILATION DU MODÈLE
# ==========================================
# Image d'entrée standardisée en 256x256 pixels en grayscale (ou RGB -> 3 canaux)
model = unet_model(input_shape=(256, 256, 1))

model.compile(
    optimizer='adam', 
    loss='binary_crossentropy', 
    metrics=['accuracy', tf.keras.metrics.IoU(num_classes=2, target_class_ids=[1])]
)

model.summary()

# ==========================================
# 3. SIMULATION D'ENTRAÎNEMENT (Structure type)
# ==========================================
print("\n--- Pipeline prêt pour l'entraînement ---")
print("Remplace ces données factices par ton dataset de plaques d'acier (ex: Kaggle Severstal)")

# Exemple de génération de données factices pour tester l'exécution du script :
# X_train = np.random.rand(100, 256, 256, 1)  # 100 images d'acier
# y_train = np.randint(0, 2, size=(100, 256, 256, 1)) # 100 masques de fissures
# model.fit(X_train, y_train, batch_size=16, epochs=10)

# ==========================================
# 4. MESURE DU TEMPS D'INFÉRENCE (< 30 ms)
# ==========================================
import time

dummy_image = np.random.rand(1, 256, 256, 1).astype(np.float32)

# Préchauffage du modèle
_ = model.predict(dummy_image)

start_time = time.time()
_ = model.predict(dummy_image)
inference_time_ms = (time.time() - start_time) * 1000

print(f"\nTemps d'inférence mesuré : {inference_time_ms:.2f} ms par image")
if inference_time_ms < 30:
    print("Objectif d'inférence en temps réel atteint (< 30 ms) !")
else:
    print("Attention : optimiser le modèle (quantification TFLite) pour passer sous les 30 ms.")