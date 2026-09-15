# 👁️ Steel Defect Segmentation & Visual Quality Control (U-Net)

## 📌 À propos du projet
Ce projet implémente un modèle de **Deep Learning (Vision par Ordinateur)** basé sur une architecture **U-Net** pour la segmentation sémantique et la localisation automatique de micro-fissures sur des plaques d'acier en milieu industriel.

---

## 🚀 Fonctionnalités & Performance
1. **Architecture U-Net :** Modèle de plus de 7,6 millions de paramètres structuré avec des blocs encodeur-décodeur et des connexions de saut (*skip connections*) pour préserver la précision spatiale des défauts.
2. **Entraînement :** Réalisé sous TensorFlow/Keras avec des métriques de suivi par intersection sur union (IoU) et binaire.
3. **Optimisation Temps Réel :** Pipeline d'inférence préparé pour l'export au format **TensorFlow Lite (TFLite)** avec quantification pour descendre sous la barre des 30 ms par image (exigence de la ligne de production).

---

## 🛠️ Technologies utilisées
* **Langage :** Python 3
* **Framework :** TensorFlow / Keras, NumPy, Matplotlib
