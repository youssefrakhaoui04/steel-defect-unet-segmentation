# Steel surface defect segmentation with a U-Net

Pixel-wise segmentation of defects on steel plates (Severstal dataset) with a U-Net written from scratch in TensorFlow/Keras (7.7 M parameters), followed by TFLite conversion for embedded use.

## Dataset
[Severstal: Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection) (Kaggle). 12,568 images of 256x1600 px, 6,666 with at least one defect, masks in RLE format (`train.csv`).
**The data is not included in this repository** (Kaggle rules forbid redistribution). Download it yourself and point `DATA_ROOT` in the script to the folder containing `train.csv` and `train_images/`.

## Method
- Each image is cut into 6 tiles of 256x256.
- Split **by image** (70 / 15 / 15 %) to avoid leakage between train, validation and test.
- Balanced tile sampling for training (defects are a small minority of pixels).
- Loss: BCE + Dice. Metrics: Dice and binary IoU.
- Decision threshold chosen on the validation set, final evaluation on test tiles with the natural class distribution.
- Resumable training (per-epoch checkpoints + CSV log), so it survives Colab disconnections.
- Export to TFLite (float32, float16, int8).

## Results (400 unseen test images, run on Colab GPU)
| Metric | Value |
|---|---|
| Dice | 0.60 |
| IoU | 0.42 |
| Tile-level recall | 94 % |
| Tile-level precision | 41 % |

TFLite float16: no measurable accuracy loss compared with float32.

## Limitations
- Precision is modest (41 %): the model finds almost all defective tiles but raises many false alarms. Raising the threshold or training longer would trade some recall for precision.
- TFLite int8 quantization failed (Dice ~0.005) with my calibration setup, so it is not usable as is.
- Inference latency on embedded hardware (e.g. Raspberry Pi) has **not been measured yet**.
- Only a subset of tiles is used for training (6,000 on GPU), so the results are not a full-dataset benchmark.

## Usage
```bash
pip install -r requirements.txt
# edit DATA_ROOT at the top of unet_severstal.py, then:
python unet_severstal.py
```
On Google Colab the script mounts Drive, unzips `severstal.zip` and saves weights/results to Drive. Outputs (weights, `historique.csv`, training curves, `.tflite` files) go to `resultats_unet/`.
