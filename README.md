# Low-Light Image Enhancement using Hybrid Loss and Grad-CAM

## 📌 Overview

This project enhances low-light images using a deep learning model based on U-Net.
A hybrid loss function combining spatial, perceptual, and frequency constraints is proposed.

---

## 🚀 Key Features

* U-Net based image enhancement
* Hybrid loss (L1 + SSIM + Perceptual + Frequency)
* Ablation study (Baseline vs Perceptual vs Hybrid)
* Quantitative evaluation using PSNR and SSIM
* Visual comparison of outputs
* Grad-CAM for model explainability

---

## 📊 Results

| Model      | PSNR (dB) | SSIM       |
| ---------- | --------- | ---------- |
| Baseline   | 25.13     | 0.8735     |
| Perceptual | 19.83     | 0.8110     |
| Hybrid     | **25.57** | **0.8860** |

---

## 🧠 Key Insight

Perceptual loss alone degrades performance, while combining it with spatial and frequency losses improves results.

---

## 📁 Project Structure

```
src/
 ├── model.py
 ├── train.py
 ├── evaluate.py
 ├── visualize.py
 ├── gradcam.py
 ├── data.py
```

---

## ▶️ How to Run

### Train

```
python -m src.train
```

### Evaluate

```
python -m src.evaluate
```

### Visualize Results

```
python src/visualize.py
```

### Grad-CAM

```
python src/gradcam.py
```

---

## 📌 Conclusion

The hybrid loss improves both quantitative metrics (PSNR, SSIM) and visual quality.
Grad-CAM confirms improved attention to low-light and structural regions.

---

## 📦 Requirements

```
torch
torchvision
numpy
matplotlib
opencv-python
pytorch-msssim
```
