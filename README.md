#  Image Segmentation Model Benchmarking & U-Net Enhancements

This repository contains an in-depth comparative study of several state-of-the-art **image segmentation architectures**, with a focused effort on enhancing **U-Net**, the top-performing baseline model.

The work is done as part of the CS604 project and includes architectural benchmarking, performance evaluation, and experiments with U-Net improvements like **U-Net++**, **attention mechanisms**, **ensembles**, and **Transformer-based hybrids**.

---

## 🎯 Objective

The project aims to:

1. **Benchmark popular segmentation backbones** on a common dataset.
2. **Evaluate segmentation quality** using metrics like Dice Score and IoU.
3. **Explore U-Net variants** to boost performance for medical image segmentation tasks.

---

##  Architectures Explored

| Architecture         | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| **U-Net**            | Baseline encoder-decoder with skip connections                              |
| **DenseNet**         | Encoder with dense connectivity                                             |
| **SegNet**           | Symmetric encoder-decoder with max-pooling indices                          |
| **DeepLabV3+**       | Atrous Spatial Pyramid Pooling (ASPP) for multiscale context                |
| **SegFormer**        | Lightweight Transformer backbone with MLP decoder                          |

> Result: **U-Net consistently outperformed** other architectures across multiple datasets in terms of Dice and IoU scores.

---

##  U-Net Variants Implemented

| Variant                          | Purpose                                                       |
|----------------------------------|---------------------------------------------------------------|
| `unet-dice (final).ipynb`        | Base U-Net with Dice loss for improved class imbalance        |
| `cs604-project-unetplusplusensemble-v7.ipynb` | Ensemble of U-Net++ variants to boost generalization  |
| `cs604-project-unetplusplus-attention-v4.ipynb` | U-Net++ + attention gates for focused segmentation    |
| `cs604-transattunet_gaussian.ipynb` | Transformer-style attention on U-Net + Gaussian smoothing |

Each variant was evaluated against the baseline on the same validation splits.

---

## 📁 Project Structure
```
├── unet-dice (final).ipynb # 🔹 Baseline U-Net with Dice loss
├── cs604-project-unetplusplusensemble-v7.ipynb # 🔹 U-Net++ ensemble model 
├── cs604-project-unetplusplus-attention-v4.ipynb # 🔹 Attention-enhanced U-Net++ 
├── cs604-transattunet_gaussian.ipynb # 🔹 Transformer-inspired U-Net variant 
```
---

## Evaluation Metrics

All models were evaluated using:

- **Dice Coefficient** (primary loss and evaluation metric)
- **Intersection over Union (IoU)**
- **Pixel Accuracy**

---

##  Key Insights

- **U-Net** remains a strong baseline for medical image segmentation.
- **U-Net++** improves boundary precision and robustness.
- **Attention mechanisms** further help in segmenting smaller regions and reduce false positives.
- **Transformer-style blocks** offer potential but require tuning and longer training.

---

##  Dependencies

This project uses:
- `PyTorch`, `Torchvision`
- `Albumentations`, `OpenCV`, `Segmentation Models PyTorch`
- `scikit-learn`, `matplotlib`, `seaborn`

Install using:
```bash
pip install -r requirements.txt
