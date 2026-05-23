# ♻️ Waste Product Classification Using Transfer Learning (VGG16)
### Two-stage fine-tuning · TensorFlow · Keras · Binary Classification · Sustainability AI

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow)
![VGG16](https://img.shields.io/badge/VGG16-ImageNet_Pretrained-purple?style=flat-square)
![Task](https://img.shields.io/badge/Task-Binary_Classification-green?style=flat-square)
![License](https://img.shields.io/badge/License-Educational-lightgrey?style=flat-square)

> Classifies waste into Organic and Recyclable categories using a two-stage VGG16 transfer learning strategy — feature extraction followed by selective fine-tuning of block5 layers — built for real-world automated waste sorting applications.

---

## 🎯 Why Two-Stage Training?

Most transfer learning tutorials stop at feature extraction — freeze the backbone, train the head, done. This project goes further with **selective fine-tuning**: after feature extraction converges, the deeper VGG16 convolutional blocks (block5) are unfrozen and retrained at a lower learning rate. This adapts ImageNet features to the waste domain, improving generalization beyond what a frozen backbone achieves.

---

## 🏗️ Model Architecture

```
Input (150 × 150 × 3)
        │
        ▼
┌───────────────────────────────┐
│   VGG16 Backbone              │
│   (ImageNet pretrained)       │
│   include_top=False           │
│                               │
│   Stage 1: All layers frozen  │
│   Stage 2: block5 unfrozen    │
└───────────┬───────────────────┘
            │
            ▼
        Flatten
            │
        Dense(512, ReLU) → Dropout(0.3)
            │
        Dense(512, ReLU) → Dropout(0.3)
            │
        Dense(1, Sigmoid)
            │
            ▼
    Organic (O) / Recyclable (R)
```

---

## 🔬 Training Strategy

### Stage 1 — Feature Extraction
| Setting | Value |
|---------|-------|
| Backbone | VGG16 (all layers frozen) |
| Optimizer | Adam |
| LR Schedule | Exponential decay |
| Callbacks | Early stopping · Model checkpointing |
| Goal | Rapid convergence, no overfitting |

### Stage 2 — Fine-Tuning
| Setting | Value |
|---------|-------|
| Unfrozen Layers | VGG16 block5 (deeper conv layers) |
| Optimizer | RMSprop |
| Learning Rate | Lower than Stage 1 — stable fine-tuning |
| Goal | Domain adaptation, improved generalization |

**Why RMSprop for fine-tuning?** Lower, more stable gradient updates when retraining pretrained weights — prevents catastrophic forgetting of ImageNet features.

---

## 📊 Results

| Model | Performance |
|-------|------------|
| Feature Extraction only | Strong baseline accuracy |
| Fine-Tuned (block5 unfrozen) | **Outperforms** — better generalization on test data |

Evaluated on unseen test images using accuracy, precision, recall, F1-score, and full classification reports for both stages.

> ℹ️ Add your specific accuracy numbers here after running the notebook — e.g. "Feature extraction: 91% → Fine-tuned: 94%"

---

## 🗂 Dataset Structure

```
dataset/
├── train/
│   ├── O/          # Organic waste images
│   └── R/          # Recyclable waste images
└── test/
    ├── O/
    └── R/
```

- Images resized to `150 × 150`
- Pixel values normalized to `[0, 1]`
- Augmentation via Keras `ImageDataGenerator`

> Dataset source excluded for licensing compliance. A placeholder URL is provided in the notebook.

---

## ⚙️ Tech Stack

| Tool | Purpose |
|------|---------|
| TensorFlow / Keras | Model building, training, callbacks |
| VGG16 (ImageNet) | Pretrained backbone |
| Keras ImageDataGenerator | Augmentation + preprocessing |
| NumPy | Numerical operations |
| Matplotlib | Training curves + predictions visualization |
| Scikit-learn | Classification reports, metrics |

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/amarkumar55/waste-product-classification-vgg16.git
cd waste-product-classification-vgg16

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook waste_classification_vgg16.ipynb
```

Place your dataset in the `dataset/` folder following the structure above before running.

---

## 📁 Repository Structure

```
waste-product-classification-vgg16/
│
├── dataset/
│   ├── train/
│   └── test/
├── vgg16_feature_extraction.keras    # Stage 1 saved weights
├── vgg16_fine_tuned.keras            # Stage 2 saved weights (best model)
├── waste_classification_vgg16.ipynb  # Full pipeline notebook
├── requirements.txt
└── README.md
```

---

## 🌍 Real-World Applications

- Automated waste sorting conveyor systems
- Smart recycling bins with embedded vision
- Municipal waste management automation
- Industrial waste compliance monitoring
- Environmental impact reporting systems

---

## 🔭 Roadmap

- [ ] Add accuracy numbers for both stages after final evaluation
- [ ] Multi-class extension (paper, glass, metal, plastic, etc.)
- [ ] Grad-CAM visualization — show what the model focuses on
- [ ] FastAPI deployment for real-time image inference
- [ ] Docker containerization for edge device deployment

---

## 👤 Author

**Amar Kumar** — Senior Backend Engineer · IBM Certified AI Engineer  
📌 [LinkedIn](https://www.linkedin.com/in/amarkumar241429017) · 💻 [GitHub](https://github.com/amarkumar55)

---

*Two-stage transfer learning for sustainability — because frozen backbones are just the starting point.*
