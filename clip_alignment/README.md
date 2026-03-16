# Cross-Modal Representation Alignment

## 1. Training

Train the model for **Cross-Modal Representation Alignment**:

```bash
python train_clip.py
```

## 2. Memory Bank Construction

After training, construct the **Memory Bank** by extracting image and text features.

### Image Feature Extraction

```bash
python compute_image_feature.py
```

### Text Feature Extraction

```bash
python compute_text_feature.py
```

The extracted image and text features together form the **Memory Bank**.
