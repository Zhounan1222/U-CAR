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


## 3. Cross-Modal Retrieval
Image → Report

Perform cross-modal retrieval from image to report using FAISS:
```bash
python faissi2t.py
```
This script retrieves the most relevant reports for a given image by searching the text feature memory bank.

Report → Image

Retrieval from report to image is implemented in a similar way.
