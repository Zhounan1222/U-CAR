import json
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor, SwinModel
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import re
from PIL import Image
import os
import numpy as np
from tqdm import tqdm

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize image processor and move to GPU
vit_feature_extractor = AutoImageProcessor.from_pretrained('swin-base-patch4-window7-224')

def parse_image_batch(img_batch):
    pixel_values = vit_feature_extractor(img_batch, return_tensors="pt", size=224).pixel_values
    return pixel_values.to(device)

# Load annotation data
with open("./mimic_cxr/annotation.json", "r") as f:
    annotation_data = json.load(f)

def clean_report(report):
    report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
        .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
        .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
        .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
        .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
        .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
        .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ').replace(':', ' :') \
        .strip().lower().split('. ')
    sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+()\[\]{}]', '', t.replace('"', '').replace('/', '')
                        .replace('\\', '').replace("'", '').strip().lower())
    tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
    return ' . '.join(tokens) + ' .'

# Create dataset class for image processing
class ImageDataset(Dataset):
    def __init__(self, image_ids, image_dict, image_root='./mimic_cxr/images'):
        self.image_ids = list(image_ids)
        self.image_dict = image_dict
        self.image_root = image_root
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_paths = self.image_dict[image_id]
        images = []
        
        for img_path in image_paths:
            try:
                with Image.open(os.path.join(self.image_root, img_path)) as pil:
                    array = np.array(pil, dtype=np.uint8)
                    if array.shape[-1] != 3 or len(array.shape) != 3:
                        array = np.array(pil.convert("RGB"), dtype=np.uint8)
                    images.append(array)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                # Return blank image if loading fails
                images.append(np.zeros((224, 224, 3), dtype=np.uint8))
        
        return {
            'image_id': image_id,
            'image_arrays': images  # List of numpy arrays
        }

# Extract all image IDs and paths
report_dict = {}
image_dict = {}

for split in ["train", "val", "test"]:
    for entry in annotation_data[split]:
        image_id = entry["id"]
        report_dict[image_id] = clean_report(entry["report"])
        image_dict[image_id] = entry['image_path']

print(f"Total images: {len(image_dict)}")

# Load model pretrain clip checkpoint
checkpoint = torch.load(
    'train_clip.pt',
    map_location=device
)
checkpoint_model = checkpoint['model']

# Initialize visual encoder
tokenizer = AutoTokenizer.from_pretrained("Bio_ClinicalBERT")
if tokenizer.bos_token_id is None:
    tokenizer.bos_token_id = tokenizer.cls_token_id
text_encoder = AutoModel.from_pretrained("Bio_ClinicalBERT")  
new_dict = {}
for k, v in checkpoint_model.items():
    if "text_encoder." in k:
        new_dict[k.replace("text_encoder.", "")] = v
# load pre-trained model
text_encoder.load_state_dict(new_dict,strict=False)
# 计算 Train Report 特征

# Initialize projection layer
text_proj = nn.Linear(text_encoder.config.hidden_size, 768)


# Extract train/val/test splits
train_images = set()
val_images = set()
test_images = set()
report_dict = {}
image_dict = {}

for split in ["train", "val", "test"]:
    for entry in annotation_data[split]:
        image_id = entry["id"]
        report_text = clean_report(entry["report"])
        report_dict[image_id] = report_text
        image_dict[image_id] = entry['image_path']

        if split == "train":
            train_images.add(image_id)
        elif split == "val":
            val_images.add(image_id)
        elif split == "test":
            test_images.add(image_id)

print(f"Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")

def collate_fn(batch):
    """Custom collate function to handle variable number of images per sample"""
    image_ids = [item['image_id'] for item in batch]
    all_images = []
    image_counts = []
    
    for item in batch:
        images = item['image_arrays']
        image_counts.append(len(images))
        all_images.extend(images)
    
    return {
        'image_ids': image_ids,
        'image_arrays': all_images,
        'image_counts': image_counts
    }
    
# Create dataset class for batch processing
class ReportDataset(Dataset):
    def __init__(self, image_ids, report_dict, tokenizer, max_length=128):
        self.image_ids = list(image_ids)
        self.reports = [report_dict[img_id] for img_id in image_ids]
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        report = self.reports[idx]
        tokens = self.tokenizer(
            report, 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        return {
            'input_ids': tokens['input_ids'].squeeze(),
            'attention_mask': tokens['attention_mask'].squeeze(),
            'image_id': self.image_ids[idx]
        }

batch_size = 512 # Adjust based on your GPU memory
train_dataset = ReportDataset(train_images, report_dict, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

def process_batch(batch):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    
    with torch.no_grad():
        text_features = text_encoder(input_ids, attention_mask=attention_mask)["last_hidden_state"]
        eos_token_indices = attention_mask.sum(dim=-1) - 1
        text_features = text_features[torch.arange(text_features.shape[0], device=device), eos_token_indices]
        text_features = text_proj(text_features)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
    
    return text_features.cpu().numpy(), batch['image_id']

# Process all reports in batches
train_report_feature_dict = {}
for batch_idx, batch in enumerate(train_loader):
    features, image_ids = process_batch(batch)
    
    for img_id, feature in zip(image_ids, features):
        train_report_feature_dict[img_id] = feature.tolist()
    
    print(f"Processed batch {batch_idx + 1}/{len(train_loader)}")

# Save features
with open("report_features.json", "w") as f:
    json.dump(train_report_feature_dict, f)

print("Feature extraction completed!")


