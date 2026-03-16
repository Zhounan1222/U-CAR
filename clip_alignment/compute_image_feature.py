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
with open("/data/mimic_cxr/annotation.json", "r") as f:
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
visual_encoder = SwinModel.from_pretrained('swin-base-patch4-window7-224').to(device)

# Load pretrained weights
new_dict = {}
for k, v in checkpoint_model.items():
    if "visual_encoder." in k:
        new_dict[k.replace("visual_encoder.", "")] = v
visual_encoder.load_state_dict(new_dict, strict=False)

# Initialize projection layer
vision_proj = nn.Linear(visual_encoder.num_features, 768).to(device)

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

def encode_img_batch(image_batch, counts):
    """Process a batch of images with variable counts per sample"""
    # Process all images together
    pixel_values = parse_image_batch(image_batch)
    
    with torch.no_grad():
        # Get embeddings for all images
        embeddings = visual_encoder(pixel_values)['last_hidden_state']
        embeddings = embeddings.mean(dim=1)  # Average over spatial dimensions
        embeddings = vision_proj(embeddings)
        
        # Normalize features
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
        
        # Split back into original samples
        split_embeddings = torch.split(embeddings, counts)
        
        # Average embeddings for each sample's images
        sample_features = [torch.mean(group, dim=0) for group in split_embeddings]
        
    return torch.stack(sample_features)

# Create dataset and dataloader
image_dataset = ImageDataset(list(image_dict.keys()), image_dict)
image_loader = DataLoader(
    image_dataset,
    batch_size=512,  # Adjust based on GPU memory
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)

# Process all images in batches
image_feature_dict = {}
for batch in tqdm(image_loader, desc="Processing images"):
    # Convert numpy arrays to PIL Images for processing
    pil_images = [Image.fromarray(img) for img in batch['image_arrays']]
    
    # Process batch
    features = encode_img_batch(pil_images, batch['image_counts'])
    
    # Move to CPU and convert to list
    features = features.cpu().numpy()
    
    # Store results
    for img_id, feat in zip(batch['image_ids'], features):
        image_feature_dict[img_id] = feat.tolist()

# Save all image features
with open("image_features_mimic.json", "w") as f:
    json.dump(image_feature_dict, f)

print("Image feature extraction completed!")