import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor
import yaml
import numpy as np
from pathlib import Path

# Custom Dataset to Load YOLOv8 Annotations
class ASLDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, yaml_file, processor):
        self.image_dir = Path(image_dir)
        self.annotation_dir = Path(annotation_dir)
        self.processor = processor

        # Load YAML config
        if not os.path.exists(yaml_file):
            raise FileNotFoundError(f"YAML file not found at: {yaml_file}")
        with open(yaml_file, 'r') as f:
            self.config = yaml.safe_load(f)
        self.classes = self.config['names']  # List of class names (A-Z)
        self.num_classes = len(self.classes)

        # Get image and annotation files
        self.image_files = sorted([f for f in self.image_dir.glob('*.jpg')])  # Adjust extension if needed
        self.annotation_files = sorted([f for f in self.annotation_dir.glob('*.txt')])

        if not self.image_files:
            raise ValueError(f"No images found in {self.image_dir}")
        if len(self.image_files) != len(self.annotation_files):
            raise ValueError(f"Mismatch: {len(self.image_files)} images, {len(self.annotation_files)} annotations")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = str(self.image_files[idx])
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]

        # Load YOLOv8 annotations
        ann_path = str(self.annotation_files[idx])
        boxes = []
        labels = []
        with open(ann_path, 'r') as f:
            for line in f:
                class_id, x_center, y_center, w, h = map(float, line.strip().split())
                # Convert YOLO format (normalized) to COCO (absolute pixel values)
                x_min = (x_center - w / 2) * width
                y_min = (y_center - h / 2) * height
                x_max = (x_center + w / 2) * width
                y_max = (y_center + h / 2) * height
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(int(class_id))

        # Convert to DETR format
        annotations = {
            'image_id': idx,
            'annotations': [
                {'bbox': box, 'category_id': label}
                for box, label in zip(boxes, labels)
            ]
        }

        # Preprocess with DETR processor
        encoding = self.processor(images=img, annotations=annotations, return_tensors="pt")
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}  # Remove batch dim
        encoding['image_id'] = idx
        return encoding

# Collate function for DataLoader
def collate_fn(batch):
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    labels = [
        {
            'boxes': item['labels']['boxes'],
            'class_labels': item['labels']['class_labels'],
            'image_id': item['image_id']
        }
        for item in batch
    ]
    return {'pixel_values': pixel_values, 'labels': labels}

# Training function
def train_model():
    # Paths for YAML
    yaml_file = r"D:\My Nuclear Codes\Python\progs\ASL translates\dataset\data.yaml"
    if not os.path.exists(yaml_file):
        raise FileNotFoundError(
            f"YAML file not found at: {yaml_file}. "
            "Please download the dataset from Roboflow and place 'data.yaml' in 'dataset/'."
        )

    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)
    
    train_img_dir = config['train']
    train_ann_dir = config['train'].replace('images', 'labels')
    val_img_dir = config['val']
    val_ann_dir = config['val'].replace('images', 'labels')

    # Verify directories
    for d in [train_img_dir, train_ann_dir, val_img_dir, val_ann_dir]:
        if not os.path.exists(d):
            raise FileNotFoundError(f"Directory not found: {d}. Check 'data.yaml' paths.")

    # DETR processor and model
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=26,  # 26 classes (A-Z)
        ignore_mismatched_sizes=True
    )

    # Create datasets
    train_dataset = ASLDataset(train_img_dir, train_ann_dir, yaml_file, processor)
    val_dataset = ASLDataset(val_img_dir, val_ann_dir, yaml_file, processor)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    # Training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            pixel_values = batch['pixel_values'].to(device)
            labels = [
                {
                    'boxes': label['boxes'].to(device),
                    'class_labels': label['class_labels'].to(device),
                    'image_id': label['image_id']
                }
                for label in batch['labels']
            ]

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader)}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch['pixel_values'].to(device)
                labels = [
                    {
                        'boxes': label['boxes'].to(device),
                        'class_labels': label['class_labels'].to(device),
                        'image_id': label['image_id']
                    }
                    for label in batch['labels']
                ]
                outputs = model(pixel_values=pixel_values, labels=labels)
                val_loss += outputs.loss.item()

        print(f"Epoch {epoch + 1}, Val Loss: {val_loss / len(val_loader)}")

    # Save model
    output_dir = r"D:\My Nuclear Codes\Python\progs\ASL translates\output\asl_detr_model"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"Error: {e}")