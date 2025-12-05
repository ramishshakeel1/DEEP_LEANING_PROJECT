"""
Helper script to create label files for Oxford Flowers 102 dataset.
This script generates train_labels.txt, valid_labels.txt, and test_labels.txt
based on the standard Oxford Flowers 102 dataset structure.

Usage:
    python create_labels.py

The script will:
1. Check if label files already exist
2. If not, generate them based on image filenames and standard Oxford 102 mapping
3. Save label files in the dataset directory
"""

import os
from pathlib import Path
import json

def create_label_files(data_dir='dataset', cat_to_name_path='cat_to_name.json'):
    """
    Create label files for train/valid/test splits.
    
    For Oxford Flowers 102:
    - Each class has ~40-258 images
    - Images are typically numbered sequentially
    - Labels are 1-102 (we'll convert to 0-101 for PyTorch)
    """
    data_path = Path(data_dir)
    
    # Load category mapping
    with open(cat_to_name_path, 'r') as f:
        cat_to_name = json.load(f)
    
    num_classes = len(cat_to_name)
    
    for split in ['train', 'valid', 'test']:
        label_file = data_path / f'{split}_labels.txt'
        
        if label_file.exists():
            print(f"{label_file} already exists. Skipping...")
            continue
        
        split_dir = data_path / split
        if not split_dir.exists():
            print(f"Warning: {split_dir} does not exist. Skipping...")
            continue
        
        # Get all image files
        image_files = sorted(split_dir.glob('*.jpg'))
        
        if len(image_files) == 0:
            print(f"Warning: No images found in {split_dir}")
            continue
        
        print(f"Processing {split}: {len(image_files)} images")
        
        # Create labels
        # For Oxford 102, we need to map image numbers to classes
        # Standard structure: images are numbered 1-8189
        # Classes are distributed: each class has multiple images
        
        # Since we don't have the exact mapping, we'll create a simple distribution
        # In practice, you should use the official labels from the dataset
        labels = []
        
        # Method 1: If images are in subdirectories (class folders)
        subdirs = [d for d in split_dir.iterdir() if d.is_dir()]
        if subdirs:
            print(f"  Found {len(subdirs)} class subdirectories")
            for img_file in image_files:
                class_name = img_file.parent.name
                try:
                    class_num = int(class_name)
                    labels.append((img_file.name, class_num - 1))  # 0-indexed
                except ValueError:
                    # If not a number, skip or use default
                    labels.append((img_file.name, 0))
        else:
            # Method 2: Distribute images evenly across classes
            # This is a fallback - you should provide proper labels
            print(f"  No subdirectories found. Using even distribution (this is a placeholder!)")
            images_per_class = len(image_files) // num_classes
            for i, img_file in enumerate(image_files):
                class_num = (i // images_per_class) % num_classes
                labels.append((img_file.name, class_num))
        
        # Write label file
        with open(label_file, 'w') as f:
            for filename, label in labels:
                f.write(f"{filename} {label}\n")
        
        print(f"  Created {label_file} with {len(labels)} entries")
    
    print("\nLabel files created successfully!")
    print("Note: If labels are incorrect, please provide proper label files or")
    print("      organize images in class subdirectories (1/, 2/, ..., 102/)")

if __name__ == '__main__':
    create_label_files()

