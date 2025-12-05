"""
Fix test_labels.txt to include ALL 819 test images

This ensures every test image has a correct label

Run: python fix_complete_test_labels.py
"""

import scipy.io as sio
import numpy as np
from pathlib import Path
import shutil

def create_complete_test_labels(data_dir='dataset'):
    """
    Create complete test_labels.txt with ALL 819 images using official Oxford labels.
    """
    
    print("="*60)
    print("CREATING COMPLETE TEST LABELS")
    print("="*60)
    
    # Load official labels
    if not Path('imagelabels.mat').exists() or not Path('setid.mat').exists():
        print("Error: Need imagelabels.mat and setid.mat")
        print("Run: python download_oxford_labels.py first")
        return False
    
    labels_mat = sio.loadmat('imagelabels.mat')
    setid_mat = sio.loadmat('setid.mat')
    
    # Extract data
    all_labels = labels_mat['labels'][0] - 1  # Convert to 0-indexed (0-101)
    test_ids = setid_mat['tstid'][0] - 1  # Convert to 0-indexed
    
    print(f"Official test set size: {len(test_ids)} images")
    
    # Get all test images in your directory
    data_path = Path(data_dir)
    test_dir = data_path / 'test'
    
    if not test_dir.exists():
        print(f"Error: {test_dir} does not exist!")
        return False
    
    # Get ALL images in test directory
    test_images = sorted(test_dir.glob('*.jpg'))
    print(f"Images in test directory: {len(test_images)}")
    
    if len(test_images) != 819:
        print(f"[WARNING] Expected 819 images, found {len(test_images)}")
    
    # Create mapping: image number -> label
    image_num_to_label = {}
    for img_num in test_ids:
        label = all_labels[img_num]
        image_num_to_label[img_num] = label
    
    # Map each test image to its label
    label_mapping = {}
    missing_labels = []
    
    for img_path in test_images:
        # Extract image number from filename
        # Format: image_XXXXX.jpg -> XXXXX (1-8189)
        try:
            img_num_str = img_path.stem.replace('image_', '')
            img_num = int(img_num_str) - 1  # Convert to 0-indexed (0-8188)
            
            if img_num in image_num_to_label:
                label = image_num_to_label[img_num]
                label_mapping[img_path.name] = label
            else:
                # This image is not in the official test set!
                print(f"  Warning: {img_path.name} (image {img_num+1}) not in official test set")
                missing_labels.append(img_path.name)
                # Use a default label (we'll fix this below)
                label_mapping[img_path.name] = 0
        
        except ValueError as e:
            print(f"  Error parsing filename: {img_path.name}: {e}")
            continue
    
    print(f"\nMapped {len(label_mapping)} images to labels")
    
    if missing_labels:
        print(f"\n[WARNING] {len(missing_labels)} images not in official test set!")
        print(f"First few: {missing_labels[:5]}")
        print("\nThese images might be from train/valid that were moved to test.")
        print("Attempting to find their correct labels...")
        
        # Try to find labels for missing images
        fixed_count = 0
        for img_name in missing_labels:
            try:
                img_num_str = img_name.replace('image_', '').replace('.jpg', '')
                img_num = int(img_num_str) - 1  # 0-indexed
                
                if 0 <= img_num < len(all_labels):
                    # Use the label from the full dataset
                    label = all_labels[img_num]
                    label_mapping[img_name] = label
                    fixed_count += 1
                else:
                    print(f"  Cannot find label for {img_name} (index {img_num} out of range)")
            except Exception as e:
                print(f"  Error fixing {img_name}: {e}")
        
        print(f"[OK] Fixed {fixed_count}/{len(missing_labels)} missing labels")
    
    # Backup old file
    label_file = data_path / 'test_labels.txt'
    if label_file.exists():
        backup_file = data_path / 'test_labels.txt.backup2'
        shutil.copy(label_file, backup_file)
        print(f"\n[OK] Backed up old labels to {backup_file}")
    
    # Write complete label file
    with open(label_file, 'w') as f:
        for img_name in sorted(label_mapping.keys()):
            label = label_mapping[img_name]
            f.write(f"{img_name} {label}\n")
    
    print(f"\n[OK] Created {label_file} with {len(label_mapping)} labels")
    
    # Verify
    print("\nVerification:")
    label_values = list(label_mapping.values())
    print(f"  Total labels: {len(label_values)}")
    print(f"  Label range: {min(label_values)} to {max(label_values)}")
    print(f"  Unique labels: {len(set(label_values))}")
    
    from collections import Counter
    label_counts = Counter(label_values)
    print(f"  Avg per class: {len(label_values) / len(label_counts):.1f}")
    print(f"  Most common: {label_counts.most_common(3)}")
    
    print("\n" + "="*60)
    print("SUCCESS!")
    print("="*60)
    print(f"All {len(label_mapping)} test images now have labels!")
    print("\nNext: Re-evaluate your model")
    print("Expected: ~95% test accuracy (matching validation)")
    
    return True

def verify_complete_coverage(data_dir='dataset'):
    """Verify that ALL test images have labels."""
    
    print("\n" + "="*60)
    print("VERIFYING COMPLETE LABEL COVERAGE")
    print("="*60)
    
    data_path = Path(data_dir)
    test_dir = data_path / 'test'
    label_file = data_path / 'test_labels.txt'
    
    # Get all test images
    test_images = set(img.name for img in test_dir.glob('*.jpg'))
    print(f"Images in test directory: {len(test_images)}")
    
    # Get all labeled images
    labeled_images = set()
    if label_file.exists():
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    labeled_images.add(parts[0])
        print(f"Images in label file: {len(labeled_images)}")
    else:
        print("[ERROR] No label file found!")
        return False
    
    # Check coverage
    missing = test_images - labeled_images
    extra = labeled_images - test_images
    
    if missing:
        print(f"\n[ERROR] {len(missing)} images WITHOUT labels:")
        for img in sorted(list(missing)[:10]):
            print(f"  - {img}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
        return False
    
    if extra:
        print(f"\n[WARNING] {len(extra)} labels for NON-EXISTENT images:")
        for img in sorted(list(extra)[:10]):
            print(f"  - {img}")
    
    if not missing and not extra:
        print("\n[SUCCESS] PERFECT! All test images have labels!")
        print(f"   {len(test_images)} images = {len(labeled_images)} labels")
        return True
    elif not missing:
        print("\n[SUCCESS] All test images have labels (some extra labels exist)")
        return True
    
    return False

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--verify':
        # Just verify coverage
        verify_complete_coverage()
    else:
        # Create complete labels
        success = create_complete_test_labels()
        
        if success:
            # Verify coverage
            verify_complete_coverage()

