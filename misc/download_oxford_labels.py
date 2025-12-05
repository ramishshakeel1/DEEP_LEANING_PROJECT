"""
Download and process official Oxford Flowers 102 labels

This will create correct label files for your dataset

Run: python download_oxford_labels.py

"""

import urllib.request

import scipy.io as sio

import numpy as np

from pathlib import Path

import shutil

def download_official_labels():

    """

    Download official Oxford Flowers 102 label files.

    Official dataset splits:

    - imagelabels.mat: Contains class labels for all 8189 images

    - setid.mat: Contains train/val/test split indices

    """

    

    print("="*60)

    print("DOWNLOADING OXFORD FLOWERS 102 OFFICIAL LABELS")

    print("="*60)

    

    base_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"

    

    # Download files

    files_to_download = {

        'imagelabels.mat': 'imagelabels.mat',

        'setid.mat': 'setid.mat'

    }

    

    for filename, local_name in files_to_download.items():

        url = base_url + filename

        local_path = Path(local_name)

        

        if local_path.exists():

            print(f"[OK] {local_name} already exists")

        else:

            print(f"Downloading {filename}...")

            try:

                # Create SSL context that doesn't verify certificates

                import ssl

                ssl_context = ssl.create_default_context()

                ssl_context.check_hostname = False

                ssl_context.verify_mode = ssl.CERT_NONE

                

                # Use urlopen with SSL context

                req = urllib.request.Request(url)

                with urllib.request.urlopen(req, context=ssl_context) as response, open(local_path, 'wb') as out_file:

                    shutil.copyfileobj(response, out_file)

                print(f"[OK] Downloaded {local_name}")

            except Exception as e:

                print(f"[ERROR] Error downloading {filename}: {e}")

                print(f"  Please download manually from: {url}")

                return False

    

    return True

def create_correct_label_files(data_dir='dataset'):

    """

    Create correct label files using official Oxford Flowers 102 labels.

    """

    

    # Check if .mat files exist

    if not Path('imagelabels.mat').exists() or not Path('setid.mat').exists():

        print("Error: Label files not found. Downloading...")

        if not download_official_labels():

            return False

    

    print("\n" + "="*60)

    print("CREATING CORRECT LABEL FILES")

    print("="*60)

    

    # Load official labels

    labels_mat = sio.loadmat('imagelabels.mat')

    setid_mat = sio.loadmat('setid.mat')

    

    # Extract data

    # labels: 1-102 (need to convert to 0-101)

    all_labels = labels_mat['labels'][0] - 1  # Convert to 0-indexed

    

    # Split indices (1-indexed in .mat file)

    train_ids = setid_mat['trnid'][0] - 1  # Convert to 0-indexed

    val_ids = setid_mat['valid'][0] - 1

    test_ids = setid_mat['tstid'][0] - 1

    

    print(f"Total images: {len(all_labels)}")

    print(f"Train: {len(train_ids)} images")

    print(f"Valid: {len(val_ids)} images")

    print(f"Test: {len(test_ids)} images")

    

    data_path = Path(data_dir)

    

    # Create label files for each split

    splits = {

        'train': train_ids,

        'valid': val_ids,

        'test': test_ids

    }

    

    for split_name, indices in splits.items():

        split_dir = data_path / split_name

        

        if not split_dir.exists():

            print(f"\nWarning: {split_dir} does not exist!")

            continue

        

        # Get all images in this split directory (including subdirectories)

        image_files = []

        # Check subdirectories first (class folders)

        subdirs = [d for d in split_dir.iterdir() if d.is_dir()]

        if subdirs:

            for subdir in subdirs:

                image_files.extend(sorted(subdir.glob('*.jpg')))

        else:

            # Images directly in split directory

            image_files = sorted(split_dir.glob('*.jpg'))

        

        print(f"\n{split_name.upper()}:")

        print(f"  Images in directory: {len(image_files)}")

        print(f"  Expected images: {len(indices)}")

        

        if len(image_files) == 0:

            print(f"  [ERROR] No images found!")

            continue

        

        # Create mapping: image filename -> label

        # Oxford format: image_XXXXX.jpg where XXXXX is 1-8189

        image_to_label = {}

        

        for img_path in image_files:

            # Extract image number from filename

            # Format: image_00001.jpg -> 1

            try:

                img_num_str = img_path.stem.replace('image_', '')

                img_num = int(img_num_str) - 1  # Convert to 0-indexed (0-8188)

                

                # Check if this image belongs to this split

                if img_num in indices:

                    label = all_labels[img_num]

                    image_to_label[img_path.name] = label

                else:

                    # Image is in wrong split directory!

                    print(f"  Warning: {img_path.name} (image {img_num+1}) should not be in {split_name}")

            

            except ValueError as e:

                print(f"  Error parsing filename: {img_path.name}")

                continue

        

        # Write label file

        label_file = data_path / f'{split_name}_labels.txt'

        

        # Backup old file

        if label_file.exists():

            backup_file = data_path / f'{split_name}_labels.txt.backup'

            shutil.copy(label_file, backup_file)

            print(f"  Backed up old labels to {backup_file}")

        

        # Write new labels

        with open(label_file, 'w') as f:

            for img_name in sorted(image_to_label.keys()):

                label = image_to_label[img_name]

                f.write(f"{img_name} {label}\n")

        

        print(f"  [OK] Created {label_file} with {len(image_to_label)} labels")

        

        # Verify labels

        if len(image_to_label) > 0:

            label_values = list(image_to_label.values())

            print(f"  Label range: {min(label_values)} to {max(label_values)}")

            print(f"  Unique labels: {len(set(label_values))}")

    

    print("\n" + "="*60)

    print("SUCCESS!")

    print("="*60)

    print("Correct label files created!")

    print("\nNext steps:")

    print("1. Re-run training: python train_flowers.py --model resnet18 --epochs 30")

    print("2. Or evaluate saved model")

    print("3. Expected test accuracy: ~95% (matching validation)")

    

    return True

def verify_label_quality():

    """Quick verification that labels are correct."""

    

    print("\n" + "="*60)

    print("VERIFYING LABEL QUALITY")

    print("="*60)

    

    for split in ['train', 'valid', 'test']:

        label_file = Path('dataset') / f'{split}_labels.txt'

        

        if not label_file.exists():

            print(f"\n{split.upper()}: [ERROR] No label file")

            continue

        

        # Read labels

        labels = []

        with open(label_file, 'r') as f:

            for line in f:

                parts = line.strip().split()

                if len(parts) >= 2:

                    labels.append(int(parts[1]))

        

        print(f"\n{split.upper()}:")

        print(f"  Total images: {len(labels)}")

        print(f"  Unique labels: {len(set(labels))}")

        print(f"  Label range: {min(labels)} to {max(labels)}")

        

        # Check for cycling pattern (the bug)

        consecutive_resets = 0

        for i in range(1, len(labels)):

            if labels[i] < labels[i-1]:

                consecutive_resets += 1

        

        if consecutive_resets > 50:  # Threshold for suspicion

            print(f"  ⚠️ WARNING: Detected {consecutive_resets} label resets!")

            print(f"  This suggests a cycling pattern (INCORRECT LABELS)")

        else:

            print(f"  [OK] Labels look good ({consecutive_resets} natural resets)")

        

        # Check distribution

        from collections import Counter

        label_counts = Counter(labels)

        avg_per_class = len(labels) / len(label_counts)

        

        print(f"  Avg images per class: {avg_per_class:.1f}")

        

        # Show most/least common

        most_common = label_counts.most_common(3)

        least_common = sorted(label_counts.items(), key=lambda x: x[1])[:3]

        

        print(f"  Most common classes: {most_common}")

        print(f"  Least common classes: {least_common}")

if __name__ == '__main__':

    import sys

    

    print("Oxford Flowers 102 - Official Label Downloader")

    print("="*60)

    

    # Check if we should just verify

    if len(sys.argv) > 1 and sys.argv[1] == '--verify':

        verify_label_quality()

    else:

        # Download and create correct labels

        success = create_correct_label_files()

        

        if success:

            # Verify the result

            verify_label_quality()

        else:

            print("\nFailed to create label files.")

            print("Please ensure you have scipy installed: pip install scipy")

