## Title

**Deep Learning-Based Flower Classification on the Oxford 102 Dataset Using Baseline CNN and Transfer Learning (ResNet18, EfficientNet-B0)**

---

## 1. Project Overview

**Goal**: Build a complete, GPU-accelerated deep learning pipeline to classify images of flowers from the **Oxford 102 Flowers** dataset into **102 classes**, reproducing and improving on a 2024 CNN-based research paper.  

**Key characteristics of the project:**
- **Dataset**: Oxford 102 Flowers (102 categories, 8,189 images total), split into **train / validation / test**.
- **Models**:
  - A **baseline CNN** closely following the paper’s architecture.
  - **Transfer learning models**: `ResNet18` and `EfficientNet-B0` (and optionally `MobileNetV2`).
- **Framework**: PyTorch + TorchVision.
- **Hardware target**: Single NVIDIA RTX 4060 GPU (8 GB VRAM), with **mixed precision (FP16)** to keep training fast and memory‑efficient.
- **Training time target**: About **1–3 hours** total for main experiments (depending on epochs and GPU).
- **Audience level**: Implementation and comments are written for a **2nd‑year university student** who understands Python and has basic deep learning knowledge.

All code is organized so that you can:
- Train a **single model** (`train_flowers.py`).
- Train **two models and compare them** (`train_and_compare_models.py`).
- **Visualize predictions** and model behavior (`visualize_predictions.py`).
- **Verify labels and evaluation correctness** (`verify_label_matching.py`).

---

## 2. Dataset: Oxford 102 Flowers

### 2.1 Dataset Description

- **Name**: 102 Category Flower Dataset (Oxford Flowers 102).
- **Authors**: Maria-Elena Nilsback and Andrew Zisserman.
- **Classes**: 102 flower species.
- **Images per class**: Between **40 and 258** images per class.
- **Variations**:
  - Large variations in **scale, pose, illumination**.
  - Some classes are **visually very similar**.
  - Some classes have large **intra-class variation** (same class looks very different across images).

This makes the dataset **challenging** and a good benchmark for CNNs and transfer learning.

### 2.2 Directory Structure Used in the Project

At project root, the dataset is expected in a folder like:

```text
dataset/
  train/
    1/
      image_00001.jpg
      ...
    2/
    ...
    102/
  valid/
    1/
    ...
    102/
  test/
    image_00001.jpg
    ...
cat_to_name.json
README.md   (original dataset readme)
```

- **`train/` and `valid/`**:
  - Organized as **subdirectories per class** (e.g., `train/1`, `train/2`, …, `train/102`).
  - The class folder name (string form of `1…102`) is used to derive a **0‑indexed label**:  
    \[
    \text{label} = \text{int(folder\_name)} - 1 \in [0, 101]
    \]

- **`test/`**:
  - Contains images directly (e.g., `test/image_00001.jpg`) without subfolders in the Kaggle-style setup.
  - Labels for test images are loaded from a **text file** `test_labels.txt` in `dataset/`.

- **`cat_to_name.json`**:
  - Maps **original category IDs (1–102)** to **human-readable class names** (e.g., `"1": "pink primrose"`).
  - Used to print and visualize class names.

### 2.3 Train / Valid / Test Split

The project uses a **fixed split** consistent with the official Oxford label files (`setid.mat`) / Kaggle derivative:

- **Train**: ~70% of images per class.
- **Validation**: ~15% of images per class.
- **Test**: ~15% of images per class.

Exact numbers depend on the specific dataset organization (Kaggle version vs. strictly official split), but in this project:
- Train & valid images are read from **class subfolders**, so labels come from **directory names**.
- Test labels come from a dedicated `test_labels.txt` file, which was **fixed** to ensure **all 819 test images** are present.

### 2.4 Label Handling and Verification

Label handling is critical, and this project includes **explicit verification**:

- **Train / Valid**:
  - `FlowerDataset` checks if the split folder (`dataset/train`, `dataset/valid`) contains **subdirectories**.
  - Each directory name `d.name` is interpreted as a class ID:
    - Convert to int: `class_num = int(d.name)`.
    - Convert to zero-based index: `label = class_num - 1`.
  - For each image, `label = class_to_idx[folder_name]`.

- **Test**:
  - If subfolders are absent, `FlowerDataset` looks for a label file:
    - `dataset/test_labels.txt`.
  - Format of each line:
    ```text
    filename label_index
    ```
    where `label_index` ∈ [0, 101].
  - A small script and verification step ensured that:
    - **All 819 test images** appear in `test_labels.txt`.
    - No extra entries exist.
    - All labels are within [0, 101].

- **`verify_label_matching.py`**:
  - **`verify_label_files()`**: Checks that:
    - The number of images in each split directory matches the entries in the corresponding `*_labels.txt` file.
    - All images are present in the label file (no missing / extra).
    - Label ranges are correct and number of unique labels is close to 102.
  - **`verify_dataset_loading()`**:
    - Constructs a `FlowerDataset` and compare its internal labels to the labels loaded directly from the `*_labels.txt` files.
    - Ensures the dataset’s label indexing matches the external files.
  - **`verify_evaluation_matching()`**:
    - Simulates the evaluation loop with a `DataLoader`.
    - Confirms that:
      - Labels inside the batch (`labels` in the loader) match `dataset.labels[idx]`.
      - Those labels are consistent with the label files.
  - **Outcome**: Verified that **evaluation compares predictions to the correct ground-truth labels**, and both predictions and labels are 0‑indexed (0–101).

This verification is especially important for a research paper, because any **off‑by‑one** or misaligned label mapping would invalidate the reported accuracy.

---

## 3. Data Preprocessing and Augmentation

All transforms are defined in `train_flowers.py` via `get_transforms(split, image_size, use_advanced_aug=False)`.

### 3.1 Basic Preprocessing

For each split (`train`, `valid`, `test`), the main preprocessing steps are:

- **Resize / Center-crop or Pad** to a fixed size:
  - For baseline CNN: **200×200**.
  - For pretrained models (ResNet18, EfficientNet): **224×224**.
- **Convert to Tensor**: `transforms.ToTensor()`.
- **Normalize** using ImageNet mean/std (for compatibility with torchvision pretrained models), typically:
  \[
  \text{mean} = [0.485, 0.456, 0.406], \quad
  \text{std}  = [0.229, 0.224, 0.225]
  \]

### 3.2 Training-Time Augmentation

The **train transforms** include:
- **Random horizontal flip**.
- **Random color jitter**, e.g. random changes to:
  - Brightness
  - Contrast
  - Saturation
  - Hue
- Optional **advanced augmentation** (controlled by `use_advanced_augmentation` / flags):
  - Could be extended with **MixUp** or **CutMix**, but the main training code focuses on standard geometric + color transforms for simplicity and robustness on a 2nd‑year level.

**Validation and Test transforms** only apply **deterministic** resizing and normalization (no random augmentation) to keep evaluation fair and reproducible.

### 3.3 Data Loading

`FlowerDataset` (defined in `train_flowers.py`):
- Handles:
  - Scanning directories (including subfolders).
  - Mapping filenames to labels from:
    - **Class folder names** (train/valid).
    - **Label text files** (test).
  - Applying transforms to each image.
- Return per sample:
  - For training/eval:
    - `image_tensor` of shape `(3, H, W)`.
    - `label` (0–101).
  - For visualization (in `FlowerDatasetWithOriginal`):
    - `image_tensor`, `label`, `original_image_numpy`, `path`.

`DataLoader` settings:
- **Batch size**: default **32** (configurable via CLI).
- **Shuffle**: `True` for train, `False` for valid/test.
- **num_workers**: configurable (default 4 for main training to speed up loading).
- **pin_memory=True** when using GPU to speed up host-to-device transfers.

---

## 4. Model Architectures

The project includes:
- A **custom baseline CNN** (to reflect the reference 2024 paper).
- **Transfer learning models** from torchvision: `ResNet18`, `MobileNetV2`, `EfficientNet-B0`.

### 4.1 Baseline CNN (From Paper)

Defined as `BaselineCNN` in `train_flowers.py`:

```284:360:train_flowers.py
class BaselineCNN(nn.Module):
    """
    Baseline CNN architecture from the research paper:
    "Flower Pictures Recognition Based on Advanced Convolutional Neural Network"
    ...
    """
    def __init__(self, num_classes=102, dropout_rate=0.5, image_size=200):
        super(BaselineCNN, self).__init__()
        self.num_classes = num_classes
        self.image_size = image_size
        
        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 200x200 -> 100x100
        )
        ...
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
        )
        
        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate)
        )
        
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.classifier(x)
        return x
```

**Key design points:**
- **Conv trunk**:
  - 4 convolutional blocks:
    - Conv(3→32) → BN → ReLU → MaxPool
    - Conv(32→64) → BN → ReLU → MaxPool
    - Conv(64→128) → BN → ReLU → MaxPool
    - Conv(128→256) → BN → ReLU → **Global Average Pooling** (AdaptiveAvgPool2d to 1×1).
- **Feature flattening**:
  - After global pooling, features are of size **256** per image.
- **Dense head**:
  - `fc1`: 256 → 256 → BN → ReLU → Dropout(0.5).
  - `fc2`: 256 → 128 → BN → ReLU → Dropout(0.5).
  - `classifier`: 128 → 102 (one logit per class).
- **Regularization**:
  - **BatchNorm** and **Dropout** in fully connected layers.
  - **Weight decay (L2)** added via the optimizer (`AdamW`).

This model is relatively lightweight, making it appropriate for a 2nd‑year project and feasible to train on 8 GB VRAM.

### 4.2 Transfer Learning Models

Transfer learning is implemented through a helper function `get_pretrained_model`:

```364:439:train_flowers.py
def get_pretrained_model(model_name='resnet18', num_classes=102, freeze_backbone=False):
    """
    Get pretrained model for transfer learning.
    ...
    """
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        ...
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )
        ...
    elif model_name == 'mobilenetv2':
        model = models.mobilenet_v2(pretrained=True)
        ...
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )
        ...
    elif model_name == 'efficientnet':
        from torchvision.models import efficientnet_b0
        model = efficientnet_b0(pretrained=True)
        ...
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )
        ...
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model
```

#### 4.2.1 ResNet18

- Standard **ResNet18** backbone (pretrained on ImageNet).
- Final fully connected layer is replaced with:
  - Dropout(0.5) → Linear(num_features → 102).
- Optionally, early layers can be frozen (via `freeze_backbone=True`), but in this project:
  - For final experiments, the backbone is generally **fine-tuned** (last blocks + classifier trainable).

#### 4.2.2 EfficientNet-B0

- Uses `torchvision.models.efficientnet_b0(pretrained=True)`.
- Original classifier is replaced with:
  - Dropout(0.5) → Linear(num_features → 102).
- A subset of final feature layers plus classifier are usually trainable in fine-tuning.
- EfficientNet-B0 typically achieves **higher accuracy** than ResNet18 on this dataset, at similar or slightly higher computational cost.

#### 4.2.3 MobileNetV2 (Optional)

- Similar pattern:
  - `models.mobilenet_v2(pretrained=True)`.
  - Replace final layer with Dropout + Linear to 102 classes.
- Offers a **lightweight** alternative for low‑resource experiments.

---

## 5. Training Pipeline

The main training logic is defined in:
- `train_flowers.py` (single model).
- `train_and_compare_models.py` (two models + comparison).

### 5.1 Device and GPU Handling

`train_flowers.py` detects and configures GPU:

```732:757:train_flowers.py
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"✓ CUDA is available!")
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    try:
        test_tensor = torch.randn(1, 1).to(device)
        print(f"✓ GPU test successful - GPU is ready for training!")
    except Exception as e:
        print(f"⚠ Warning: GPU test failed: {e}")
        print("Falling back to CPU...")
        device = torch.device('cpu')
else:
    device = torch.device('cpu')
    print(f"⚠ CUDA is NOT available - Using CPU")
    ...
```

- If CUDA is available, **GPU is used automatically**.
- A small tensor is moved to GPU to verify configuration; otherwise falls back to CPU with guidance.

### 5.2 Core Training Functions

#### 5.2.1 `train_epoch`

```445:479:train_flowers.py
def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None, use_mixed_precision=False):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        if use_mixed_precision and scaler:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc
```

**Notes**:
- Uses **CrossEntropyLoss** as criterion.
- Supports **mixed precision training** via `torch.cuda.amp.autocast` and `GradScaler`:
  - Reduces memory usage.
  - Speeds up training on RTX 4060.

#### 5.2.2 `validate`

```481:502:train_flowers.py
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc
```

- No gradient computation.
- Returns **validation loss and accuracy** per epoch.

#### 5.2.3 `train_model`

```504:597:train_flowers.py
def train_model(model, train_loader, val_loader, config, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['lr_scheduler_factor'],
        patience=config['lr_scheduler_patience'],
        verbose=True
    )
    scaler = GradScaler() if config['use_mixed_precision'] else None
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    ...
    for epoch in range(config['epochs']):
        train_loss, train_acc = train_epoch(...)
        val_loss, val_acc = validate(...)
        scheduler.step(val_loss)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({...}, os.path.join(config['weights_dir'], f'best_{config["model_name"]}.pth'))
        else:
            patience_counter += 1
        ...
        if patience_counter >= config['early_stopping_patience']:
            print("Early stopping...")
            break
    ...
    return history, best_val_acc
```

**Training details**:
- **Optimizer**: `AdamW` (Adam with decoupled weight decay).
- **Learning rate scheduler**: `ReduceLROnPlateau` on **validation loss**:
  - Reduces LR by `factor` (`0.5` default) if no improvement for `patience` epochs.
- **Early stopping**:
  - Stops training if validation loss does not improve for `early_stopping_patience` epochs.
- **Best model checkpoint**:
  - Stores:
    - `model_state_dict`
    - `optimizer_state_dict`
    - `val_loss`, `val_acc`
    - `epoch`
  - Saved under `weights/best_<model_name>.pth`.

These mechanisms help:
- Avoid overfitting.
- Adjust learning rate automatically.
- Keep training within the **1–3 hour** time budget.

### 5.3 Training Entry Points and CLI

#### 5.3.1 Single-Model Training (`train_flowers.py`)

```704:727:train_flowers.py
parser = argparse.ArgumentParser(description='Flower Classification Training')
parser.add_argument('--model', type=str, default='baseline',
                   choices=['baseline', 'resnet18', 'mobilenetv2', 'efficientnet'],
                   help='Model architecture to use')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--image_size', type=int, default=224, help='Image size')
parser.add_argument('--data_dir', type=str, default='dataset', help='Dataset directory')
parser.add_argument('--no_mixed_precision', action='store_true',
                   help='Disable mixed precision training')
...
```

Example training commands:
- Baseline CNN:
  - `python train_flowers.py --model baseline --epochs 50 --batch_size 32 --lr 0.001`
- ResNet18:
  - `python train_flowers.py --model resnet18 --epochs 30 --batch_size 32 --lr 0.0005`
- EfficientNet-B0:
  - `python train_flowers.py --model efficientnet --epochs 30 --batch_size 32 --lr 0.0005`

#### 5.3.2 Dual-Model Training and Comparison (`train_and_compare_models.py`)

```446:455:train_and_compare_models.py
parser = argparse.ArgumentParser(description='Train and Compare Models')
parser.add_argument('--model1', type=str, default='resnet18', ...)
parser.add_argument('--model2', type=str, default='efficientnet', ...)
parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--num_samples', type=int, default=12,
                   help='Number of samples for comparison visualization')
parser.add_argument('--skip_training', action='store_true',
                   help='Skip training, only generate comparison (requires existing models)')
```

- Trains **model1** (e.g., ResNet18) and then **model2** (e.g., EfficientNet), or loads them from saved weights.
- Generates:
  - Individual results (curves, confusion matrices).
  - A **comparison visualization**.
  - **Text summary** of model comparison (`results/model_comparison_summary.txt`).

Example:
- `python train_and_compare_models.py --model1 resnet18 --model2 efficientnet --epochs 30`
- `python train_and_compare_models.py --model1 resnet18 --model2 efficientnet --skip_training`

---

## 6. Evaluation Metrics and Outputs

Evaluation is handled by `evaluate_model` in `train_flowers.py`:

```603:641:train_flowers.py
def evaluate_model(model, test_loader, device, class_names=None):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    accuracy = 100. * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    cm = confusion_matrix(all_labels, all_preds)
    if class_names is None:
        class_names = [f'Class {i}' for i in range(102)]
    report = classification_report(all_labels, all_preds, 
                                  target_names=class_names[:102],
                                  output_dict=True)
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }
```

### 6.1 Metrics

- **Overall test accuracy**:
  - Percentage of correctly classified test samples.
- **Confusion matrix**:
  - 102×102 matrix showing how often each true class is predicted as each other class.
  - For visualization, often only the **top-N most frequent classes** are shown (e.g., top 20).
- **Classification report** (from `scikit-learn`):
  - For each class:
    - Precision
    - Recall
    - F1-score
    - Support (number of instances)

### 6.2 Plots and Saved Artifacts

`train_flowers.py` generates:
- **Training curves** (`plot_training_history`):
  - Plots **train vs validation loss** and **train vs validation accuracy** vs epochs.
  - Saved as: `results/training_curves_<model>.png`.
- **Confusion matrix** (`plot_confusion_matrix`):
  - Heatmap for top-N classes or full 102 classes.
  - Saved as: `results/confusion_matrix_<model>.png`.

`train_and_compare_models.py` additionally generates:
- **Comparison visualization** (`create_comparison_visualization`):
  - Side-by-side bar charts of class probabilities for each of two models for multiple test images.
  - Each row:
    - Original image.
    - Model 1 probabilities and predicted vs true class (flagged True/False).
    - Model 2 probabilities and predicted vs true class.
  - Saved as:
    - `results/model_comparison_resnet18_vs_efficientnet.png` (for the main experiment).
- **Text summary** (`save_comparison_summary`):
  - Summarizes:
    - Test accuracies.
    - Best validation accuracies.
    - Agreement statistics (both correct, both wrong, one correct, other wrong).
  - Saved as:
    - `results/model_comparison_summary.txt`.

`visualize_predictions.py`:
- Produces **single-model prediction visualization**:
  - For each selected test image:
    - Left: original image with labels (true and predicted).
    - Right: probability distribution over classes (bar chart) with true class in blue and predicted in red if incorrect.
  - Saved as:
    - `results/prediction_comparison_<model>.png` (e.g., `prediction_comparison_resnet18.png`).

---

## 7. Model Comparison: ResNet18 vs EfficientNet-B0

The project includes a **concrete comparison experiment** between ResNet18 and EfficientNet-B0 on the Oxford 102 test set.

### 7.1 Final Quantitative Results

From `results/model_comparison_summary.txt`:

```1:29:results/model_comparison_summary.txt
======================================================================
MODEL COMPARISON SUMMARY
======================================================================

Model 1: RESNET18
Model 2: EFFICIENTNET
...
RESNET18: 95.12%
EFFICIENTNET: 96.83%
Difference: +1.71% (efficientnet better)
...
RESNET18 - Best Val Acc: 95.11%
EFFICIENTNET - Best Val Acc: 97.56%
...
Agreement: 94.14%
Both correct: 93.41%
Both wrong: 1.47%
resnet18 only correct: 1.71%
efficientnet only correct: 3.42%
```

**Summary of these results:**
- **Test Accuracy**:
  - **ResNet18**: **95.12%**
  - **EfficientNet-B0**: **96.83%**
  - **Difference**: EfficientNet is **+1.71 percentage points** better.

- **Best Validation Accuracy**:
  - ResNet18: **95.11%**
  - EfficientNet-B0: **97.56%**
  - EfficientNet shows stronger generalization on the validation set as well.

- **Prediction Agreement**:
  - **Agreement (same prediction)**: 94.14% of test cases.
  - **Both correct**: 93.41% of test cases.
  - **Both wrong**: 1.47% of test cases.
  - **ResNet18 only correct**: 1.71%.
  - **EfficientNet only correct**: 3.42%.

Interpretation:
- Both networks are generally strong and **agree** most of the time.
- When they disagree, **EfficientNet is right about twice as often** as ResNet18 (3.42% vs 1.71%).
- EfficientNet is especially useful for the **harder edge-cases**, which is consistent with its more powerful architecture.

### 7.2 Qualitative Visualization

The figure `results/model_comparison_resnet18_vs_efficientnet.png` shows:
- Rows of example test images.
- For each row:
  - First column: original image with true label.
  - Second column: ResNet18 probabilities over classes, true vs predicted label, and a **True/False** indicator.
  - Third column: EfficientNet probabilities, again with highlighting and True/False indicator.

This visualization helps:
- Understand **where both models succeed** (both True).
- Inspect **hard misclassifications** (both False).
- Study **cases where only one model is correct**, revealing strengths and weaknesses of each architecture.

---

## 8. Label and Evaluation Correctness (Important for Research)

One of your explicit goals was to **ensure that evaluation compares predictions to the correct labels** and that there are **no label mistakes**.  
This is crucial for a detailed and trustworthy research paper.

### 8.1 `verify_label_matching.py` Design

This script:
- Checks **completeness and range** of label files.
- Verifies that `FlowerDataset` reads the labels as intended.
- Confirms that the **DataLoader and evaluation loop** see exactly the same labels as the dataset and the label files.

Key checks:
- **File-level verification**:
  - Count images in `dataset/<split>` (recursively including subfolders).
  - Count entries in `<split>_labels.txt`.
  - Match:
    - Missing images.
    - Extra labels.
    - Label range ∈ [0, 101].
    - Approximate number of unique labels ≈ 102.
- **Dataset-level verification**:
  - Build `FlowerDataset(data_dir, split, transform)` and manually load the same label file.
  - Check for mismatches between `dataset.labels[idx]` and label file.
- **Evaluation-level verification**:
  - Run through a `DataLoader`.
  - Collect labels from batches and cross-check them against `dataset.labels` and the label file.

### 8.2 Outcome

- The script confirmed that:
  - The **label mapping** is consistent across:
    - Directory structure (for train/valid).
    - Label text files (especially `test_labels.txt`).
    - Dataset objects.
    - Evaluation pipeline.
  - Both labels and predictions are consistently **0-indexed [0–101]**.
  - The evaluation logic **computes accuracy as direct equality** between predicted and ground-truth labels.

This provides strong evidence that the final reported accuracies (e.g., **95.12% vs 96.83%**) are **valid and trustworthy**.

---

## 9. Implementation Environment, Dependencies, and Practical Constraints

### 9.1 Dependencies (`requirements.txt`)

```1:8:requirements.txt
torch>=2.0.0
torchvision>=0.15.0
numpy<2.0.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
seaborn>=0.12.0
Pillow>=10.0.0
tqdm>=4.65.0
```

Notes:
- **`numpy<2.0.0`** is enforced to avoid DLL/compatibility issues with libraries like Matplotlib and scikit-learn on Windows.
- These versions are compatible with **PyTorch 2.x + CUDA** on an RTX 4060.

### 9.2 GPU and Training Time Constraints

- Hardware target: **NVIDIA RTX 4060 (8 GB)**.
- Memory-sensitive design:
  - Batch size **32**.
  - Mixed precision enabled by default where supported.
  - Efficient network architectures (no extremely large models).
- Expected training duration (approximate, depends on implementation details and I/O):
  - Baseline CNN: relatively fast training (fewer layers).
  - ResNet18: moderate training time.
  - EfficientNet-B0: somewhat longer than ResNet18 but still **within 1–3 hours** total for 30–50 epochs on this GPU.

---

## 10. How to Use This Report for a Detailed Research Paper

You can map sections of this markdown directly into the standard research paper structure:

- **Introduction / Related Work**:
  - Use **Section 2.1** and the source references in `README.md` to describe the dataset and problem importance.

- **Dataset & Preprocessing (Methods)**:
  - Use **Sections 2–3**:
    - Dataset description and split.
    - Directory structure and label handling.
    - Preprocessing and data augmentation pipeline.

- **Model Architecture (Methods)**:
  - Use **Section 4**:
    - Baseline CNN design, referencing the 2024 paper.
    - Details on ResNet18 and EfficientNet-B0 transfer learning setup.
    - Explanation of regularization (dropout, weight decay).

- **Training Procedure (Methods)**:
  - Use **Section 5**:
    - Optimizer, learning rate, scheduler.
    - Early stopping, mixed precision, batch size.
    - GPU utilization.

- **Evaluation (Experiments / Results)**:
  - Use **Section 6**:
    - Define metrics: accuracy, confusion matrix, per-class precision/recall/F1.
    - Describe the visualization plots and what they show.

- **Results & Discussion**:
  - Use **Section 7**:
    - Quantitative comparison: 95.12% vs 96.83%.
    - Agreement analysis and error structure.
    - Qualitative visualizations for interpreting mistakes.

- **Reliability / Sanity Checks**:
  - Use **Section 8**:
    - Detail how label matching was verified.
    - Argue that evaluation is correct and robust.

- **Implementation Details / Reproducibility**:
  - Use **Section 9**:
    - Dependencies, versions, hardware.
    - Comments on training time and practical constraints.

If you tell me your exact target format (e.g., IEEE conference, with specific sections like “Abstract, Introduction, Methodology, Experiments, Conclusion”), I can convert this markdown into a **ready-to-paste structured paper** with more formal language and references.

---

## 11. Summary of the Entire Project in One Paragraph

This project implements a complete deep learning pipeline for the Oxford 102 Flowers dataset using both a custom CNN (based on a 2024 research paper) and modern transfer learning models (ResNet18, EfficientNet-B0), with careful dataset handling, label verification, mixed-precision GPU training, and detailed evaluation. The data is split into train/validation/test, with images preprocessed and augmented, and labels rigorously checked for correctness. Models are trained with AdamW, learning rate scheduling, and early stopping, and evaluated through accuracy, confusion matrices, per-class reports, and rich visualizations of predictions. Final experiments show that EfficientNet-B0 outperforms ResNet18 by about 1.7 percentage points in test accuracy (96.83% vs 95.12%) and is more reliable on hard cases, providing a strong empirical basis for a detailed research paper.


