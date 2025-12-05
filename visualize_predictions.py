"""
Visualize Image Predictions with Confidence Scores

Creates comparison visualizations showing:
- Input image
- Prediction confidence bar chart
- True label vs predicted label
- Correct/Incorrect indicator

Run: python visualize_predictions.py --model resnet18 --num_samples 20
"""

import os
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Import from train_flowers.py
from train_flowers import FlowerDataset, get_pretrained_model, BaselineCNN, get_transforms

# Custom dataset that also returns original image
class FlowerDatasetWithOriginal(FlowerDataset):
    """Extended dataset that returns original image for visualization."""
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load original image
        try:
            original_image = Image.open(img_path).convert('RGB')
        except Exception as e:
            original_image = Image.new('RGB', (224, 224), color='black')
        
        # Convert original to numpy for storage (DataLoader can't handle PIL Images)
        original_array = np.array(original_image)
        
        # Apply transforms to get tensor
        if self.transform:
            image = self.transform(original_image)
        else:
            image = transforms.ToTensor()(original_image)
        
        return image, label, original_array, str(img_path)


def custom_collate_fn(batch):
    """Custom collate function to handle variable-sized images."""
    images = []
    labels = []
    original_images = []
    paths = []
    
    for item in batch:
        images.append(item[0])
        labels.append(item[1])
        original_images.append(item[2])
        paths.append(item[3])
    
    # Stack tensors
    images = torch.stack(images, 0)
    labels = torch.tensor(labels)
    
    return images, labels, original_images, paths

def visualize_predictions(model, test_loader, device, class_names, num_samples=20, 
                         save_dir='results', model_name='resnet18'):
    """
    Create visualization comparing predictions with confidence scores.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test set
        device: torch device
        class_names: List of class names
        num_samples: Number of samples to visualize
        save_dir: Directory to save results
        model_name: Name of the model
    """
    model.eval()
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Collect predictions
    all_original_images = []
    all_labels = []
    all_preds = []
    all_probs = []
    all_paths = []
    
    with torch.no_grad():
        for images, labels, original_images, paths in test_loader:
            images_gpu = images.to(device)
            outputs = model(images_gpu)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            # Store original images (PIL Images before normalization)
            for i in range(len(images)):
                all_original_images.append(original_images[i])
                all_labels.append(labels[i].item())
                all_preds.append(predicted[i].item())
                all_probs.append(probs[i].cpu().numpy())
                all_paths.append(paths[i])
    
    # Select samples: mix of correct and incorrect predictions
    correct_indices = [i for i in range(len(all_labels)) if all_labels[i] == all_preds[i]]
    incorrect_indices = [i for i in range(len(all_labels)) if all_labels[i] != all_preds[i]]
    
    # Select samples
    num_correct = min(num_samples // 2, len(correct_indices))
    num_incorrect = min(num_samples - num_correct, len(incorrect_indices))
    
    selected_correct = np.random.choice(correct_indices, num_correct, replace=False)
    selected_incorrect = np.random.choice(incorrect_indices, num_incorrect, replace=False)
    
    selected_indices = list(selected_correct) + list(selected_incorrect)
    np.random.shuffle(selected_indices)
    
    # Create visualization (matching image style)
    rows = len(selected_indices)
    fig, axes = plt.subplots(rows, 2, figsize=(14, 3.5 * rows))
    fig.patch.set_facecolor('white')
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, sample_idx in enumerate(selected_indices):
        original_array = all_original_images[sample_idx]
        true_label = all_labels[sample_idx]
        pred_label = all_preds[sample_idx]
        probs = all_probs[sample_idx]
        
        is_correct = (true_label == pred_label)
        
        # Left: Image
        ax_img = axes[idx, 0]
        
        # Display original image (numpy array) - make it square
        ax_img.imshow(original_array)
        ax_img.axis('off')
        ax_img.set_aspect('equal')
        
        # Add border around image
        for spine in ax_img.spines.values():
            spine.set_visible(False)
        
        # Add label text below image (matching the image style)
        if is_correct:
            # Correct: Show both labels in blue
            label_text = f"{true_label} ({pred_label})"
            ax_img.text(0.5, -0.08, label_text, transform=ax_img.transAxes,
                       ha='center', fontsize=11, color='blue', weight='bold')
        else:
            # Incorrect: Show true label in blue, prediction in red
            true_text = f"{true_label}"
            pred_text = f"({pred_label})"
            ax_img.text(0.35, -0.08, true_text, transform=ax_img.transAxes,
                       ha='center', fontsize=11, color='blue', weight='bold')
            ax_img.text(0.65, -0.08, pred_text, transform=ax_img.transAxes,
                       ha='center', fontsize=11, color='red', weight='bold')
            ax_img.text(0.5, -0.15, "True label (Prediction)", 
                       transform=ax_img.transAxes, ha='center', fontsize=9,
                       color='gray', style='italic')
        
        # Right: Confidence bar chart
        ax_chart = axes[idx, 1]
        
        # Create bar chart (matching the image style)
        x_pos = np.arange(len(probs))
        bars = ax_chart.bar(x_pos, probs, color='lightgray', width=0.8, 
                           edgecolor='black', linewidth=0.5)
        
        # Highlight true label in blue
        bars[true_label].set_color('blue')
        bars[true_label].set_alpha(0.8)
        bars[true_label].set_edgecolor('darkblue')
        bars[true_label].set_linewidth(1.5)
        
        # Highlight predicted label in red (if different)
        if not is_correct:
            bars[pred_label].set_color('red')
            bars[pred_label].set_alpha(0.8)
            bars[pred_label].set_edgecolor('darkred')
            bars[pred_label].set_linewidth(1.5)
        
        ax_chart.set_xlim(-0.5, len(probs) - 0.5)
        ax_chart.set_ylim(0, max(probs) * 1.15)
        ax_chart.set_xlabel('Class', fontsize=9)
        ax_chart.set_ylabel('Confidence', fontsize=9)
        ax_chart.tick_params(labelsize=8)
        ax_chart.grid(True, alpha=0.2, axis='y', linestyle='--')
        ax_chart.set_facecolor('white')
        
        # Add border to chart
        for spine in ax_chart.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)
        
        # Add True/False indicator (matching image style - large text)
        if is_correct:
            ax_chart.text(0.98, 0.98, 'True', transform=ax_chart.transAxes,
                         ha='right', va='top', fontsize=24, color='blue',
                         weight='bold', family='sans-serif')
        else:
            ax_chart.text(0.98, 0.98, 'False', transform=ax_chart.transAxes,
                         ha='right', va='top', fontsize=24, color='red',
                         weight='bold', family='sans-serif')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'prediction_comparison_{model_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {save_path}")
    print(f"Showed {num_correct} correct and {num_incorrect} incorrect predictions")
    
    return save_path

def main():
    parser = argparse.ArgumentParser(description='Visualize Image Predictions')
    parser.add_argument('--model', type=str, default='resnet18',
                       choices=['baseline', 'resnet18', 'mobilenetv2', 'efficientnet'],
                       help='Model architecture')
    parser.add_argument('--num_samples', type=int, default=20,
                       help='Number of samples to visualize')
    parser.add_argument('--weights_dir', type=str, default='weights',
                       help='Directory with model weights')
    parser.add_argument('--data_dir', type=str, default='dataset',
                       help='Dataset directory')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Results directory')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Image size')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load class names
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    class_names = [cat_to_name[str(i+1)] for i in range(102)]
    
    # Create transforms (use function from train_flowers.py)
    test_transform = get_transforms('test', args.image_size, use_advanced_aug=False)
    
    # Create dataset
    print("Loading test dataset...")
    test_dataset = FlowerDatasetWithOriginal(args.data_dir, 'test', test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, 
                           num_workers=0, collate_fn=custom_collate_fn)
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Load model
    print(f"\nLoading {args.model} model...")
    if args.model == 'baseline':
        from train_flowers import BaselineCNN
        model = BaselineCNN(num_classes=102, dropout_rate=0.5, image_size=200)
    else:
        model = get_pretrained_model(args.model, num_classes=102, freeze_backbone=False)
    
    # Load weights
    weights_path = os.path.join(args.weights_dir, f'best_{args.model}.pth')
    if not os.path.exists(weights_path):
        print(f"Error: Model weights not found at {weights_path}")
        print("Please train the model first or check the weights directory.")
        return
    
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    
    # Create visualizations
    print(f"\nCreating visualizations for {args.num_samples} samples...")
    save_path = visualize_predictions(
        model, test_loader, device, class_names,
        num_samples=args.num_samples,
        save_dir=args.results_dir,
        model_name=args.model
    )
    
    print(f"\n{'='*60}")
    print("Visualization Complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {save_path}")

if __name__ == '__main__':
    main()

