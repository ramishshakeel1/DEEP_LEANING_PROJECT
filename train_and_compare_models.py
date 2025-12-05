"""
Train Multiple Models and Generate Comparison Visualizations

This script:
1. Trains ResNet18 and EfficientNet models
2. Saves results for each model
3. Generates comparison visualizations showing both models' predictions side by side

Run: python train_and_compare_models.py
"""

import os
import json
import time
import argparse
import subprocess
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Import from train_flowers.py
from train_flowers import (
    FlowerDataset, get_transforms, get_pretrained_model, BaselineCNN,
    train_model, evaluate_model, plot_training_history, plot_confusion_matrix
)

CONFIG = {
    'batch_size': 32,
    'epochs': 30,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'dropout_rate': 0.5,
    'image_size': 224,
    'num_classes': 102,
    'num_workers': 4,
    'use_mixed_precision': True,
    'early_stopping_patience': 10,
    'lr_scheduler_patience': 5,
    'lr_scheduler_factor': 0.5,
    'data_dir': 'dataset',
    'results_dir': 'results',
    'weights_dir': 'weights',
    'cat_to_name_path': 'cat_to_name.json',
}

def train_single_model(model_name, config, device):
    """Train a single model and return results."""
    print(f"\n{'='*70}")
    print(f"TRAINING {model_name.upper()}")
    print(f"{'='*70}\n")
    
    # Update config
    config['model_name'] = model_name
    if model_name == 'baseline':
        config['image_size'] = 200
    else:
        config['image_size'] = 224
    
    # Load class names
    with open(config['cat_to_name_path'], 'r') as f:
        cat_to_name = json.load(f)
    class_names = [cat_to_name[str(i+1)] for i in range(102)]
    
    # Create datasets
    print("Loading datasets...")
    train_transform = get_transforms('train', config['image_size'], False)
    val_transform = get_transforms('valid', config['image_size'])
    test_transform = get_transforms('test', config['image_size'])
    
    train_dataset = FlowerDataset(config['data_dir'], 'train', train_transform)
    val_dataset = FlowerDataset(config['data_dir'], 'valid', val_transform)
    test_dataset = FlowerDataset(config['data_dir'], 'test', test_transform)
    
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True,
        num_workers=config['num_workers'], pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], pin_memory=True
    )
    
    print(f"Train: {len(train_dataset)}, Valid: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create model
    print(f"\nCreating {model_name} model...")
    if model_name == 'baseline':
        model = BaselineCNN(
            num_classes=config['num_classes'],
            dropout_rate=config['dropout_rate'],
            image_size=config['image_size']
        ).to(device)
    else:
        model = get_pretrained_model(
            model_name,
            num_classes=config['num_classes'],
            freeze_backbone=False
        ).to(device)
    
    # Train model
    history, best_val_acc = train_model(model, train_loader, val_loader, config, device)
    
    # Load best model
    checkpoint = torch.load(os.path.join(config['weights_dir'], 
                                        f'best_{model_name}.pth'),
                           map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set
    print(f"\nEvaluating {model_name} on test set...")
    results = evaluate_model(model, test_loader, device, class_names)
    
    print(f"\n{model_name.upper()} Test Accuracy: {results['accuracy']:.2f}%")
    
    # Save plots
    plot_training_history(history, 
                         os.path.join(config['results_dir'], 
                                     f'training_curves_{model_name}.png'))
    plot_confusion_matrix(results['confusion_matrix'],
                         os.path.join(config['results_dir'],
                                     f'confusion_matrix_{model_name}.png'),
                         class_names)
    
    return {
        'model': model,
        'history': history,
        'results': results,
        'class_names': class_names,
        'test_loader': test_loader,
        'model_name': model_name
    }

def create_comparison_visualization(model1_data, model2_data, num_samples=12, save_dir='results'):
    """
    Create side-by-side comparison visualization of two models' predictions.
    
    Args:
        model1_data: Dictionary with model1 results (from train_single_model)
        model2_data: Dictionary with model2 results (from train_single_model)
        num_samples: Number of samples to visualize
        save_dir: Directory to save results
    """
    model1 = model1_data['model']
    model2 = model2_data['model']
    model1_name = model1_data['model_name']
    model2_name = model2_data['model_name']
    test_loader = model1_data['test_loader']
    class_names = model1_data['class_names']
    device = next(model1.parameters()).device
    
    # Ensure both models are on the correct device and in eval mode
    model1 = model1.to(device)
    model2 = model2.to(device)
    model1.eval()
    model2.eval()
    
    # We need to get original images, so create a custom dataset
    # Import the custom dataset class
    import sys
    import importlib.util
    spec = importlib.util.spec_from_file_location("visualize_predictions", "visualize_predictions.py")
    vis_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vis_module)
    FlowerDatasetWithOriginal = vis_module.FlowerDatasetWithOriginal
    custom_collate_fn = vis_module.custom_collate_fn
    
    # Create dataset that returns original images
    test_transform = get_transforms('test', 224, False)
    data_dir = CONFIG['data_dir']
    test_dataset_orig = FlowerDatasetWithOriginal(data_dir, 'test', test_transform)
    test_loader_orig = DataLoader(
        test_dataset_orig, batch_size=32, shuffle=False,
        num_workers=0, collate_fn=custom_collate_fn
    )
    
    # Collect predictions from both models
    all_images = []
    all_labels = []
    all_preds1 = []
    all_preds2 = []
    all_probs1 = []
    all_probs2 = []
    
    print("\nCollecting predictions from both models...")
    print(f"Device: {device}")
    print(f"Model1 device: {next(model1.parameters()).device}")
    print(f"Model2 device: {next(model2.parameters()).device}")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader_orig):
            images, labels, original_images, paths = batch_data
            images_gpu = images.to(device)
            labels = labels.to(device)
            
            # Model 1 predictions
            outputs1 = model1(images_gpu)
            probs1 = torch.softmax(outputs1, dim=1)
            _, preds1 = outputs1.max(1)
            
            # Model 2 predictions
            try:
                outputs2 = model2(images_gpu)
                probs2 = torch.softmax(outputs2, dim=1)
                _, preds2 = outputs2.max(1)
            except Exception as e:
                print(f"\nERROR with model2 (batch {batch_idx}): {e}")
                print(f"Input shape: {images_gpu.shape}")
                print(f"Model2 type: {type(model2)}")
                print(f"Model2 device: {next(model2.parameters()).device}")
                import traceback
                traceback.print_exc()
                raise
            
            # Store original images and predictions
            for i in range(len(images)):
                all_images.append(original_images[i])
                all_labels.append(labels[i].item())
                all_preds1.append(preds1[i].item())
                all_preds2.append(preds2[i].item())
                all_probs1.append(probs1[i].cpu().numpy())
                all_probs2.append(probs2[i].cpu().numpy())
    
    # Select interesting samples: mix of correct/incorrect for both models
    correct_both = [i for i in range(len(all_labels)) 
                   if all_labels[i] == all_preds1[i] and all_labels[i] == all_preds2[i]]
    incorrect_both = [i for i in range(len(all_labels)) 
                     if all_labels[i] != all_preds1[i] and all_labels[i] != all_preds2[i]]
    diff_predictions = [i for i in range(len(all_labels)) 
                       if all_preds1[i] != all_preds2[i]]
    
    # Select samples
    num_each = num_samples // 3
    selected = []
    
    if len(correct_both) > 0:
        selected.extend(np.random.choice(correct_both, 
                                        min(num_each, len(correct_both)), 
                                        replace=False).tolist())
    if len(incorrect_both) > 0:
        selected.extend(np.random.choice(incorrect_both, 
                                        min(num_each, len(incorrect_both)), 
                                        replace=False).tolist())
    if len(diff_predictions) > 0:
        selected.extend(np.random.choice(diff_predictions, 
                                        min(num_samples - len(selected), len(diff_predictions)), 
                                        replace=False).tolist())
    
    # Fill remaining slots randomly
    remaining = num_samples - len(selected)
    if remaining > 0:
        all_indices = list(range(len(all_labels)))
        available = [i for i in all_indices if i not in selected]
        if len(available) > 0:
            selected.extend(np.random.choice(available, 
                                            min(remaining, len(available)), 
                                            replace=False).tolist())
    
    np.random.shuffle(selected)
    selected = selected[:num_samples]
    
    # Create visualization
    rows = len(selected)
    fig, axes = plt.subplots(rows, 3, figsize=(18, 3.5 * rows))
    fig.patch.set_facecolor('white')
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, sample_idx in enumerate(selected):
        image = all_images[sample_idx]
        true_label = all_labels[sample_idx]
        pred1 = all_preds1[sample_idx]
        pred2 = all_preds2[sample_idx]
        probs1 = all_probs1[sample_idx]
        probs2 = all_probs2[sample_idx]
        
        is_correct1 = (true_label == pred1)
        is_correct2 = (true_label == pred2)
        
        # Column 1: Image
        ax_img = axes[idx, 0]
        ax_img.imshow(image)
        ax_img.axis('off')
        ax_img.set_aspect('equal')
        
        # Label text
        ax_img.text(0.5, -0.08, f"True: {true_label}", 
                   transform=ax_img.transAxes, ha='center', 
                   fontsize=11, color='black', weight='bold')
        
        # Column 2: Model 1 predictions
        ax_chart1 = axes[idx, 1]
        x_pos = np.arange(len(probs1))
        bars1 = ax_chart1.bar(x_pos, probs1, color='lightgray', width=0.8,
                             edgecolor='black', linewidth=0.5)
        
        bars1[true_label].set_color('blue')
        bars1[true_label].set_alpha(0.8)
        bars1[true_label].set_edgecolor('darkblue')
        bars1[true_label].set_linewidth(1.5)
        
        if not is_correct1:
            bars1[pred1].set_color('red')
            bars1[pred1].set_alpha(0.8)
            bars1[pred1].set_edgecolor('darkred')
            bars1[pred1].set_linewidth(1.5)
        
        ax_chart1.set_xlim(-0.5, len(probs1) - 0.5)
        ax_chart1.set_ylim(0, max(probs1) * 1.15)
        ax_chart1.set_xlabel('Class', fontsize=9)
        ax_chart1.set_ylabel('Confidence', fontsize=9)
        ax_chart1.tick_params(labelsize=8)
        ax_chart1.grid(True, alpha=0.2, axis='y', linestyle='--')
        ax_chart1.set_facecolor('white')
        
        for spine in ax_chart1.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)
        
        # Model 1 label and result
        pred_text1 = f"Pred: {pred1}"
        color1 = 'blue' if is_correct1 else 'red'
        ax_chart1.text(0.5, 1.05, f"{model1_name.upper()}\n{pred_text1}",
                      transform=ax_chart1.transAxes, ha='center',
                      fontsize=10, color=color1, weight='bold')
        
        result_text1 = 'True' if is_correct1 else 'False'
        result_color1 = 'blue' if is_correct1 else 'red'
        ax_chart1.text(0.98, 0.98, result_text1, transform=ax_chart1.transAxes,
                      ha='right', va='top', fontsize=22, color=result_color1,
                      weight='bold')
        
        # Column 3: Model 2 predictions
        ax_chart2 = axes[idx, 2]
        bars2 = ax_chart2.bar(x_pos, probs2, color='lightgray', width=0.8,
                             edgecolor='black', linewidth=0.5)
        
        bars2[true_label].set_color('blue')
        bars2[true_label].set_alpha(0.8)
        bars2[true_label].set_edgecolor('darkblue')
        bars2[true_label].set_linewidth(1.5)
        
        if not is_correct2:
            bars2[pred2].set_color('red')
            bars2[pred2].set_alpha(0.8)
            bars2[pred2].set_edgecolor('darkred')
            bars2[pred2].set_linewidth(1.5)
        
        ax_chart2.set_xlim(-0.5, len(probs2) - 0.5)
        ax_chart2.set_ylim(0, max(probs2) * 1.15)
        ax_chart2.set_xlabel('Class', fontsize=9)
        ax_chart2.set_ylabel('Confidence', fontsize=9)
        ax_chart2.tick_params(labelsize=8)
        ax_chart2.grid(True, alpha=0.2, axis='y', linestyle='--')
        ax_chart2.set_facecolor('white')
        
        for spine in ax_chart2.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)
        
        # Model 2 label and result
        pred_text2 = f"Pred: {pred2}"
        color2 = 'blue' if is_correct2 else 'red'
        ax_chart2.text(0.5, 1.05, f"{model2_name.upper()}\n{pred_text2}",
                      transform=ax_chart2.transAxes, ha='center',
                      fontsize=10, color=color2, weight='bold')
        
        result_text2 = 'True' if is_correct2 else 'False'
        result_color2 = 'blue' if is_correct2 else 'red'
        ax_chart2.text(0.98, 0.98, result_text2, transform=ax_chart2.transAxes,
                      ha='right', va='top', fontsize=22, color=result_color2,
                      weight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'model_comparison_{model1_name}_vs_{model2_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nComparison visualization saved to {save_path}")
    return save_path

def save_comparison_summary(model1_data, model2_data, save_dir='results'):
    """Save a text summary comparing both models."""
    model1_name = model1_data['model_name']
    model2_name = model2_data['model_name']
    results1 = model1_data['results']
    results2 = model2_data['results']
    history1 = model1_data['history']
    history2 = model2_data['history']
    
    summary_path = os.path.join(save_dir, f'model_comparison_summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("MODEL COMPARISON SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Model 1: {model1_name.upper()}\n")
        f.write(f"Model 2: {model2_name.upper()}\n\n")
        
        f.write("-"*70 + "\n")
        f.write("TEST ACCURACY COMPARISON\n")
        f.write("-"*70 + "\n")
        f.write(f"{model1_name.upper()}: {results1['accuracy']:.2f}%\n")
        f.write(f"{model2_name.upper()}: {results2['accuracy']:.2f}%\n")
        diff = results2['accuracy'] - results1['accuracy']
        f.write(f"Difference: {diff:+.2f}% ({model2_name} {'better' if diff > 0 else 'worse'})\n\n")
        
        f.write("-"*70 + "\n")
        f.write("TRAINING HISTORY\n")
        f.write("-"*70 + "\n")
        f.write(f"{model1_name.upper()} - Best Val Acc: {max(history1['val_acc']):.2f}%\n")
        f.write(f"{model2_name.upper()} - Best Val Acc: {max(history2['val_acc']):.2f}%\n\n")
        
        f.write("-"*70 + "\n")
        f.write("PREDICTION AGREEMENT\n")
        f.write("-"*70 + "\n")
        preds1 = np.array(results1['predictions'])
        preds2 = np.array(results2['predictions'])
        labels = np.array(results1['labels'])
        
        agreement = np.sum(preds1 == preds2) / len(preds1) * 100
        both_correct = np.sum((preds1 == labels) & (preds2 == labels)) / len(labels) * 100
        both_wrong = np.sum((preds1 != labels) & (preds2 != labels)) / len(labels) * 100
        model1_only_correct = np.sum((preds1 == labels) & (preds2 != labels)) / len(labels) * 100
        model2_only_correct = np.sum((preds1 != labels) & (preds2 == labels)) / len(labels) * 100
        
        f.write(f"Agreement: {agreement:.2f}%\n")
        f.write(f"Both correct: {both_correct:.2f}%\n")
        f.write(f"Both wrong: {both_wrong:.2f}%\n")
        f.write(f"{model1_name} only correct: {model1_only_correct:.2f}%\n")
        f.write(f"{model2_name} only correct: {model2_only_correct:.2f}%\n")
    
    print(f"Comparison summary saved to {summary_path}")

def main():
    parser = argparse.ArgumentParser(description='Train and Compare Models')
    parser.add_argument('--model1', type=str, default='resnet18',
                       choices=['baseline', 'resnet18', 'mobilenetv2', 'efficientnet'],
                       help='First model to train')
    parser.add_argument('--model2', type=str, default='efficientnet',
                       choices=['baseline', 'resnet18', 'mobilenetv2', 'efficientnet'],
                       help='Second model to train')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_samples', type=int, default=12,
                       help='Number of samples for comparison visualization')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training, only generate comparison (requires existing models)')
    
    args = parser.parse_args()
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: {device}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print(f"Using device: {device} (CPU)")
    
    # Update config
    CONFIG['epochs'] = args.epochs
    CONFIG['batch_size'] = args.batch_size
    
    # Create directories
    os.makedirs(CONFIG['results_dir'], exist_ok=True)
    os.makedirs(CONFIG['weights_dir'], exist_ok=True)
    
    model1_data = None
    model2_data = None
    
    # Train or load models
    if not args.skip_training:
        # Train Model 1
        model1_data = train_single_model(args.model1, CONFIG.copy(), device)
        
        # Train Model 2
        model1_data['model'] = model1_data['model'].cpu()  # Free GPU memory
        torch.cuda.empty_cache()
        
        model2_data = train_single_model(args.model2, CONFIG.copy(), device)
    else:
        # Load existing models
        print("Loading existing models...")
        from train_flowers import FlowerDataset, get_transforms, get_pretrained_model, BaselineCNN
        
        # Load class names
        with open(CONFIG['cat_to_name_path'], 'r') as f:
            cat_to_name = json.load(f)
        class_names = [cat_to_name[str(i+1)] for i in range(102)]
        
        # Create test dataset
        test_transform = get_transforms('test', 224, False)
        test_dataset = FlowerDataset(CONFIG['data_dir'], 'test', test_transform)
        test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], 
                               shuffle=False, num_workers=0)
        
        # Load Model 1
        if args.model1 == 'baseline':
            model1 = BaselineCNN(num_classes=102, dropout_rate=0.5, image_size=200)
        else:
            model1 = get_pretrained_model(args.model1, 102, False)
        
        checkpoint1 = torch.load(os.path.join(CONFIG['weights_dir'], 
                                              f'best_{args.model1}.pth'),
                                 map_location=device)
        model1.load_state_dict(checkpoint1['model_state_dict'])
        model1 = model1.to(device)
        model1.eval()
        
        results1 = evaluate_model(model1, test_loader, device, class_names)
        history1 = {'val_acc': [checkpoint1.get('val_acc', 0)]}
        
        model1_data = {
            'model': model1,
            'results': results1,
            'history': history1,
            'class_names': class_names,
            'test_loader': test_loader,
            'model_name': args.model1
        }
        
        # Load Model 2
        print(f"\nLoading {args.model2} model...")
        if args.model2 == 'baseline':
            model2 = BaselineCNN(num_classes=102, dropout_rate=0.5, image_size=200)
        else:
            model2 = get_pretrained_model(args.model2, 102, False)
        
        checkpoint_path2 = os.path.join(CONFIG['weights_dir'], f'best_{args.model2}.pth')
        if not os.path.exists(checkpoint_path2):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path2}")
        
        print(f"Loading checkpoint from {checkpoint_path2}...")
        checkpoint2 = torch.load(checkpoint_path2, map_location=device)
        model2.load_state_dict(checkpoint2['model_state_dict'])
        model2 = model2.to(device)
        model2.eval()
        
        # Verify model2 works with a test input
        print("Verifying model2 with test input...")
        test_input = torch.randn(1, 3, 224, 224).to(device)
        try:
            with torch.no_grad():
                test_output = model2(test_input)
            print(f"Model2 test output shape: {test_output.shape}")
        except Exception as e:
            print(f"ERROR: Model2 failed test forward pass: {e}")
            raise
        
        results2 = evaluate_model(model2, test_loader, device, class_names)
        history2 = {'val_acc': [checkpoint2.get('val_acc', 0)]}
        
        model2_data = {
            'model': model2,
            'results': results2,
            'history': history2,
            'class_names': class_names,
            'test_loader': test_loader,
            'model_name': args.model2
        }
    
    # Create comparison visualization
    print(f"\n{'='*70}")
    print("CREATING COMPARISON VISUALIZATION")
    print(f"{'='*70}\n")
    
    comparison_path = create_comparison_visualization(
        model1_data, model2_data, 
        num_samples=args.num_samples,
        save_dir=CONFIG['results_dir']
    )
    
    # Save comparison summary
    save_comparison_summary(model1_data, model2_data, CONFIG['results_dir'])
    
    print(f"\n{'='*70}")
    print("COMPARISON COMPLETE!")
    print(f"{'='*70}")
    print(f"Comparison visualization: {comparison_path}")
    print(f"Summary saved to: {CONFIG['results_dir']}/model_comparison_summary.txt")
    print(f"\nModel 1 ({args.model1}): {model1_data['results']['accuracy']:.2f}%")
    print(f"Model 2 ({args.model2}): {model2_data['results']['accuracy']:.2f}%")

if __name__ == '__main__':
    main()

