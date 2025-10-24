"""
CTMS Model - Complete Usage Example
===================================

This script demonstrates:
1. Loading data in your format
2. Training the CTMS model
3. Computing baselines from CN participants
4. Detecting anomalies
5. Personalizing fusion weights

Paper Reference: Section 2 - System Design
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import json

# Import CTMS components
from ctms_model_complete import CTMSModel, compute_anomaly_scores
from ctms_data import create_dataloaders, compute_baseline_statistics


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model: CTMSModel, 
                train_loader: DataLoader,
                optimizer: optim.Optimizer,
                criterion: nn.Module,
                device: str) -> float:
    """
    Train for one epoch.
    
    Args:
        model: CTMS model
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function (e.g., BCELoss)
        device: Device to train on
    
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc='Training')
    
    for batch in progress_bar:
        activity_ids = batch['activity_ids'].to(device)
        timestamps = batch['timestamps'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(activity_ids, timestamps)
        
        # Compute loss
        loss = criterion(outputs['output'], labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches


def evaluate(model: CTMSModel,
             val_loader: DataLoader,
             criterion: nn.Module,
             device: str) -> dict:
    """
    Evaluate model on validation set.
    
    Args:
        model: CTMS model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to evaluate on
    
    Returns:
        Dictionary with metrics: loss, accuracy, sensitivity, specificity
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Evaluating'):
            activity_ids = batch['activity_ids'].to(device)
            timestamps = batch['timestamps'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(activity_ids, timestamps)
            
            # Compute loss
            loss = criterion(outputs['output'], labels)
            total_loss += loss.item()
            
            # Store predictions
            preds = (outputs['output'] > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    accuracy = (all_preds == all_labels).mean()
    
    # Sensitivity (True Positive Rate): TP / (TP + FN)
    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Specificity (True Negative Rate): TN / (TN + FP)
    tn = ((all_preds == 0) & (all_labels == 0)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        'loss': total_loss / len(val_loader),
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity
    }


def train_model(model: CTMSModel,
                train_loader: DataLoader,
                val_loader: DataLoader,
                num_epochs: int = 50,
                learning_rate: float = 1e-4,
                device: str = 'cuda',
                save_path: str = 'ctms_model.pt'):
    """
    Full training loop.
    
    Args:
        model: CTMS model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs
        learning_rate: Learning rate
        device: Device to train on
        save_path: Path to save best model
    """
    model.to(device)
    
    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.BCELoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Val Sensitivity: {val_metrics['sensitivity']:.4f}")
        print(f"Val Specificity: {val_metrics['specificity']:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_metrics['loss'])
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy']
            }, save_path)
            print(f"✓ Saved best model to {save_path}")


# ============================================================================
# DEPLOYMENT FUNCTIONS
# ============================================================================

def detect_anomalies(model: CTMSModel,
                     data_loader: DataLoader,
                     baseline_stats: dict,
                     threshold_sigma: float = 3.0,
                     device: str = 'cuda') -> list:
    """
    Detect anomalous behavior segments.
    
    Paper Reference: Lines 203-210
    Anomalous Segment = 1 if AD_Score > τ_normal else 0
    
    Args:
        model: Trained CTMS model
        data_loader: Data loader for monitored participant
        baseline_stats: Baseline statistics from CN participants
        threshold_sigma: Detection threshold in standard deviations (default: 3.0)
        device: Device to run on
    
    Returns:
        List of anomalous segments with details
    """
    model.eval()
    model.to(device)
    
    anomalous_segments = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Detecting anomalies'):
            activity_ids = batch['activity_ids'].to(device)
            timestamps = batch['timestamps'].to(device)
            anon_ids = batch['anon_ids']
            
            # Get encodings
            outputs = model(activity_ids, timestamps, return_encodings_only=True)
            
            # Compute anomaly scores (Paper Lines 195-200)
            ad_scores, dim_scores = compute_anomaly_scores(
                outputs, baseline_stats, alpha=None
            )
            
            # Detect anomalies (Paper Line 205)
            anomalous_mask = ad_scores > threshold_sigma
            
            # Store anomalous segments
            for i in range(len(anon_ids)):
                if anomalous_mask[i]:
                    anomalous_segments.append({
                        'anon_id': anon_ids[i],
                        'ad_score': ad_scores[i].item(),
                        'circadian_score': dim_scores['circadian'][i].item(),
                        'task_score': dim_scores['task'][i].item(),
                        'movement_score': dim_scores['movement'][i].item(),
                        'social_score': dim_scores['social'][i].item(),
                        'cdi': outputs['cdi'][i].item(),
                        'tir': outputs['tir'][i].item(),
                        'me': outputs['me'][i].item(),
                        'sws': outputs['sws'][i].item(),
                        'activity_sequence': activity_ids[i].cpu().numpy().tolist(),
                        'timestamps': timestamps[i].cpu().numpy().tolist()
                    })
    
    print(f"Detected {len(anomalous_segments)} anomalous segments")
    return anomalous_segments


def personalize_weights(model: CTMSModel,
                        anomalous_segments: list,
                        learning_rate: float = 0.1):
    """
    Personalize fusion weights based on symptom manifestation patterns.
    
    Paper Reference: Section 3.6 - Dimensional Weight Personalization
    Lines 222-226: α_d^(t+1) = α_d^(t) + η * (α_{d,LLM} - α_d^(t))
    
    This is a simplified version. In practice, you'd:
    1. Send anomalous segments to LLM for analysis
    2. LLM identifies dominant dimensions
    3. Update weights accordingly
    
    Args:
        model: CTMS model
        anomalous_segments: List of detected anomalies
        learning_rate: Step size for weight adjustment (η in paper)
    """
    if not anomalous_segments:
        print("No anomalous segments to analyze")
        return
    
    # Analyze dimensional scores across all anomalies
    dim_scores = {
        'circadian': [],
        'task': [],
        'movement': [],
        'social': []
    }
    
    for segment in anomalous_segments:
        dim_scores['circadian'].append(segment['circadian_score'])
        dim_scores['task'].append(segment['task_score'])
        dim_scores['movement'].append(segment['movement_score'])
        dim_scores['social'].append(segment['social_score'])
    
    # Compute mean scores per dimension
    mean_scores = {
        k: np.mean(v) for k, v in dim_scores.items()
    }
    
    # Convert to normalized weights (this simulates LLM recommendation)
    total_score = sum(mean_scores.values())
    alpha_llm = torch.tensor([
        mean_scores['circadian'] / total_score,
        mean_scores['task'] / total_score,
        mean_scores['movement'] / total_score,
        mean_scores['social'] / total_score
    ])
    
    print(f"\nPersonalized weights based on {len(anomalous_segments)} anomalies:")
    print(f"  Circadian: {alpha_llm[0]:.3f}")
    print(f"  Task:      {alpha_llm[1]:.3f}")
    print(f"  Movement:  {alpha_llm[2]:.3f}")
    print(f"  Social:    {alpha_llm[3]:.3f}")
    
    # Update model weights (Paper Line 224)
    model.update_fusion_weights(alpha_llm, learning_rate=learning_rate)
    print("✓ Model weights updated")


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    """
    Complete workflow: Train → Baseline → Deploy → Personalize
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='CTMS Model Training and Deployment')
    parser.add_argument('--sequence_file', type=str, required=True,
                       help='Path to sequence data (JSON/JSONL)')
    parser.add_argument('--label_file', type=str, required=True,
                       help='Path to label data (JSON/JSONL)')
    parser.add_argument('--mode', type=str, choices=['train', 'deploy', 'personalize'],
                       default='train', help='Operation mode')
    parser.add_argument('--model_path', type=str, default='ctms_model.pt',
                       help='Path to save/load model')
    parser.add_argument('--baseline_path', type=str, default='baseline_stats.pt',
                       help='Path to save/load baseline statistics')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("CTMS Model - Activity-Based AD Detection")
    print("Paper Implementation: Section 2 - System Design")
    print("=" * 80)
    
    # ========================================================================
    # MODE 1: TRAINING
    # ========================================================================
    if args.mode == 'train':
        print("\n[MODE] Training Phase")
        print("-" * 80)
        
        # Create dataloaders
        print("\n1. Loading data...")
        train_loader, val_loader, test_loader = create_dataloaders(
            sequence_file=args.sequence_file,
            label_file=args.label_file,
            batch_size=args.batch_size,
            max_seq_len=500
        )
        
        # Initialize model
        print("\n2. Initializing model...")
        model = CTMSModel(d_model=128, num_activities=21, num_task_templates=20)
        print(f"✓ Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Train model
        print("\n3. Training model...")
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            device=args.device,
            save_path=args.model_path
        )
        
        # Compute baseline statistics from CN participants
        print("\n4. Computing baseline statistics from CN participants...")
        # Filter for CN samples in training set
        baseline_stats = compute_baseline_statistics(
            dataloader=train_loader,
            model=model,
            device=args.device
        )
        
        # Save baseline statistics
        torch.save(baseline_stats, args.baseline_path)
        print(f"✓ Saved baseline statistics to {args.baseline_path}")
        
        # Test model
        print("\n5. Testing model...")
        test_metrics = evaluate(model, test_loader, nn.BCELoss(), args.device)
        print(f"\nTest Results:")
        print(f"  Accuracy:    {test_metrics['accuracy']:.4f}")
        print(f"  Sensitivity: {test_metrics['sensitivity']:.4f}")
        print(f"  Specificity: {test_metrics['specificity']:.4f}")
    
    # ========================================================================
    # MODE 2: DEPLOYMENT (Anomaly Detection)
    # ========================================================================
    elif args.mode == 'deploy':
        print("\n[MODE] Deployment Phase - Anomaly Detection")
        print("-" * 80)
        
        # Load model
        print("\n1. Loading trained model...")
        model = CTMSModel(d_model=128, num_activities=21, num_task_templates=20)
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded model from {args.model_path}")
        
        # Load baseline statistics
        print("\n2. Loading baseline statistics...")
        baseline_stats = torch.load(args.baseline_path)
        print(f"✓ Loaded baseline statistics from {args.baseline_path}")
        
        # Load monitoring data
        print("\n3. Loading monitoring data...")
        _, _, deploy_loader = create_dataloaders(
            sequence_file=args.sequence_file,
            label_file=args.label_file,
            batch_size=args.batch_size,
            max_seq_len=500
        )
        
        # Detect anomalies
        print("\n4. Detecting anomalous behavior segments...")
        anomalous_segments = detect_anomalies(
            model=model,
            data_loader=deploy_loader,
            baseline_stats=baseline_stats,
            threshold_sigma=3.0,
            device=args.device
        )
        
        # Save results
        output_file = 'anomalous_segments.json'
        with open(output_file, 'w') as f:
            json.dump(anomalous_segments, f, indent=2)
        print(f"✓ Saved anomalous segments to {output_file}")
        
        # Print summary
        if anomalous_segments:
            print("\nAnomaly Summary:")
            print(f"  Total anomalous segments: {len(anomalous_segments)}")
            avg_ad_score = np.mean([s['ad_score'] for s in anomalous_segments])
            print(f"  Average AD score: {avg_ad_score:.3f}σ")
            
            # Dimensional breakdown
            avg_dim_scores = {
                'Circadian': np.mean([s['circadian_score'] for s in anomalous_segments]),
                'Task': np.mean([s['task_score'] for s in anomalous_segments]),
                'Movement': np.mean([s['movement_score'] for s in anomalous_segments]),
                'Social': np.mean([s['social_score'] for s in anomalous_segments])
            }
            print("\n  Average dimensional anomaly scores:")
            for dim, score in avg_dim_scores.items():
                print(f"    {dim}: {score:.3f}σ")
    
    # ========================================================================
    # MODE 3: PERSONALIZATION
    # ========================================================================
    elif args.mode == 'personalize':
        print("\n[MODE] Personalization Phase")
        print("-" * 80)
        
        # Load model
        print("\n1. Loading trained model...")
        model = CTMSModel(d_model=128, num_activities=21, num_task_templates=20)
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded model from {args.model_path}")
        
        # Load anomalous segments
        print("\n2. Loading anomalous segments...")
        with open('anomalous_segments.json', 'r') as f:
            anomalous_segments = json.load(f)
        print(f"✓ Loaded {len(anomalous_segments)} anomalous segments")
        
        # Personalize weights
        print("\n3. Personalizing fusion weights...")
        personalize_weights(
            model=model,
            anomalous_segments=anomalous_segments,
            learning_rate=0.1
        )
        
        # Save personalized model
        personalized_path = args.model_path.replace('.pt', '_personalized.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'anomalous_segments_analyzed': len(anomalous_segments)
        }, personalized_path)
        print(f"✓ Saved personalized model to {personalized_path}")
    
    print("\n" + "=" * 80)
    print("✓ Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()