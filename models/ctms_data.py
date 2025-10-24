"""
Data Processing Module for CTMS Model
=====================================

Handles the specific data format:
{
  "anon_id": 1,
  "month": "2023-02", 
  "sequence": [{"ts": timestamp, "action_id": id}, ...]
}

Activity mapping (21 activities, indices 0-20):
{
  "cleaning": 0,
  "communicating_or_socializing": 1,
  "doing_some_action_not_listed": 2,
  "dressing_or_undressing": 3,
  "drinking": 4,
  "exercising": 5,
  "handling_objects": 6,
  "no_one_or_senior_not_present": 7,
  "rubbing_hands": 8,
  "sitting": 9,
  "sleeping_or_lying": 10,
  "smoking": 11,
  "standing": 12,
  "static_action": 13,
  "stretching_or_yawning": 14,
  "taking_medicine_or_eating": 15,
  "touching_head_or_grooming": 16,
  "transitioning_from_sitting_or_lying": 17,
  "using_phone": 18,
  "walking": 19,
  "watching_TV": 20
}

Labels format:
{
  "anon_id": 1,
  "label": "CI",  # CI: Cognitively Impaired, CN: Cognitively Normal
  "gender": "F",
  "age": "85",
  "status": "MCI",  # MCI or Normal
  "scores": {"moca": 6.0, "zbi": 16.0, "dss": 5.0, "fas": 44.0}
}
"""

import torch
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


# Activity name to index mapping (as provided in data description)
ACTIVITY_MAPPING = {
    "cleaning": 0,
    "communicating_or_socializing": 1,
    "doing_some_action_not_listed": 2,
    "dressing_or_undressing": 3,
    "drinking": 4,
    "exercising": 5,
    "handling_objects": 6,
    "no_one_or_senior_not_present": 7,
    "rubbing_hands": 8,
    "sitting": 9,
    "sleeping_or_lying": 10,
    "smoking": 11,
    "standing": 12,
    "static_action": 13,
    "stretching_or_yawning": 14,
    "taking_medicine_or_eating": 15,
    "touching_head_or_grooming": 16,
    "transitioning_from_sitting_or_lying": 17,
    "using_phone": 18,
    "walking": 19,
    "watching_TV": 20
}

NUM_ACTIVITIES = 21  # Activities are indexed 0-20


class CTMSDataset(Dataset):
    """
    Dataset for CTMS model.
    
    Loads sequences from JSON/JSONL files and prepares them for model input.
    
    Args:
        sequence_file: Path to file with activity sequences (JSON/JSONL)
        label_file: Path to file with participant labels and metadata (JSON/JSONL)
        max_seq_len: Maximum sequence length (default: 500)
        min_seq_len: Minimum sequence length (default: 10)
        window_size: If set, split long sequences into windows (default: None)
        stride: Stride for sliding window (default: None, uses window_size)
    """
    def __init__(self, 
                 sequence_file: str,
                 label_file: str,
                 max_seq_len: int = 500,
                 min_seq_len: int = 10,
                 window_size: Optional[int] = None,
                 stride: Optional[int] = None):
        
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.window_size = window_size
        self.stride = stride if stride is not None else window_size
        
        # Load data
        self.sequences = self._load_sequences(sequence_file)
        self.labels = self._load_labels(label_file)
        
        # Create index mapping
        self.samples = self._create_samples()
        
        print(f"Loaded {len(self.samples)} samples from {len(self.sequences)} participants")
    
    def _load_sequences(self, filepath: str) -> Dict[int, List[Dict]]:
        """
        Load activity sequences from JSON/JSONL file.
        
        Returns:
            Dictionary mapping anon_id to list of monthly sequences
        """
        sequences = defaultdict(list)
        
        # Try JSONL format first
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    anon_id = data['anon_id']
                    sequences[anon_id].append(data)
        except json.JSONDecodeError:
            # Try JSON format
            with open(filepath, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        anon_id = item['anon_id']
                        sequences[anon_id].append(item)
                else:
                    raise ValueError("Unknown JSON format")
        
        return dict(sequences)
    
    def _load_labels(self, filepath: str) -> Dict[int, Dict]:
        """
        Load participant labels and metadata.
        
        Returns:
            Dictionary mapping anon_id to label information
        """
        labels = {}
        
        # Try JSONL format first
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    anon_id = data['anon_id']
                    labels[anon_id] = data
        except json.JSONDecodeError:
            # Try JSON format
            with open(filepath, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        anon_id = item['anon_id']
                        labels[anon_id] = item
                else:
                    raise ValueError("Unknown JSON format")
        
        return labels
    
    def _create_samples(self) -> List[Dict]:
        """
        Create training samples from sequences.
        
        If window_size is set, splits long sequences into windows.
        Otherwise, uses full sequences (truncated to max_seq_len).
        
        Returns:
            List of sample dictionaries with keys:
            - anon_id, month, activity_ids, timestamps, label
        """
        samples = []
        
        for anon_id, monthly_data in self.sequences.items():
            if anon_id not in self.labels:
                continue  # Skip if no label
            
            label_info = self.labels[anon_id]
            label = 1 if label_info['label'] == 'CI' else 0
            
            for month_data in monthly_data:
                sequence = month_data['sequence']
                
                # Extract activity IDs and timestamps
                activity_ids = [item['action_id'] for item in sequence]
                timestamps = [item['ts'] for item in sequence]
                
                # Filter out invalid sequences
                if len(activity_ids) < self.min_seq_len:
                    continue
                
                # Create windows if specified
                if self.window_size is not None:
                    for start_idx in range(0, len(activity_ids), self.stride):
                        end_idx = start_idx + self.window_size
                        
                        if end_idx > len(activity_ids):
                            break
                        
                        window_activities = activity_ids[start_idx:end_idx]
                        window_timestamps = timestamps[start_idx:end_idx]
                        
                        samples.append({
                            'anon_id': anon_id,
                            'month': month_data['month'],
                            'window_idx': start_idx,
                            'activity_ids': window_activities,
                            'timestamps': window_timestamps,
                            'label': label,
                            'label_info': label_info
                        })
                else:
                    # Use full sequence (truncate if needed)
                    if len(activity_ids) > self.max_seq_len:
                        activity_ids = activity_ids[:self.max_seq_len]
                        timestamps = timestamps[:self.max_seq_len]
                    
                    samples.append({
                        'anon_id': anon_id,
                        'month': month_data['month'],
                        'activity_ids': activity_ids,
                        'timestamps': timestamps,
                        'label': label,
                        'label_info': label_info
                    })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dictionary with:
            - activity_ids: [seq_len] tensor
            - timestamps: [seq_len] tensor
            - label: scalar tensor (0: CN, 1: CI)
            - anon_id: participant ID
            - seq_len: actual sequence length
        """
        sample = self.samples[idx]
        
        activity_ids = torch.tensor(sample['activity_ids'], dtype=torch.long)
        timestamps = torch.tensor(sample['timestamps'], dtype=torch.long)
        label = torch.tensor(sample['label'], dtype=torch.float32)
        
        return {
            'activity_ids': activity_ids,
            'timestamps': timestamps,
            'label': label,
            'anon_id': sample['anon_id'],
            'seq_len': len(activity_ids)
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.
    
    Pads sequences to the same length within a batch.
    
    Args:
        batch: List of samples from __getitem__
    
    Returns:
        Batched dictionary with padded tensors
    """
    # Find max sequence length in batch
    max_len = max(item['seq_len'] for item in batch)
    batch_size = len(batch)
    
    # Initialize padded tensors
    activity_ids_padded = torch.zeros(batch_size, max_len, dtype=torch.long)
    timestamps_padded = torch.zeros(batch_size, max_len, dtype=torch.long)
    labels = torch.zeros(batch_size, dtype=torch.float32)
    seq_lengths = torch.zeros(batch_size, dtype=torch.long)
    anon_ids = []
    
    # Fill in data
    for i, item in enumerate(batch):
        seq_len = item['seq_len']
        activity_ids_padded[i, :seq_len] = item['activity_ids']
        timestamps_padded[i, :seq_len] = item['timestamps']
        labels[i] = item['label']
        seq_lengths[i] = seq_len
        anon_ids.append(item['anon_id'])
    
    return {
        'activity_ids': activity_ids_padded,
        'timestamps': timestamps_padded,
        'labels': labels,
        'seq_lengths': seq_lengths,
        'anon_ids': anon_ids
    }


def create_dataloaders(sequence_file: str,
                       label_file: str,
                       batch_size: int = 32,
                       train_ratio: float = 0.7,
                       val_ratio: float = 0.15,
                       test_ratio: float = 0.15,
                       max_seq_len: int = 500,
                       window_size: Optional[int] = None,
                       num_workers: int = 4,
                       seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders with proper splitting.
    
    Args:
        sequence_file: Path to sequence data
        label_file: Path to label data
        batch_size: Batch size
        train_ratio: Proportion for training (default: 0.7)
        val_ratio: Proportion for validation (default: 0.15)
        test_ratio: Proportion for testing (default: 0.15)
        max_seq_len: Maximum sequence length
        window_size: Window size for splitting sequences (optional)
        num_workers: Number of DataLoader workers
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create full dataset
    dataset = CTMSDataset(
        sequence_file=sequence_file,
        label_file=label_file,
        max_seq_len=max_seq_len,
        window_size=window_size
    )
    
    # Split dataset
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    # Use random_split for reproducibility
    torch.manual_seed(seed)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Dataset split:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader


def compute_baseline_statistics(dataloader: DataLoader,
                                model: torch.nn.Module,
                                device: str = 'cuda') -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Compute baseline statistics from cognitively normal (CN) participants.
    
    This implements the baseline computation described in Paper Lines 187-191:
    μ_{d,normal} = (1/N) * sum(h_{d,i})
    σ_{d,normal} = sqrt((1/N) * sum((h_{d,i} - μ_{d,normal})^2))
    
    Args:
        dataloader: DataLoader containing only CN (label=0) samples
        model: Trained CTMS model
        device: Device to run on
    
    Returns:
        Dictionary with 'mean' and 'std' for each dimension:
        {
            'mean': {'h_c': tensor, 'h_t': tensor, 'h_m': tensor, 'h_s': tensor},
            'std': {'h_c': tensor, 'h_t': tensor, 'h_m': tensor, 'h_s': tensor}
        }
    """
    model.eval()
    model.to(device)
    
    # Collect encodings
    all_h_c, all_h_t, all_h_m, all_h_s = [], [], [], []
    
    with torch.no_grad():
        for batch in dataloader:
            # Filter for CN samples only (label == 0)
            cn_mask = batch['labels'] == 0
            if not cn_mask.any():
                continue
            
            activity_ids = batch['activity_ids'][cn_mask].to(device)
            timestamps = batch['timestamps'][cn_mask].to(device)
            
            # Get encodings
            outputs = model(activity_ids, timestamps, return_encodings_only=True)
            
            all_h_c.append(outputs['h_c'].cpu())
            all_h_t.append(outputs['h_t'].cpu())
            all_h_m.append(outputs['h_m'].cpu())
            all_h_s.append(outputs['h_s'].cpu())
    
    # Concatenate all encodings
    all_h_c = torch.cat(all_h_c, dim=0)
    all_h_t = torch.cat(all_h_t, dim=0)
    all_h_m = torch.cat(all_h_m, dim=0)
    all_h_s = torch.cat(all_h_s, dim=0)
    
    # Compute statistics (Paper Lines 189-190)
    baseline_stats = {
        'mean': {
            'h_c': all_h_c.mean(dim=0),
            'h_t': all_h_t.mean(dim=0),
            'h_m': all_h_m.mean(dim=0),
            'h_s': all_h_s.mean(dim=0)
        },
        'std': {
            'h_c': all_h_c.std(dim=0),
            'h_t': all_h_t.std(dim=0),
            'h_m': all_h_m.std(dim=0),
            'h_s': all_h_s.std(dim=0)
        }
    }
    
    print(f"Computed baseline statistics from {all_h_c.size(0)} CN samples")
    
    return baseline_stats


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test CTMS data processing')
    parser.add_argument('--sequence_file', type=str, required=True,
                       help='Path to sequence data (JSON/JSONL)')
    parser.add_argument('--label_file', type=str, required=True,
                       help='Path to label data (JSON/JSONL)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--max_seq_len', type=int, default=500,
                       help='Maximum sequence length')
    
    args = parser.parse_args()
    
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        sequence_file=args.sequence_file,
        label_file=args.label_file,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len
    )
    
    print("\nTesting data loading...")
    batch = next(iter(train_loader))
    
    print(f"\nBatch shapes:")
    print(f"  activity_ids: {batch['activity_ids'].shape}")
    print(f"  timestamps: {batch['timestamps'].shape}")
    print(f"  labels: {batch['labels'].shape}")
    print(f"  seq_lengths: {batch['seq_lengths'].shape}")
    
    print(f"\nSample data:")
    print(f"  First sequence length: {batch['seq_lengths'][0].item()}")
    print(f"  First 10 activities: {batch['activity_ids'][0, :10].tolist()}")
    print(f"  Label distribution: CN={((batch['labels']==0).sum().item())}, CI={(batch['labels']==1).sum().item()}")
    
    print("\n✓ Data processing working correctly!")