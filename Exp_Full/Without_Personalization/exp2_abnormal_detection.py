"""
Exp2: Abnormal Segment Detection with Sequence Timestamps
Detects abnormal segments and outputs corresponding sequence timestamps.
"""

import sys
sys.path.append('/home/heming/Desktop/AD-Moments-1/AD-Moments')

import json
import torch
import numpy as np
from tqdm import tqdm
from scipy import stats
from Model.ctms_model import CTMSModel
from datetime import datetime

# Configuration
DATA_PATH = '/home/heming/Desktop/AD-Moments-1/AD-Moments/sample_data/dataset_one_month.jsonl'
OUTPUT_DIR = '/home/heming/Desktop/AD-Moments-1/AD-Moments/Exp_Full/Without_Personalization/outputs'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Split configuration (80/20)
DROP_SUBJECTS = {'CN': [18, 42, 43], 'CI': []}
SPLIT_SEED = 0
TRAIN_RATIO = 0.8

# Model configuration
ALPHA = [0.5, 0.3, 0.1, 0.1]  # Circadian-dominant
SEQ_LEN = 30  # Number of events per window
STRIDE = 10   # Sliding window stride
BATCH_SIZE = 32

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

def sequence_to_model_input(sequence):
    """
    Convert raw sequence to model input format.
    
    Args:
        sequence: List of {ts: timestamp, action_id: int}
    
    Returns:
        dict with activity_ids and hours arrays
    """
    if len(sequence) == 0:
        return {
            'activity_ids': np.array([0]),  # Padding
            'hours': np.array([0.0])
        }
    
    activity_ids = []
    hours = []
    
    for item in sequence:
        # Action ID (should be 0-21 for 22 activities)
        act_id = item['action_id'] - 1  # Convert to 0-indexed
        if act_id < 0 or act_id >= 22:
            act_id = 0  # Default to 0 if invalid
        activity_ids.append(act_id)
        
        # Extract hour from timestamp
        dt = datetime.fromtimestamp(item['ts'])
        hour = dt.hour + dt.minute / 60.0  # Hour as float (0-23.99)
        hours.append(hour)
    
    return {
        'activity_ids': np.array(activity_ids, dtype=np.int64),
        'hours': np.array(hours, dtype=np.float32),
        'raw_sequence': sequence  # Keep original for reference
    }

def create_windows_from_sequence(activity_ids, hours, seq_len, stride):
    """Create sliding windows from activity sequences."""
    windows_ids = []
    windows_hours = []
    
    if len(activity_ids) < seq_len:
        # If sequence too short, return empty
        return [], []
    
    for i in range(0, len(activity_ids) - seq_len + 1, stride):
        windows_ids.append(activity_ids[i:i+seq_len])
        windows_hours.append(hours[i:i+seq_len])
    
    return windows_ids, windows_hours

def load_and_split_dataset():
    """Load dataset and perform train/test split."""
    data = []
    with open(DATA_PATH, 'r') as f:
        for line in f:
            subject = json.loads(line)
            label = subject['label']
            subject_id = subject['anon_id']
            
            if label == 'CN' and subject_id in DROP_SUBJECTS['CN']:
                continue
            if label == 'CI' and subject_id in DROP_SUBJECTS['CI']:
                continue
            
            # Convert sequence to model input
            model_input = sequence_to_model_input(subject['sequence'])
            subject['activity_ids'] = model_input['activity_ids']
            subject['hours'] = model_input['hours']
            subject['subject_id'] = subject_id
            subject['raw_sequence'] = subject['sequence']  # Keep original
            
            data.append(subject)
    
    # Separate by label
    cn_subjects = [s for s in data if s['label'] == 'CN']
    ci_subjects = [s for s in data if s['label'] == 'CI']
    
    # Split CN into train/test
    np.random.seed(SPLIT_SEED)
    n_cn = len(cn_subjects)
    n_train = int(n_cn * TRAIN_RATIO)
    
    cn_indices = np.random.permutation(n_cn)
    cn_train = [cn_subjects[i] for i in cn_indices[:n_train]]
    cn_test = [cn_subjects[i] for i in cn_indices[n_train:]]
    
    return cn_train, cn_test, ci_subjects

def compute_baseline(cn_train, model):
    """Compute baseline statistics from CN training set."""
    all_norms = {
        'circadian': [],
        'task': [],
        'movement': [],
        'social': []
    }
    
    for subject in tqdm(cn_train, desc="Computing baseline"):
        # Create windows
        windows_ids, windows_hours = create_windows_from_sequence(
            subject['activity_ids'], subject['hours'], SEQ_LEN, STRIDE
        )
        
        if len(windows_ids) == 0:
            continue
        
        n_windows = len(windows_ids)
        
        for i in range(0, n_windows, BATCH_SIZE):
            batch_end = min(i + BATCH_SIZE, n_windows)
            batch_ids = torch.LongTensor(np.array(windows_ids[i:batch_end])).to(DEVICE)
            batch_hours = torch.FloatTensor(np.array(windows_hours[i:batch_end])).to(DEVICE)
            
            with torch.no_grad():
                circ_emb = model.circadian_encoder(batch_ids, batch_hours)
                task_emb = model.task_encoder(batch_ids)
                move_emb = model.movement_encoder(batch_ids)
                soc_emb = model.social_encoder(batch_ids)
                
                all_norms['circadian'].extend(torch.norm(circ_emb, dim=1).cpu().numpy())
                all_norms['task'].extend(torch.norm(task_emb, dim=1).cpu().numpy())
                all_norms['movement'].extend(torch.norm(move_emb, dim=1).cpu().numpy())
                all_norms['social'].extend(torch.norm(soc_emb, dim=1).cpu().numpy())
    
    baseline_stats = {}
    for dim in all_norms:
        norms = np.array(all_norms[dim])
        baseline_stats[dim] = {
            'mean': float(np.mean(norms)),
            'std': float(np.std(norms))
        }
    
    return baseline_stats

def detect_abnormal_segments(subjects, model, baseline_stats, threshold_percentile=95):
    """Detect abnormal segments and extract sequence timestamps."""
    abnormal_records = []
    
    # First pass: collect all scores
    all_scores = []
    print("Computing scores for threshold...")
    for subject in tqdm(subjects):
        windows_ids, windows_hours = create_windows_from_sequence(
            subject['activity_ids'], subject['hours'], SEQ_LEN, STRIDE
        )
        
        if len(windows_ids) == 0:
            continue
        
        n_windows = len(windows_ids)
        
        for i in range(0, n_windows, BATCH_SIZE):
            batch_end = min(i + BATCH_SIZE, n_windows)
            batch_ids = torch.LongTensor(np.array(windows_ids[i:batch_end])).to(DEVICE)
            batch_hours = torch.FloatTensor(np.array(windows_hours[i:batch_end])).to(DEVICE)
            
            with torch.no_grad():
                circ_emb = model.circadian_encoder(batch_ids, batch_hours)
                task_emb = model.task_encoder(batch_ids)
                move_emb = model.movement_encoder(batch_ids)
                soc_emb = model.social_encoder(batch_ids)
                
                circ_norms = torch.norm(circ_emb, dim=1).cpu().numpy()
                task_norms = torch.norm(task_emb, dim=1).cpu().numpy()
                move_norms = torch.norm(move_emb, dim=1).cpu().numpy()
                soc_norms = torch.norm(soc_emb, dim=1).cpu().numpy()
                
                z_circ = np.abs((circ_norms - baseline_stats['circadian']['mean']) / baseline_stats['circadian']['std'])
                z_task = np.abs((task_norms - baseline_stats['task']['mean']) / baseline_stats['task']['std'])
                z_move = np.abs((move_norms - baseline_stats['movement']['mean']) / baseline_stats['movement']['std'])
                z_soc = np.abs((soc_norms - baseline_stats['social']['mean']) / baseline_stats['social']['std'])
                
                scores = (ALPHA[0] * z_circ + ALPHA[1] * z_task + ALPHA[2] * z_move + ALPHA[3] * z_soc)
                all_scores.extend(scores)
    
    threshold = np.percentile(all_scores, threshold_percentile)
    print(f"Abnormal threshold ({threshold_percentile}th percentile): {threshold:.4f}")
    
    # Second pass: detect abnormal segments and extract sequence timestamps
    print("Detecting abnormal segments and extracting timestamps...")
    for subject in tqdm(subjects):
        windows_ids, windows_hours = create_windows_from_sequence(
            subject['activity_ids'], subject['hours'], SEQ_LEN, STRIDE
        )
        
        if len(windows_ids) == 0:
            continue
        
        n_windows = len(windows_ids)
        window_scores = []
        
        for i in range(0, n_windows, BATCH_SIZE):
            batch_end = min(i + BATCH_SIZE, n_windows)
            batch_ids = torch.LongTensor(np.array(windows_ids[i:batch_end])).to(DEVICE)
            batch_hours = torch.FloatTensor(np.array(windows_hours[i:batch_end])).to(DEVICE)
            
            with torch.no_grad():
                circ_emb = model.circadian_encoder(batch_ids, batch_hours)
                task_emb = model.task_encoder(batch_ids)
                move_emb = model.movement_encoder(batch_ids)
                soc_emb = model.social_encoder(batch_ids)
                
                circ_norms = torch.norm(circ_emb, dim=1).cpu().numpy()
                task_norms = torch.norm(task_emb, dim=1).cpu().numpy()
                move_norms = torch.norm(move_emb, dim=1).cpu().numpy()
                soc_norms = torch.norm(soc_emb, dim=1).cpu().numpy()
                
                z_circ = np.abs((circ_norms - baseline_stats['circadian']['mean']) / baseline_stats['circadian']['std'])
                z_task = np.abs((task_norms - baseline_stats['task']['mean']) / baseline_stats['task']['std'])
                z_move = np.abs((move_norms - baseline_stats['movement']['mean']) / baseline_stats['movement']['std'])
                z_soc = np.abs((soc_norms - baseline_stats['social']['mean']) / baseline_stats['social']['std'])
                
                scores = (ALPHA[0] * z_circ + ALPHA[1] * z_task + ALPHA[2] * z_move + ALPHA[3] * z_soc)
                window_scores.extend(scores)
        
        # Find abnormal windows
        abnormal_indices = np.where(np.array(window_scores) > threshold)[0]
        
        if len(abnormal_indices) > 0:
            abnormal_segments = []
            
            for window_idx in abnormal_indices:
                # Get sequence events for this window (window starts at window_idx * STRIDE)
                start_event_idx = window_idx * STRIDE
                end_event_idx = start_event_idx + SEQ_LEN
                
                # Extract the actual sequence events
                sequence_events = subject['raw_sequence'][start_event_idx:end_event_idx]
                
                # Get timestamps
                if sequence_events:
                    min_ts = min(e['ts'] for e in sequence_events)
                    max_ts = max(e['ts'] for e in sequence_events)
                else:
                    min_ts = max_ts = None
                
                abnormal_segments.append({
                    'window_idx': int(window_idx),
                    'event_start_idx': int(start_event_idx),
                    'event_end_idx': int(end_event_idx),
                    'score': float(window_scores[window_idx]),
                    'num_events': len(sequence_events),
                    'timestamp_range': {
                        'min': min_ts,
                        'max': max_ts
                    },
                    'sequence_events': sequence_events  # Original sequence events
                })
            
            # Get overall time range for subject
            if subject['raw_sequence']:
                min_ts = min(e['ts'] for e in subject['raw_sequence'])
                max_ts = max(e['ts'] for e in subject['raw_sequence'])
            else:
                min_ts = max_ts = None
            
            abnormal_records.append({
                'subject_id': subject['subject_id'],
                'label': subject['label'],
                'total_windows': len(window_scores),
                'abnormal_windows': len(abnormal_indices),
                'abnormal_percentage': float(len(abnormal_indices) / len(window_scores) * 100),
                'mean_score': float(np.mean(window_scores)),
                'max_score': float(np.max(window_scores)),
                'time_range': {
                    'min_timestamp': min_ts,
                    'max_timestamp': max_ts
                },
                'abnormal_segments': abnormal_segments
            })
    
    return {
        'threshold': float(threshold),
        'threshold_percentile': threshold_percentile,
        'total_subjects_analyzed': len(subjects),
        'subjects_with_abnormalities': len(abnormal_records),
        'abnormal_records': abnormal_records
    }

def main():
    print("="*80)
    print("EXP2: ABNORMAL SEGMENT DETECTION WITH SEQUENCE TIMESTAMPS")
    print("="*80)
    
    # Load and split data
    print("\n1. Loading and processing dataset...")
    cn_train, cn_test, ci_subjects = load_and_split_dataset()
    print(f"   CN train: {len(cn_train)}, CN test: {len(cn_test)}, CI: {len(ci_subjects)}")
    
    # Initialize model
    print("\n2. Initializing CTMS model...")
    model = CTMSModel(d_model=64)
    model.to(DEVICE)
    model.eval()
    print(f"   Device: {DEVICE}")
    print(f"   Alpha weights: {ALPHA}")
    
    # Compute baseline
    print("\n3. Computing baseline from CN train...")
    baseline_stats = compute_baseline(cn_train, model)
    print("   Baseline statistics:")
    for dim, stats in baseline_stats.items():
        print(f"     {dim}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
    
    # Detect abnormal segments
    print("\n4. Detecting abnormal segments with sequence timestamps...")
    print("\n   Analyzing CI subjects...")
    ci_abnormal = detect_abnormal_segments(ci_subjects, model, baseline_stats, threshold_percentile=95)
    
    print("\n   Analyzing CN test subjects...")
    cn_abnormal = detect_abnormal_segments(cn_test, model, baseline_stats, threshold_percentile=95)
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"CI subjects with abnormalities: {ci_abnormal['subjects_with_abnormalities']}/{ci_abnormal['total_subjects_analyzed']}")
    print(f"CN subjects with abnormalities: {cn_abnormal['subjects_with_abnormalities']}/{cn_abnormal['total_subjects_analyzed']}")
    
    # Save results
    print("\n5. Saving results...")
    ci_path = os.path.join(OUTPUT_DIR, 'exp2_abnormal_sequences_CI.json')
    with open(ci_path, 'w') as f:
        json.dump(ci_abnormal, f, indent=2)
    print(f"   CI abnormal sequences saved to: {ci_path}")
    
    cn_path = os.path.join(OUTPUT_DIR, 'exp2_abnormal_sequences_CN.json')
    with open(cn_path, 'w') as f:
        json.dump(cn_abnormal, f, indent=2)
    print(f"   CN abnormal sequences saved to: {cn_path}")
    
    # Print sample
    if ci_abnormal['abnormal_records']:
        print("\n" + "="*80)
        print("SAMPLE OUTPUT (First CI subject with abnormalities)")
        print("="*80)
        sample = ci_abnormal['abnormal_records'][0]
        print(f"Subject ID: {sample['subject_id']}")
        print(f"Label: {sample['label']}")
        print(f"Abnormal windows: {sample['abnormal_windows']}/{sample['total_windows']} ({sample['abnormal_percentage']:.1f}%)")
        
        if sample['abnormal_segments']:
            seg = sample['abnormal_segments'][0]
            print(f"\nFirst abnormal segment:")
            print(f"  Window index: {seg['window_idx']}")
            print(f"  Anomaly score: {seg['score']:.4f}")
            print(f"  Number of events: {seg['num_events']}")
            if seg['sequence_events']:
                print(f"  Time range: {seg['timestamp_range']['min']} to {seg['timestamp_range']['max']}")
                print(f"  First 5 events: {seg['sequence_events'][:5]}")
    
    print("="*80)

if __name__ == '__main__':
    main()
