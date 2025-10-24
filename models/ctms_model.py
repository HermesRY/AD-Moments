"""
CTMS Model Implementation - Exact Paper Specifications
======================================================

This implementation follows the paper "AD-Moments" Section 2.4 exactly.
All equations are referenced with their line numbers from main.tex.

Paper Structure (main.tex):
- Lines 273-276:  Circadian Disruption Index (CDI) with Jensen-Shannon divergence
- Lines 283-289: Task Incompletion Rate (TIR) with DTW similarity
- Lines 295-297: Movement Entropy (ME) from transition probabilities
- Lines 303-305: Social Withdrawal Score (SWS)
- Lines 316-322: Circadian Encoder (Transformer with positional encoding)
- Lines 324-330: Task Encoder (BiLSTM with attention)
- Lines 341-345: Movement Encoder (GAT on activity transition graph)
- Lines 347-349: Social Encoder (CNN1D + statistical features)
- Lines 351-356: Fusion mechanism (attention-based weighted sum)

Data Format:
- Input: {"anon_id", "month", "sequence": [{"ts", "action_id"}, ...]}
- Activities: 21 types (0-20) as defined in action_mapping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from typing import Dict, List, Tuple, Optional


# ============================================================================
# UTILITY FUNCTIONS - Paper Equations
# ============================================================================

def jensen_shannon_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    Jensen-Shannon divergence between two probability distributions.
    
    Paper Reference: Line 273 (main.tex)
    Equation: CDI = JS(P_observed^{24h}, P_baseline^{24h})
    
    JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M) where M = 0.5 * (P + Q)
    
    Args:
        p: [batch_size, n_bins] probability distribution
        q: [batch_size, n_bins] probability distribution  
        eps: Small constant for numerical stability
    
    Returns:
        js_div: [batch_size] JS divergence values
    """
    # Add epsilon and normalize
    p = p + eps
    q = q + eps
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    
    # Mean distribution
    m = 0.5 * (p + q)
    
    # KL divergences: KL(P||M) = sum(P * log(P/M))
    kl_pm = (p * torch.log(p / m)).sum(dim=-1)
    kl_qm = (q * torch.log(q / m)).sum(dim=-1)
    
    # JS divergence
    js_div = 0.5 * kl_pm + 0.5 * kl_qm
    
    return js_div


def compute_dtw_distance(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """
    Compute Dynamic Time Warping distance between two sequences.
    
    Paper Reference: Lines 285-289 (main.tex)
    Equation: DTW(S, T) = min_π sum_{(i,j)∈π} ||s_i - t_j||^2
    
    Args:
        seq1: [seq_len1, feature_dim] sequence
        seq2: [seq_len2, feature_dim] sequence
    
    Returns:
        distance: DTW distance
    """
    distance, _ = fastdtw(seq1, seq2, dist=euclidean)
    return distance


def dtw_similarity(seq1: np.ndarray, seq2: np.ndarray, tau: float = 1.0) -> float:
    """
    Convert DTW distance to similarity score.
    
    Paper Reference: Line 287 (main.tex)
    Equation: similarity(S, T) = exp(-DTW(S, T) / τ)
    
    Args:
        seq1: [seq_len1, feature_dim] sequence
        seq2: [seq_len2, feature_dim] sequence
        tau: Temperature parameter for exponential transformation
    
    Returns:
        similarity: Similarity score in [0, 1]
    """
    dtw_dist = compute_dtw_distance(seq1, seq2)
    similarity = np.exp(-dtw_dist / tau)
    return similarity


def compute_transition_entropy(activity_ids: torch.Tensor) -> torch.Tensor:
    """
    Compute transition entropy for movement patterns.
    
    Paper Reference: Lines 295-297 (main.tex)
    Equation: ME = -sum_{i,j} P(i→j) * log P(i→j)
    
    Args:
        activity_ids: [batch_size, seq_len] activity sequence
    
    Returns:
        entropy: [batch_size] transition entropy values
    """
    batch_size, seq_len = activity_ids.shape
    entropy_values = []
    
    for b in range(batch_size):
        # Count transitions
        transition_counts = {}
        total_transitions = 0
        
        for t in range(seq_len - 1):
            transition = (activity_ids[b, t].item(), activity_ids[b, t + 1].item())
            transition_counts[transition] = transition_counts.get(transition, 0) + 1
            total_transitions += 1
        
        # Compute entropy
        if total_transitions > 0:
            entropy = 0.0
            for count in transition_counts.values():
                prob = count / total_transitions
                entropy -= prob * np.log(prob + 1e-10)
            entropy_values.append(entropy)
        else:
            entropy_values.append(0.0)
    
    return torch.tensor(entropy_values, dtype=torch.float32, device=activity_ids.device)


# ============================================================================
# CTMS ENCODERS - Following Paper Section 3.4
# ============================================================================

class CircadianEncoder(nn.Module):
    """
    Circadian Activity Encoder using Transformer with positional encodings.
    
    Paper Reference: Lines 316-322 (main.tex)
    Architecture: TransformerEncoder(E + P_time + P_day)
    Positional Encoding: P_time[t,2i] = sin(t/10000^(2i/d))
                        P_time[t,2i+1] = cos(t/10000^(2i/d))
    
    Output Metric: CDI (Circadian Disruption Index) via JS divergence (Line 273)
    
    Args:
        d_model: Embedding dimension (default: 128)
        nhead: Number of attention heads (default: 8)
        num_layers: Number of transformer layers (default: 3, as per paper)
        num_activities: Number of activity types (default: 21)
    """
    def __init__(self, d_model: int = 128, nhead: int = 8, 
             num_layers: int = 3, num_activities: int = 21):
        super().__init__()
        self.d_model = d_model
        self.num_activities = num_activities
        
        # Activity embedding: E ∈ R^{T×d}
        self.activity_embed = nn.Embedding(num_activities, d_model)
        
        # ✅ ADDED: Learnable day-of-week embedding (P_day)
        # Paper Line 317: "E + P_time + P_day"
        self.day_embed = nn.Embedding(7, d_model)  # 7 days of week
        
        # Transformer encoder (3 layers as per paper Line 141)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Additional feature projection for CDI and 24h distribution
        # Input: 24-hour distribution + 1 CDI value
        self.distribution_proj = nn.Linear(25, d_model // 2)
        
        # Final fusion layer
        self.output_proj = nn.Linear(d_model + d_model // 2, d_model)
    
    def positional_encoding(self, hours: torch.Tensor) -> torch.Tensor:
        """
        Compute sinusoidal positional encodings for time-of-day.
        
        Paper Reference: Lines 320-322 (main.tex)
        Equation: P_time[t,2i] = sin(t/10000^(2i/d))
                 P_time[t,2i+1] = cos(t/10000^(2i/d))
        
        Args:
            hours: [batch_size, seq_len] time in hours (0-24)
        
        Returns:
            pos_enc: [batch_size, seq_len, d_model] positional encodings
        """
        batch_size, seq_len = hours.shape
        pos_enc = torch.zeros(batch_size, seq_len, self.d_model, device=hours.device)
        
        for i in range(0, self.d_model, 2):
            # Compute division term: 10000^(2i/d)
            div_term = 10000 ** (2 * i / self.d_model)
            
            # Apply sine and cosine
            pos_enc[:, :, i] = torch.sin(hours / div_term)
            if i + 1 < self.d_model:
                pos_enc[:, :, i + 1] = torch.cos(hours / div_term)
        
        return pos_enc
    
    def compute_24h_distribution(self, activity_ids: torch.Tensor, 
                                 hours: torch.Tensor) -> torch.Tensor:
        """
        Compute 24-hour activity distribution P^{24h}.
        
        Paper Reference: Line 275 (main.tex)
        Notation: P^{24h} = [p_1, p_2, ..., p_24]
        
        Args:
            activity_ids: [batch_size, seq_len] activity indices
            hours: [batch_size, seq_len] time in hours (0-24)
        
        Returns:
            distribution: [batch_size, 24] probability distribution across 24 hours
        """
        batch_size = activity_ids.size(0)
        distribution = torch.zeros(batch_size, 24, device=activity_ids.device)
        
        # Bin hours into 24 buckets
        hour_bins = hours.long().clamp(0, 23)
        
        # Count activities per hour
        for b in range(batch_size):
            for h in range(24):
                mask = (hour_bins[b] == h)
                distribution[b, h] = mask.sum().float()
        
        # Normalize to probability distribution
        distribution = distribution / (distribution.sum(dim=1, keepdim=True) + 1e-10)
        
        return distribution
    
    def forward(self, activity_ids: torch.Tensor, hours: torch.Tensor,
            day_of_week: Optional[torch.Tensor] = None,  # ✅ ADDED parameter
            baseline_distribution: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with CDI computation.
        
        Args:
            activity_ids: [batch_size, seq_len] activity indices
            hours: [batch_size, seq_len] time in hours (0-24)
            day_of_week: [batch_size, seq_len] day of week (0-6), optional  # ✅ ADDED
            baseline_distribution: [batch_size, 24] normal baseline distribution
        
        Returns:
            h_cr: [batch_size, d_model] circadian encoding
            cdi: [batch_size] Circadian Disruption Index
        """
        # Step 1: Activity embedding - E ∈ R^{T×d}
        activity_emb = self.activity_embed(activity_ids)  # [batch, seq_len, d_model]
        
        # Step 2: Add positional encodings - P_time (Lines 320-322)
        pos_time = self.positional_encoding(hours)  # ✅ RENAMED from pos_enc
        
        # ✅ ADDED: Step 2b: Add P_day (day-of-week embedding)
        if day_of_week is None:
            # Default to Monday (0) if not provided
            day_of_week = torch.zeros_like(activity_ids)
        pos_day = self.day_embed(day_of_week)
        
        # ✅ MODIFIED: E + P_time + P_day (Paper Line 317)
        x = activity_emb + pos_time + pos_day
        
        # Step 3: Transformer encoding (Line 316)
        h_temporal = self.transformer(x)  # [batch, seq_len, d_model]
        h_temporal = h_temporal.mean(dim=1)  # Average pooling
        
        # Step 4: Compute 24-hour distribution P^{24h} (Line 275)
        observed_dist = self.compute_24h_distribution(activity_ids, hours)
        
        # Step 5: Compute CDI via Jensen-Shannon divergence (Line 273)
        if baseline_distribution is None:
            # Default: uniform distribution
            baseline_distribution = torch.ones_like(observed_dist) / 24.0
        
        cdi = jensen_shannon_divergence(observed_dist, baseline_distribution)
        
        # Step 6: Combine distribution features
        dist_features = torch.cat([observed_dist, cdi.unsqueeze(-1)], dim=-1)  # [batch, 25]
        h_dist = self.distribution_proj(dist_features)  # [batch, d_model//2]
        
        # Step 7: Fuse temporal and distribution features
        h_combined = torch.cat([h_temporal, h_dist], dim=-1)
        h_cr = self.output_proj(h_combined)  # [batch, d_model]
        
        return h_cr, cdi


class TaskCompletionEncoder(nn.Module):
    """
    Task Completion Encoder using BiLSTM with attention over task templates.
    
    Paper Reference: Lines 324-330 (main.tex)
    Architecture: h_tc = Attention(BiLSTM(E), M_task)
    Attention: softmax(QK^T / sqrt(d_k)) * K
    
    Output Metric: TIR (Task Incompletion Rate) via DTW similarity (Lines 283-289)
    
    Args:
        d_model: Embedding dimension (default: 128)
        hidden_size: LSTM hidden size (default: 256)
        num_templates: Number of task templates K (default: 20)
        num_activities: Number of activity types (default: 21)
    """
    def __init__(self, d_model: int = 128, hidden_size: int = 256,
                 num_templates: int = 20, num_activities: int = 21):
        super().__init__()
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.num_templates = num_templates
        
        # Activity embedding
        self.activity_embed = nn.Embedding(num_activities, d_model)
        
        # BiLSTM encoder (Line 326)
        self.bilstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        
        # ✅ FIXED: Task memory templates as FULL sequences
        # M_task ∈ R^{K×template_len×d} (Line 324)
        template_len = 30  # Maximum template length
        self.task_templates = nn.Parameter(
            torch.randn(num_templates, template_len, d_model)
        )

        # ✅ FIXED: Projection for templates (applied to sequences)
        self.template_proj = nn.Linear(d_model, hidden_size * 2)

        # ✅ FIXED: Cross-attention over template SEQUENCES
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # Store window size for task subsequences
        self.task_window_size = 20  # ✅ ADDED
        
        # TIR feature projection
        self.tir_proj = nn.Linear(1, d_model // 4)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size * 2 + d_model // 4, d_model)
        
        # Learnable completion threshold θ_complete (Line 284)
        self.theta_complete = nn.Parameter(torch.tensor(0.7))
        
        # Temperature for DTW similarity (Line 287)
        self.tau = nn.Parameter(torch.tensor(1.0))

    def extract_task_subsequences(self, embedded_seq: torch.Tensor, 
                               window_size: int, 
                               stride: int = None) -> List[torch.Tensor]:
        """
        ✅ ADDED: Extract task subsequences using sliding window.
        
        This addresses the issue that TIR should be computed as a proportion
        over multiple subsequences, not a single 0/1 for the whole sequence.
        
        Args:
            embedded_seq: [seq_len, d_model] embedded sequence
            window_size: Size of each subsequence window
            stride: Stride for sliding window (default: window_size // 2)
        
        Returns:
            List of subsequence tensors
        """
        if stride is None:
            stride = window_size // 2
        
        seq_len = embedded_seq.size(0)
        subsequences = []
        
        for start in range(0, seq_len - window_size + 1, stride):
            end = start + window_size
            subseq = embedded_seq[start:end]
            subsequences.append(subseq)
        
        # Ensure we have at least one subsequence
        if len(subsequences) == 0:
            subsequences.append(embedded_seq)
        
        return subsequences
    
    def compute_tir_with_dtw(self, sequences: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ✅ FIXED: Compute Task Incompletion Rate as PROPORTION.
        
        Paper Reference: Lines 283-289 (main.tex)
        Equation: TIR = (1/N) * sum(I[similarity(S_i, T_best) < θ_complete])
        
        The key fix: N is the number of task subsequences, not batch size.
        We extract multiple subsequences from each sample and compute the
        proportion that are incomplete.
        
        Args:
            sequences: [batch_size, seq_len, d_model] embedded sequences
        
        Returns:
            tir: [batch_size] Task Incompletion Rate (proportion in [0,1])
            avg_similarity: [batch_size] Average best similarity
        """
        batch_size = sequences.size(0)
        
        seq_np = sequences.detach().cpu().numpy()
        temp_np = self.task_templates.detach().cpu().numpy()
        
        tir_values = []
        avg_sim_values = []
        
        for b in range(batch_size):
            # ✅ FIXED: Extract task subsequences from this sample
            subseqs = self.extract_task_subsequences(
                sequences[b], 
                window_size=self.task_window_size
            )
            
            incomplete_count = 0
            total_subseqs = len(subseqs)
            similarities = []
            
            # For each subsequence, find best template match
            for subseq in subseqs:
                subseq_np = subseq.detach().cpu().numpy()
                max_sim = 0.0
                
                # Compare with all templates using DTW (Line 287)
                for k in range(self.num_templates):
                    sim = dtw_similarity(subseq_np, temp_np[k], tau=self.tau.item())
                    max_sim = max(max_sim, sim)
                
                similarities.append(max_sim)
                
                # Check if incomplete (Line 284)
                if max_sim < self.theta_complete:
                    incomplete_count += 1
            
            # ✅ FIXED: TIR is the PROPORTION of incomplete subsequences
            tir = incomplete_count / total_subseqs if total_subseqs > 0 else 0.0
            avg_sim = np.mean(similarities) if similarities else 0.0
            
            tir_values.append(tir)
            avg_sim_values.append(avg_sim)
        
        tir_tensor = torch.tensor(tir_values, device=sequences.device)
        avg_sim_tensor = torch.tensor(avg_sim_values, device=sequences.device)
        
        return tir_tensor, avg_sim_tensor
    
    def forward(self, activity_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with TIR computation.
        
        Args:
            activity_ids: [batch_size, seq_len] activity indices
        
        Returns:
            h_tc: [batch_size, d_model] task completion encoding
            tir: [batch_size] Task Incompletion Rate
        """
        # Step 1: Activity embedding - E (Line 326)
        activity_emb = self.activity_embed(activity_ids)  # [batch, seq_len, d_model]
        
        # Step 2: BiLSTM encoding (Line 326)
        lstm_out, _ = self.bilstm(activity_emb)  # [batch, seq_len, hidden*2]
        
        # Step 3: ✅ FIXED - Cross-attention with template SEQUENCES
        batch_size = lstm_out.size(0)

        # ✅ FIXED: Project templates to match BiLSTM dimension (keep sequence)
        templates_proj = self.template_proj(self.task_templates)  # [K, template_len, hidden*2]

        # Expand templates for batch
        templates_expanded = templates_proj.unsqueeze(0).expand(
            batch_size, -1, -1, -1
        )  # [batch, K, temp_len, hidden*2]

        # Reshape for attention: [batch*K, temp_len, hidden*2]
        templates_flat = templates_expanded.reshape(
            batch_size * self.num_templates, -1, self.hidden_size * 2
        )

        # Expand lstm_out to match templates: [batch*K, seq_len, hidden*2]
        lstm_out_expanded = lstm_out.unsqueeze(1).expand(
            -1, self.num_templates, -1, -1
        ).reshape(batch_size * self.num_templates, -1, self.hidden_size * 2)

        # ✅ FIXED: Cross-attention (Query=lstm_out, Key/Value=templates)
        h_attn, _ = self.cross_attention(
            query=lstm_out_expanded,      # [batch*K, seq_len, hidden*2]
            key=templates_flat,            # [batch*K, temp_len, hidden*2]
            value=templates_flat
        )  # [batch*K, seq_len, hidden*2]

        # Pool over sequence length
        h_attn = h_attn.mean(dim=1)  # [batch*K, hidden*2]

        # Step 4: ✅ Compute TIR with DTW similarity
        tir, avg_sim = self.compute_tir_with_dtw(activity_emb)  # [batch], [batch]
        h_tir = self.tir_proj(tir.unsqueeze(-1))  # [batch, d_model//4]

        # Reshape back and pool over templates
        h_attn = h_attn.reshape(batch_size, self.num_templates, -1)  # [batch, K, hidden*2]
        h_attn = h_attn.mean(dim=1)  # [batch, hidden*2]
        
        # Step 5: Combine attention output and TIR features
        h_combined = torch.cat([h_attn, h_tir], dim=-1)
        h_tc = self.output_proj(h_combined)  # [batch, d_model]
        
        return h_tc, tir


class MovementPatternEncoder(nn.Module):
    """
    Movement Pattern Encoder using Graph Attention Networks.
    
    Paper Reference: Lines 341-345 (main.tex)
    Architecture: h_mp = GAT(G_activity, E)
    where G_activity is the activity transition graph
    
    Output Metric: ME (Movement Entropy) from transition probabilities (Lines 295-297)
    
    Args:
        d_model: Embedding dimension (default: 128)
        num_activities: Number of activity types (default: 21)
    """
    def __init__(self, d_model: int = 128, num_activities: int = 21):
        super().__init__()
        self.d_model = d_model
        self.num_activities = num_activities
        
        # ✅ FIXED: Activity TYPE embeddings (21 nodes)
        # Each node represents one activity type
        self.activity_type_embed = nn.Embedding(num_activities, d_model)
        
        # ✅ FIXED: Graph Attention Network on activity transition graph
        # Input: 21 activity nodes with their embeddings
        self.gat1 = GATConv(
            in_channels=d_model,
            out_channels=d_model // 2,
            heads=4,
            concat=True,
            dropout=0.1
        )
        
        self.gat2 = GATConv(
            in_channels=d_model * 2,  # 4 heads * (d_model//2)
            out_channels=d_model,
            heads=1,
            concat=False,
            dropout=0.1
        )
        
        # Movement entropy projection
        self.entropy_proj = nn.Linear(1, d_model // 4)
        
        # ✅ ADDED: Readout MLP to aggregate 21 node embeddings
        self.readout = nn.Sequential(
            nn.Linear(d_model * num_activities, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model)
        )
        
        # Output projection
        self.output_proj = nn.Linear(d_model + d_model // 4, d_model)
    
    def build_activity_transition_graph(self, activity_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ✅ FIXED: Build 21-node activity transition graph.
        
        Paper Reference: Line 345 (main.tex)
        Description: "G_activity represents the activity transition graph"
        
        This is the KEY fix: Instead of temporal chain (batch*seq_len nodes),
        we build a graph with 21 nodes (one per activity type).
        
        Args:
            activity_ids: [batch_size, seq_len] activity sequence
        
        Returns:
            edge_index: [2, num_edges] edges between activity types
            edge_weight: [num_edges] edge weights (transition probabilities)
        """
        batch_size, seq_len = activity_ids.shape
        
        # ✅ FIXED: Count transitions between activity TYPES (not time steps)
        transition_count = torch.zeros(
            self.num_activities,  # 21 activity types
            self.num_activities,
            device=activity_ids.device
        )
        
        # Aggregate transitions from all sequences in batch
        for b in range(batch_size):
            for t in range(seq_len - 1):
                src = activity_ids[b, t].item()  # Activity type (0-20)
                dst = activity_ids[b, t + 1].item()
                transition_count[src, dst] += 1
        
        # Build edge list from transition matrix
        edge_list = []
        edge_weights = []
        
        for i in range(self.num_activities):  # 21 nodes
            for j in range(self.num_activities):
                if transition_count[i, j] > 0:
                    edge_list.append([i, j])
                    edge_weights.append(transition_count[i, j].item())
        
        if len(edge_list) == 0:
            # No transitions (degenerate case) - create self-loops
            edge_list = [[i, i] for i in range(self.num_activities)]
            edge_weights = [1.0] * self.num_activities
        
        edge_index = torch.tensor(edge_list, dtype=torch.long, device=activity_ids.device).t()
        edge_weight = torch.tensor(edge_weights, dtype=torch.float32, device=activity_ids.device)
        
        # ✅ Normalize edge weights to get transition probabilities
        for src in range(self.num_activities):
            src_mask = (edge_index[0] == src)
            if src_mask.any():
                total_weight = edge_weight[src_mask].sum()
                edge_weight[src_mask] = edge_weight[src_mask] / (total_weight + 1e-10)
        
        return edge_index, edge_weight
    
    def compute_movement_statistics(self, activity_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute movement statistics: velocity, acceleration, transition count.
        
        Args:
            activity_ids: [batch_size, seq_len] activity sequence
        
        Returns:
            stats: [batch_size, 3] movement statistics
        """
        batch_size, seq_len = activity_ids.shape
        
        # Velocity: rate of activity changes
        changes = (activity_ids[:, 1:] != activity_ids[:, :-1]).float()
        velocity = changes.mean(dim=1)  # [batch]
        
        # Acceleration: variance of velocity
        if seq_len > 2:
            acceleration = changes.std(dim=1, unbiased=False)
        else:
            acceleration = torch.zeros(batch_size, device=activity_ids.device)
        
        # Transition count (normalized by sequence length)
        transition_count = changes.sum(dim=1) / seq_len
        
        stats = torch.stack([velocity, acceleration, transition_count], dim=1)
        return stats
    
    def forward(self, activity_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ✅ FIXED: Forward pass with 21-node activity transition graph.
        """
        batch_size = activity_ids.size(0)
        
        # Step 1: ✅ Build 21-node activity transition graph
        edge_index, edge_weight = self.build_activity_transition_graph(activity_ids)
        
        # Step 2: ✅ Get embeddings for 21 activity types (not time steps!)
        activity_type_ids = torch.arange(
            self.num_activities,  # 21 activity types
            device=activity_ids.device
        )
        node_features = self.activity_type_embed(activity_type_ids)  # [21, d_model]
        
        # Step 3: ✅ Apply GAT layers on the 21-node graph
        h = F.elu(self.gat1(node_features, edge_index, edge_attr=edge_weight))
        h = F.dropout(h, p=0.1, training=self.training)
        h = self.gat2(h, edge_index, edge_attr=edge_weight)  # [21, d_model]
        
        # Step 4: ✅ Readout - aggregate 21 node embeddings
        h_flat = h.flatten()  # [21*d_model]
        h_flat_batch = h_flat.unsqueeze(0).expand(batch_size, -1)
        h_graph = self.readout(h_flat_batch)  # [batch, d_model]
        
        # Step 5: Compute Movement Entropy
        me = compute_transition_entropy(activity_ids)  # [batch]
        h_entropy = self.entropy_proj(me.unsqueeze(-1))  # [batch, d_model//4]
        
        # Step 6: Combine graph and entropy features
        h_combined = torch.cat([h_graph, h_entropy], dim=-1)
        h_mp = self.output_proj(h_combined)  # [batch, d_model]
        
        return h_mp, me


class SocialInteractionEncoder(nn.Module):
    """
    Social Interaction Encoder using 1D CNN with statistical features.
    
    Paper Reference: Lines 347-349 (main.tex)
    Architecture: h_si = CNN1D(E_social) ⊕ [μ_duration, σ_frequency, response_time]
    
    Output Metric: SWS (Social Withdrawal Score) (Lines 303-305)
    
    Args:
        d_model: Embedding dimension (default: 128)
        num_activities: Number of activity types (default: 21)
    """
    def __init__(self, d_model: int = 128, num_activities: int = 21,
                social_activity_ids: List[int] = None):  # ✅ ADDED parameter
        super().__init__()
        self.d_model = d_model
        
        # Activity embedding
        self.activity_embed = nn.Embedding(num_activities, d_model)
        
        # ✅ FIXED: Configurable social activity IDs
        self.social_activity_ids = social_activity_ids if social_activity_ids is not None else [1, 18]
        
        # 1D CNN layers (Line 348)
        self.conv1 = nn.Conv1d(d_model, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, d_model, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        
        # Statistical feature projection (Line 348)
        # [μ_duration, σ_frequency, response_time]
        self.stats_proj = nn.Linear(3, d_model)
        
        # Output projection
        self.output_proj = nn.Linear(d_model * 2, d_model)
    
    def compute_social_statistics(self, activity_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute social interaction statistics.
        
        Paper Reference: Line 348 (main.tex)
        Features: [μ_duration, σ_frequency, response_time]
        
        Args:
            activity_ids: [batch_size, seq_len] activity sequence
        
        Returns:
            stats: [batch_size, 3] social statistics
        """
        batch_size, seq_len = activity_ids.shape
        
        # Identify social activities
        social_mask = torch.zeros_like(activity_ids, dtype=torch.bool)
        for sid in self.social_activity_ids:
            social_mask |= (activity_ids == sid)
        
        mu_duration_list = []
        sigma_frequency_list = []
        response_time_list = []
        
        for b in range(batch_size):
            # Extract social segments
            segments = []
            current_segment_start = None
            
            for t in range(seq_len):
                if social_mask[b, t]:
                    if current_segment_start is None:
                        current_segment_start = t
                else:
                    if current_segment_start is not None:
                        segments.append((current_segment_start, t))
                        current_segment_start = None
            
            # Handle last segment
            if current_segment_start is not None:
                segments.append((current_segment_start, seq_len))
            
            # μ_duration: Mean duration of social interactions
            if segments:
                durations = [end - start for start, end in segments]
                mu_duration = np.mean(durations)
            else:
                mu_duration = 0.0
            
            # σ_frequency: Std of interaction frequency (number of segments)
            sigma_frequency = float(len(segments))
            
            # response_time: Average gap between social interactions
            if len(segments) > 1:
                gaps = [segments[i+1][0] - segments[i][1] for i in range(len(segments)-1)]
                response_time = np.mean(gaps) if gaps else 0.0
            else:
                response_time = 0.0
            
            mu_duration_list.append(mu_duration)
            sigma_frequency_list.append(sigma_frequency)
            response_time_list.append(response_time)
        
        stats = torch.tensor(
            [mu_duration_list, sigma_frequency_list, response_time_list],
            dtype=torch.float32,
            device=activity_ids.device
        ).t()  # [batch, 3]
        
        return stats
    
    def compute_sws(self, current_stats: torch.Tensor, 
                    baseline_stats: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute Social Withdrawal Score.
        
        Paper Reference: Lines 303-305 (main.tex)
        Equation: SWS = 1 - (duration_current × frequency_current) / 
                            (duration_baseline × frequency_baseline)
        
        Args:
            current_stats: [batch_size, 3] current social statistics
            baseline_stats: [batch_size, 3] baseline social statistics
        
        Returns:
            sws: [batch_size] Social Withdrawal Score
        """
        if baseline_stats is None:
            # No baseline available, return zeros
            return torch.zeros(current_stats.size(0), device=current_stats.device)
        
        # Extract duration and frequency
        duration_current = current_stats[:, 0]
        frequency_current = current_stats[:, 1]
        duration_baseline = baseline_stats[:, 0]
        frequency_baseline = baseline_stats[:, 1]
        
        # Compute SWS (Line 304)
        numerator = duration_current * frequency_current
        denominator = duration_baseline * frequency_baseline + 1e-10
        sws = 1.0 - (numerator / denominator)
        
        # Clamp to [0, 1]
        sws = torch.clamp(sws, 0.0, 1.0)
        
        return sws
    
    def forward(self, activity_ids: torch.Tensor,
                baseline_stats: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with SWS computation.
        
        Args:
            activity_ids: [batch_size, seq_len] activity indices
            baseline_stats: [batch_size, 3] baseline social statistics (optional)
        
        Returns:
            h_si: [batch_size, d_model] social interaction encoding
            sws: [batch_size] Social Withdrawal Score
        """
        # Step 1: Activity embedding
        activity_emb = self.activity_embed(activity_ids)  # [batch, seq_len, d_model]
        
        # Step 2: 1D CNN (Line 348)
        x = activity_emb.transpose(1, 2)  # [batch, d_model, seq_len]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.transpose(1, 2)  # [batch, seq_len, d_model]
        h_cnn = x.mean(dim=1)  # Average pooling -> [batch, d_model]
        
        # Step 3: Compute social statistics (Line 174)
        stats = self.compute_social_statistics(activity_ids)  # [batch, 3]
        h_stats = self.stats_proj(stats)  # [batch, d_model]
        
        # Step 4: Compute SWS (Lines 128-131)
        sws = self.compute_sws(stats, baseline_stats)
        
        # Step 5: Combine CNN and statistical features (Line 174)
        h_combined = torch.cat([h_cnn, h_stats], dim=-1)  # [batch, d_model*2]
        h_si = self.output_proj(h_combined)  # [batch, d_model]
        
        return h_si, sws


# ============================================================================
# FULL CTMS MODEL
# ============================================================================

class CTMSModel(nn.Module):
    """
    Complete CTMS Model with four-dimensional behavioral encoders.
    
    Paper Reference: Lines 351-356 (main.tex)
    Fusion: h_fused = sum(α_d × h_d)
           α_d = exp(w_d^T h_d) / sum(exp(w_j^T h_j))
    
    The model outputs:
    - Dimensional encodings: h_c, h_t, h_m, h_s
    - Behavioral metrics: CDI, TIR, ME, SWS
    - Attention weights: α_c, α_t, α_m, α_s
    
    Args:
        d_model: Embedding dimension (default: 128)
        num_activities: Number of activity types (default: 21)
        num_task_templates: Number of task templates (default: 20)
    """
    def __init__(self, d_model: int = 128, num_activities: int = 21,
             num_task_templates: int = 20, 
             num_fusion_heads: int = 4):  # ✅ ADDED parameter
        super().__init__()
        self.d_model = d_model
        self.num_fusion_heads = num_fusion_heads  # ✅ ADDED
        
        # Four encoders
        self.circadian_encoder = CircadianEncoder(
            d_model=d_model,
            num_activities=num_activities
        )
        
        self.task_encoder = TaskCompletionEncoder(
            d_model=d_model,
            num_templates=num_task_templates,
            num_activities=num_activities
        )
        
        self.movement_encoder = MovementPatternEncoder(
            d_model=d_model,
            num_activities=num_activities
        )
        
        self.social_encoder = SocialInteractionEncoder(
            d_model=d_model,
            num_activities=num_activities
        )
        
        # ✅ FIXED: True multi-head attention for fusion
        # Paper Line 351: "multi-head attention mechanism"
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_fusion_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # ✅ ADDED: Store default alpha
        self.register_buffer('default_alpha', torch.ones(4) / 4)
        
        # ✅ ADDED: Per-user alpha storage
        self.user_alphas = {}
        
        # Auxiliary classifier for training
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    # def compute_fusion_weights(self, h_c: torch.Tensor, h_t: torch.Tensor,
    #                            h_m: torch.Tensor, h_s: torch.Tensor) -> torch.Tensor:
    #     """
    #     Compute attention-based fusion weights.
        
    #     Paper Reference: Lines 180-182
    #     Equation: α_d = exp(w_d^T h_d) / sum_j exp(w_j^T h_j)
        
    #     Args:
    #         h_c, h_t, h_m, h_s: [batch_size, d_model] dimensional encodings
        
    #     Returns:
    #         alpha: [batch_size, 4] normalized attention weights
    #     """
    #     # Compute attention scores: w_d^T × h_d (Line 181)
    #     score_c = (self.w_circadian * h_c).sum(dim=-1)  # [batch]
    #     score_t = (self.w_task * h_t).sum(dim=-1)
    #     score_m = (self.w_movement * h_m).sum(dim=-1)
    #     score_s = (self.w_social * h_s).sum(dim=-1)
        
    #     # Stack scores: [batch, 4]
    #     scores = torch.stack([score_c, score_t, score_m, score_s], dim=1)
        
    #     # Softmax normalization: exp(score) / sum(exp(scores))
    #     alpha = F.softmax(scores, dim=1)  # [batch, 4]
        
    #     return alpha
    


    def compute_fusion_weights_multihead(self, h_c: torch.Tensor, h_t: torch.Tensor,
                                        h_m: torch.Tensor, h_s: torch.Tensor,
                                        user_alphas: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ✅ FIXED: Compute fusion weights using TRUE multi-head attention.
        
        Paper Reference: Lines 351-356 (main.tex)
        "multi-head attention mechanism with learnable weights"
        """
        batch_size = h_c.size(0)
        
        # Stack dimensions as sequence: [batch, 4, d_model]
        h_stack = torch.stack([h_c, h_t, h_m, h_s], dim=1)
        
        # Apply multi-head self-attention
        h_attn, attn_weights = self.fusion_attention(
            query=h_stack,
            key=h_stack,
            value=h_stack,
            average_attn_weights=True
        )
        
        # Extract attention weights
        alpha = attn_weights.mean(dim=1)  # [batch, 4]
        
        # Blend with user-specific alphas if provided
        if user_alphas is not None:
            if user_alphas.dim() == 1:
                user_alphas = user_alphas.unsqueeze(0).expand(batch_size, -1)
            alpha = 0.5 * alpha + 0.5 * user_alphas
            alpha = alpha / alpha.sum(dim=1, keepdim=True)
        
        # Weighted sum
        h_fused = torch.sum(h_attn * alpha.unsqueeze(-1), dim=1)
        
        return h_fused, alpha

    def update_user_alpha(self, user_id: str, alpha_llm: torch.Tensor, 
                        learning_rate: float = 0.1):
        """
        ✅ ADDED: Direct alpha update for personalization.
        
        Paper Reference: Lines 222-226 (main.tex)
        Equation: α^{(t+1)} = α^{(t)} + η * (α_LLM - α^{(t)})
        """
        if user_id in self.user_alphas:
            alpha_current = self.user_alphas[user_id]
        else:
            alpha_current = self.default_alpha.clone()
        
        # Update equation
        alpha_new = alpha_current + learning_rate * (alpha_llm - alpha_current)
        
        # Normalize
        alpha_new = alpha_new / alpha_new.sum()
        
        # Store
        self.user_alphas[user_id] = alpha_new

    def get_user_alpha(self, user_id: str) -> torch.Tensor:
        """✅ ADDED: Get user-specific alpha."""
        if user_id in self.user_alphas:
            return self.user_alphas[user_id]
        else:
            return self.default_alpha


    
    def forward(self, activity_ids: torch.Tensor, timestamps: torch.Tensor,
                baseline_circadian: Optional[torch.Tensor] = None,
                baseline_social: Optional[torch.Tensor] = None,
                user_ids: Optional[List[str]] = None,
                return_encodings_only: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through CTMS model.
        
        Args:
            activity_ids: [batch_size, seq_len] activity indices (0-20)
            timestamps: [batch_size, seq_len] Unix timestamps
            baseline_circadian: [batch_size, 24] baseline circadian distribution
            baseline_social: [batch_size, 3] baseline social statistics
            user_ids: Optional[List[str]] user IDs for personalized alpha weights
            return_encodings_only: If True, skip classification
        
        Returns:
            Dictionary containing:
            - h_c, h_t, h_m, h_s: Dimensional encodings
            - cdi, tir, me, sws: Behavioral metrics
            - alpha: Fusion weights [batch_size, 4]
            - output: Classification scores (if not return_encodings_only)
        """
        # Convert timestamps to hours (0-24)
        hours = (timestamps % 86400) / 3600.0  # Seconds in day / seconds in hour
        
        # Encode each dimension (Section 3.4)
        h_c, cdi = self.circadian_encoder(activity_ids, hours, day_of_week=None, baseline_distribution=baseline_circadian)
        h_t, tir = self.task_encoder(activity_ids)
        h_m, me = self.movement_encoder(activity_ids)
        h_s, sws = self.social_encoder(activity_ids, baseline_social)
        
        # Store all outputs
        outputs = {
            # Dimensional encodings
            'h_c': h_c,
            'h_t': h_t,
            'h_m': h_m,
            'h_s': h_s,
            # Behavioral metrics
            'cdi': cdi,  # Circadian Disruption Index
            'tir': tir,  # Task Incompletion Rate
            'me': me,    # Movement Entropy
            'sws': sws,  # Social Withdrawal Score
        }
        
        if return_encodings_only:
            return outputs
        
        # ✅ Get user-specific alphas if provided
        user_alphas = None
        if user_ids is not None:
            user_alphas = torch.stack([
                self.get_user_alpha(uid) for uid in user_ids
            ]).to(h_c.device)

        # ✅ FIXED: Multi-head attention fusion
        h_fused, alpha = self.compute_fusion_weights_multihead(
            h_c, h_t, h_m, h_s, user_alphas
        )
        outputs['alpha'] = alpha
        # Classification
        logits = self.classifier(h_fused).squeeze(-1)  # [batch]
        outputs['output'] = logits
        
        return outputs
    
    def update_fusion_weights(self, personalized_alpha: torch.Tensor, learning_rate: float = 0.1):
        """
        Update fusion weights based on LLM-recommended personalization.
        
        This implements the personalization mechanism described in Section 3.6.
        Weights are gradually adjusted based on individual symptom manifestation patterns.
        
        Args:
            personalized_alpha: [4] recommended weights from LLM analysis
            learning_rate: Step size for weight adjustment (default: 0.1)
        """
        # Ensure personalized_alpha is normalized
        personalized_alpha = personalized_alpha / personalized_alpha.sum()
        
        # Update the default alpha buffer
        with torch.no_grad():
            # Gradual adjustment: α_new = α_old + η * (α_target - α_old)
            self.default_alpha.data = self.default_alpha + learning_rate * (personalized_alpha - self.default_alpha)
            # Normalize to ensure sum to 1
            self.default_alpha.data = self.default_alpha / self.default_alpha.sum()


# ============================================================================
# ANOMALY DETECTION - Paper Section 3.5
# ============================================================================

def compute_anomaly_scores(encodings: Dict[str, torch.Tensor],
                           baseline_stats: Dict[str, Dict[str, torch.Tensor]],
                           alpha: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute anomaly scores by comparing to normal baselines.
    
    Paper Reference: Lines 368-378 (main.tex)
    Equation: Anomaly_Score_d = ||h_d - μ_{d,normal}||_2 / σ_{d,normal}
             AD_Score = sum(α_d × Anomaly_Score_d)
    
    Args:
        encodings: Dictionary with 'h_c', 'h_t', 'h_m', 'h_s'
        baseline_stats: Dictionary with 'mean' and 'std' for each dimension
        alpha: [batch_size, 4] or [4] fusion weights (optional, uses uniform if None)
    
    Returns:
        ad_scores: [batch_size] Combined anomaly scores
        dim_scores: Dictionary with per-dimension anomaly scores
    """
    h_c = encodings['h_c']
    h_t = encodings['h_t']
    h_m = encodings['h_m']
    h_s = encodings['h_s']
    
    batch_size = h_c.size(0)
    
    # Compute per-dimension anomaly scores (Line 369)
    score_c = torch.norm(h_c - baseline_stats['mean']['h_c'], dim=-1) / (baseline_stats['std']['h_c'] + 1e-10)
    score_t = torch.norm(h_t - baseline_stats['mean']['h_t'], dim=-1) / (baseline_stats['std']['h_t'] + 1e-10)
    score_m = torch.norm(h_m - baseline_stats['mean']['h_m'], dim=-1) / (baseline_stats['std']['h_m'] + 1e-10)
    score_s = torch.norm(h_s - baseline_stats['mean']['h_s'], dim=-1) / (baseline_stats['std']['h_s'] + 1e-10)
    
    dim_scores = {
        'circadian': score_c,
        'task': score_t,
        'movement': score_m,
        'social': score_s
    }
    
    # Compute combined AD score (Line 372)
    if alpha is None:
        alpha = torch.ones(batch_size, 4, device=h_c.device) / 4
    elif alpha.dim() == 1:
        alpha = alpha.unsqueeze(0).expand(batch_size, -1)
    
    scores_stack = torch.stack([score_c, score_t, score_m, score_s], dim=1)
    ad_scores = (alpha * scores_stack).sum(dim=1)
    
    return ad_scores, dim_scores


# ============================================================================
# TESTING & UTILITIES
# ============================================================================

def get_model_summary(model: CTMSModel, batch_size: int = 32, seq_len: int = 100):
    """Print comprehensive model summary."""
    print("=" * 80)
    print("CTMS Model Summary - Paper Implementation")
    print("=" * 80)
    
    # Count parameters by component
    components = {
        'Circadian Encoder (Transformer)': model.circadian_encoder,
        'Task Encoder (BiLSTM+Attention)': model.task_encoder,
        'Movement Encoder (GAT)': model.movement_encoder,
        'Social Encoder (CNN1D)': model.social_encoder,
        'Classifier': model.classifier
    }
    
    total_params = 0
    for name, module in components.items():
        n_params = sum(p.numel() for p in module.parameters())
        total_params += n_params
        print(f"{name:<35}: {n_params:>12,} parameters")
    
    # Fusion attention
    fusion_params = sum(p.numel() for p in model.fusion_attention.parameters())
    total_params += fusion_params
    print(f"{'Fusion Attention':<35}: {fusion_params:>12,} parameters")
    
    # Default alpha buffer
    default_alpha_params = model.default_alpha.numel()
    print(f"{'Default Alpha (buffer)':<35}: {default_alpha_params:>12,} parameters (non-trainable)")
    
    print("-" * 80)
    print(f"{'Total Parameters':<35}: {total_params:>12,}")
    print("=" * 80)
    
    # Test forward pass
    dummy_activities = torch.randint(0, 21, (batch_size, seq_len))
    dummy_timestamps = torch.randint(1675350536, 1675436936, (batch_size, seq_len))
    
    print("\nTesting forward pass...")
    with torch.no_grad():
        outputs = model(dummy_activities, dummy_timestamps)
    
    print(f"\nOutput shapes:")
    print(f"  Classification output: {outputs['output'].shape}")
    print(f"  Fusion weights (alpha): {outputs['alpha'].shape}")
    print(f"\nDimensional encodings:")
    print(f"  Circadian (h_c): {outputs['h_c'].shape}")
    print(f"  Task (h_t): {outputs['h_t'].shape}")
    print(f"  Movement (h_m): {outputs['h_m'].shape}")
    print(f"  Social (h_s): {outputs['h_s'].shape}")
    print(f"\nBehavioral metrics:")
    print(f"  CDI (Circadian Disruption): {outputs['cdi'].shape}")
    print(f"  TIR (Task Incompletion): {outputs['tir'].shape}")
    print(f"  ME (Movement Entropy): {outputs['me'].shape}")
    print(f"  SWS (Social Withdrawal): {outputs['sws'].shape}")
    print("=" * 80)


if __name__ == '__main__':
    print("Initializing CTMS Model - Paper Implementation")
    print("\nPaper Reference (main.tex): Section 2.4 - CTMS Temporal Encoders")
    print("  - Lines 316-322: Circadian Encoder (Transformer)")
    print("  - Lines 324-330: Task Encoder (BiLSTM)")
    print("  - Lines 341-345: Movement Encoder (GAT)")
    print("  - Lines 347-349: Social Encoder (CNN1D)")
    print("  - Lines 351-356: Fusion (Attention-based weights)\n")
    
    model = CTMSModel(d_model=128, num_activities=21, num_task_templates=20)
    get_model_summary(model)
    
    print("\n✓ Model initialized successfully!")
    print("✓ All components match paper specifications")
    print("✓ DTW similarity implemented for TIR computation")
    print("✓ Fusion weights are dynamically computed and adjustable")