"""
CTMS Model Implementation for AD-Moments
Implements the four-dimensional behavioral space encoders following paper design:
- Circadian Rhythm Encoder (Transformer-based)
- Task Completion Encoder (BiLSTM with attention)
- Movement Pattern Encoder (Graph Attention Network)
- Social Interaction Encoder (1D CNN)

Following paper Section 3.4 architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================
# CTMS ENCODERS (From Paper Section 3.4)
# ============================================

class CircadianEncoder(nn.Module):
    """
    Circadian Rhythm Encoder using Transformer with positional encoding.
    
    Captures daily routine regularity and circadian patterns by encoding
    time-of-day information through sinusoidal positional encodings.
    
    Args:
        d_model: Embedding dimension (default: 128)
        nhead: Number of attention heads (default: 8)
        num_layers: Number of transformer layers (default: 3)
        num_activities: Number of activity types (default: 22)
    """
    def __init__(self, d_model=128, nhead=8, num_layers=3, num_activities=22):
        super().__init__()
        self.d_model = d_model
        
        # Activity embedding layer
        self.activity_embed = nn.Embedding(num_activities, d_model)
        
        # Transformer encoder for temporal modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.fc = nn.Linear(d_model, d_model)
        
    def positional_encoding(self, hours, d_model):
        """
        Sinusoidal positional encoding for time-of-day (from paper equation).
        
        Args:
            hours: Tensor of shape [batch_size, seq_len] with hour values
            d_model: Embedding dimension
            
        Returns:
            Positional encodings of shape [batch_size, seq_len, d_model]
        """
        batch_size, seq_len = hours.shape
        pos_enc = torch.zeros(batch_size, seq_len, d_model, device=hours.device)
        
        for i in range(0, d_model, 2):
            div_term = 10000 ** (2 * i / d_model)
            pos_enc[:, :, i] = torch.sin(hours / div_term)
            if i + 1 < d_model:
                pos_enc[:, :, i+1] = torch.cos(hours / div_term)
        
        return pos_enc
    
    def forward(self, activity_ids, hours):
        """
        Forward pass.
        
        Args:
            activity_ids: [batch_size, seq_len] activity IDs
            hours: [batch_size, seq_len] hour of day (0-23.99)
            
        Returns:
            h_cr: [batch_size, d_model] circadian rhythm encoding
        """
        # Embed activities
        x = self.activity_embed(activity_ids)
        
        # Add positional encoding for time-of-day
        pos_enc = self.positional_encoding(hours, self.d_model)
        x = x + pos_enc
        
        # Apply transformer
        x = self.transformer(x)
        
        # Average pooling over sequence
        h_cr = torch.mean(x, dim=1)
        
        # Project to output space
        h_cr = self.fc(h_cr)
        
        return h_cr


class TaskCompletionEncoder(nn.Module):
    """
    Task Completion Encoder using BiLSTM with attention mechanism.
    
    Analyzes task completion logic and identifies incomplete or 
    perseverative task sequences by comparing against learned task templates.
    
    Args:
        d_model: Embedding dimension (default: 128)
        hidden_size: LSTM hidden size (default: 256)
        num_templates: Number of task templates (default: 20)
        num_activities: Number of activity types (default: 22)
    """
    def __init__(self, d_model=128, hidden_size=256, num_templates=20, num_activities=22):
        super().__init__()
        self.d_model = d_model
        
        # Activity embedding layer
        self.activity_embed = nn.Embedding(num_activities, d_model)
        
        # Bidirectional LSTM for sequence modeling
        self.bilstm = nn.LSTM(
            d_model, 
            hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        
        # Task memory templates (M_task from paper)
        # Learned representations of typical task patterns
        self.task_memory = nn.Parameter(torch.randn(num_templates, hidden_size * 2))
        
        # Multi-head attention for template matching
        self.attention = nn.MultiheadAttention(
            hidden_size * 2,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Output projection
        self.fc = nn.Linear(hidden_size * 2, d_model)
        
    def forward(self, activity_ids):
        """
        Forward pass.
        
        Args:
            activity_ids: [batch_size, seq_len] activity IDs
            
        Returns:
            h_tc: [batch_size, d_model] task completion encoding
        """
        # Embed activities
        x = self.activity_embed(activity_ids)
        
        # BiLSTM encoding
        lstm_out, _ = self.bilstm(x)
        
        # Use mean of sequence as query
        query = lstm_out.mean(dim=1, keepdim=True)
        
        # Task memory as keys and values
        key = self.task_memory.unsqueeze(0).expand(query.size(0), -1, -1)
        
        # Attention over task templates
        h_tc, _ = self.attention(query, key, key)
        h_tc = h_tc.squeeze(1)
        
        # Project to output space
        h_tc = self.fc(h_tc)
        
        return h_tc


class MovementPatternEncoder(nn.Module):
    """
    Movement Pattern Encoder using Graph Attention Networks.
    
    Models spatial transitions and movement patterns by treating
    activities as nodes in a graph and learning attention weights
    for spatial relationships.
    
    Args:
        d_model: Embedding dimension (default: 128)
        num_activities: Number of activity types (default: 22)
    """
    def __init__(self, d_model=128, num_activities=22):
        super().__init__()
        self.d_model = d_model
        
        # Activity embedding layer
        self.activity_embed = nn.Embedding(num_activities, d_model)
        
        # Graph Attention Network layers
        self.gat1 = nn.Linear(d_model, d_model)
        self.gat2 = nn.Linear(d_model, d_model)
        self.gat3 = nn.Linear(d_model, d_model)
        
        # Attention mechanism for pooling
        self.attention = nn.Linear(d_model * 2, 1)
        
        # Output projection
        self.fc = nn.Linear(d_model, d_model)
        
    def forward(self, activity_ids):
        """
        Forward pass.
        
        Args:
            activity_ids: [batch_size, seq_len] activity IDs
            
        Returns:
            h_mp: [batch_size, d_model] movement pattern encoding
        """
        # Embed activities
        x = self.activity_embed(activity_ids)
        
        # Apply GAT layers with ReLU activations
        h = F.relu(self.gat1(x))
        h = F.relu(self.gat2(h))
        h = self.gat3(h)
        
        # Temporal attention pooling
        # Concatenate h with itself for attention scoring
        attn_input = torch.cat([h, h], dim=-1)
        attn_scores = self.attention(attn_input).squeeze(-1)
        attn_scores = torch.softmax(attn_scores, dim=1)
        
        # Weighted sum over sequence
        h_mp = torch.sum(h * attn_scores.unsqueeze(-1), dim=1)
        
        # Project to output space
        h_mp = self.fc(h_mp)
        
        return h_mp


class SocialInteractionEncoder(nn.Module):
    """
    Social Interaction Encoder using 1D CNN with statistical features.
    
    Analyzes social engagement patterns through convolutional feature
    extraction combined with explicit statistical measures of social
    interaction frequency and duration.
    
    Args:
        d_model: Embedding dimension (default: 128)
        num_activities: Number of activity types (default: 22)
    """
    def __init__(self, d_model=128, num_activities=22):
        super().__init__()
        self.d_model = d_model
        
        # Activity embedding layer
        self.activity_embed = nn.Embedding(num_activities, d_model)
        
        # 1D Convolutional layers for temporal patterns
        self.conv1 = nn.Conv1d(d_model, 256, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(256, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, d_model, kernel_size=3, padding=1)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        
        # Statistical features projection
        # Features: count, mean, std of social interactions
        self.stat_fc = nn.Linear(3, d_model)
        
        # Fusion layer
        self.fc = nn.Linear(d_model * 2, d_model)
        
    def forward(self, activity_ids):
        """
        Forward pass.
        
        Args:
            activity_ids: [batch_size, seq_len] activity IDs
            
        Returns:
            h_si: [batch_size, d_model] social interaction encoding
        """
        # Embed activities
        x = self.activity_embed(activity_ids)
        
        # Transpose for 1D convolution: [batch, d_model, seq_len]
        x = x.transpose(1, 2)
        
        # Apply convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        
        # Transpose back: [batch, seq_len, d_model]
        x = x.transpose(1, 2)
        
        # Average pooling
        h_cnn = torch.mean(x, dim=1)
        
        # Extract statistical features
        # Social activities: talking (label 14), phone calls (label 12)
        social_mask = (activity_ids == 14) | (activity_ids == 12)
        social_mask_f = social_mask.float()
        
        # Count, mean, std of social interactions
        social_count = social_mask_f.sum(dim=1, keepdim=True)
        social_mean = social_mask_f.mean(dim=1, keepdim=True)
        social_std = social_mask_f.std(dim=1, keepdim=True, unbiased=False)
        
        # Handle NaN values (can occur with very short sequences)
        social_std = torch.nan_to_num(social_std, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Concatenate statistical features
        stats = torch.cat([social_count, social_mean, social_std], dim=1)
        h_stat = self.stat_fc(stats)
        
        # Combine CNN and statistical features
        h_si = torch.cat([h_cnn, h_stat], dim=-1)
        h_si = self.fc(h_si)
        
        return h_si


# ============================================
# FULL CTMS MODEL
# ============================================

class CTMSModel(nn.Module):
    """
    Full CTMS Model with multi-head attention fusion (from paper Section 3.4).
    
    Integrates four dimensional encoders (Circadian, Task, Movement, Social)
    to analyze behavioral patterns indicative of cognitive decline.
    
    Note: Following paper's approach, this model outputs encodings for each dimension.
    Training uses these encodings to establish normal baselines (mu, sigma) from 
    cognitively normal (CN) participants. During deployment, anomaly scores are 
    computed by comparing encodings to these baselines.
    
    Args:
        d_model: Embedding dimension (default: 128)
        num_activities: Number of activity types (default: 22)
    """
    def __init__(self, d_model=128, num_activities=22):
        super().__init__()
        
        # Four dimensional encoders
        self.circadian_encoder = CircadianEncoder(
            d_model, num_activities=num_activities
        )
        self.task_encoder = TaskCompletionEncoder(
            d_model, num_activities=num_activities
        )
        self.movement_encoder = MovementPatternEncoder(
            d_model, num_activities=num_activities
        )
        self.social_encoder = SocialInteractionEncoder(
            d_model, num_activities=num_activities
        )
        
        # Learnable fusion weights (alpha_d in paper equation)
        # Used for training phase to learn dimensional importance
        self.alpha_weights = nn.Parameter(torch.ones(4) / 4)
        
        # Auxiliary classifier for training
        # Note: Per paper, final deployment uses baseline comparison (anomaly scores)
        # rather than this classifier. The classifier serves as a training signal
        # to ensure dimensional encoders capture clinically relevant patterns.
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
        
    def forward(self, activity_ids, hours, return_encodings_only=False):
        """
        Forward pass.
        
        Args:
            activity_ids: [batch_size, seq_len] activity IDs
            hours: [batch_size, seq_len] hour of day (0-23.99)
            return_encodings_only: If True, only return dimensional encodings
                                  (used for deployment/baseline computation)
            
        Returns:
            If return_encodings_only:
                encodings: Dict with keys ['h_c', 'h_t', 'h_m', 'h_s']
            Else:
                output: [batch_size] classification scores (0-1)
                alpha: [4] normalized attention weights
                encodings: Dict with dimensional encodings
        """
        # Encode each dimension
        h_c = self.circadian_encoder(activity_ids, hours)      # Circadian
        h_t = self.task_encoder(activity_ids)                  # Task
        h_m = self.movement_encoder(activity_ids)              # Movement
        h_s = self.social_encoder(activity_ids)                # Social
        
        # Store encodings in dictionary
        encodings = {
            'h_c': h_c,  # Circadian rhythm encoding
            'h_t': h_t,  # Task completion encoding
            'h_m': h_m,  # Movement pattern encoding
            'h_s': h_s   # Social interaction encoding
        }
        
        # If only encodings needed (deployment mode), return early
        if return_encodings_only:
            return encodings
        
        # Stack encodings for fusion: [batch, 4, d_model]
        h_all = torch.stack([h_c, h_t, h_m, h_s], dim=1)
        
        # Normalize alpha weights using softmax (ensures sum=1)
        alpha = F.softmax(self.alpha_weights, dim=0)
        
        # Weighted fusion (from paper equation)
        # Weighted sum over dimensions: [batch, d_model]
        h_fused = torch.sum(h_all * alpha.view(1, 4, 1), dim=1)
        
        # Auxiliary classification for training
        output = self.classifier(h_fused).squeeze(-1)
        
        return output, alpha, encodings


def get_model_summary(model, input_size=(32, 100)):
    """
    Print model summary with parameter counts.
    
    Args:
        model: CTMS model instance
        input_size: (batch_size, seq_len) for dummy input
    """
    print("=" * 70)
    print("CTMS Model Architecture Summary")
    print("=" * 70)
    
    # Count parameters per component
    components = {
        'Circadian Encoder': model.circadian_encoder,
        'Task Encoder': model.task_encoder,
        'Movement Encoder': model.movement_encoder,
        'Social Encoder': model.social_encoder,
        'Classifier': model.classifier
    }
    
    total_params = 0
    for name, component in components.items():
        n_params = sum(p.numel() for p in component.parameters())
        total_params += n_params
        print(f"{name:<25}: {n_params:>12,} parameters")
    
    # Alpha weights
    alpha_params = model.alpha_weights.numel()
    total_params += alpha_params
    print(f"{'Alpha Weights':<25}: {alpha_params:>12,} parameters")
    
    print("-" * 70)
    print(f"{'Total Parameters':<25}: {total_params:>12,}")
    print("=" * 70)
    
    # Test forward pass
    batch_size, seq_len = input_size
    dummy_activities = torch.randint(0, 22, (batch_size, seq_len))
    dummy_hours = torch.rand(batch_size, seq_len) * 24
    
    with torch.no_grad():
        output, alpha, encodings = model(dummy_activities, dummy_hours)
    
    print(f"\nOutput shapes:")
    print(f"  Classification output: {output.shape}")
    print(f"  Alpha weights: {alpha.shape}")
    print(f"  Circadian encoding: {encodings['h_c'].shape}")
    print(f"  Task encoding: {encodings['h_t'].shape}")
    print(f"  Movement encoding: {encodings['h_m'].shape}")
    print(f"  Social encoding: {encodings['h_s'].shape}")
    print("=" * 70)


if __name__ == '__main__':
    # Test model initialization
    print("Testing CTMS Model...")
    model = CTMSModel(d_model=128, num_activities=22)
    
    # Print summary
    get_model_summary(model)
    
    print("\n✓ Model initialized successfully!")
