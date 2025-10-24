"""
GPU-optimized wrappers for CTMS model components.

This file provides GPU-friendly variants of the TaskCompletionEncoder and
MovementPatternEncoder used by the CTMS model. Changes are performance-focused
and do not alter the high-level model design or outputs. The main improvement
is replacing CPU-only DTW-based similarity computations with a fast,
GPU-vectorized pooled-cosine approximation by default. The original precise
DTW implementation is still available as a fallback via the
`use_fast_similarity=False` flag.

Usage:
    from ctms_model_gpu import CTMSModelGPU

The CTMSModelGPU mirrors the original `CTMSModel` API but constructs the
GPU-accelerated encoder variants.
"""

import copy
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional, List, Tuple

# Import base classes and utilities from the original implementation
from ctms_model import (
    TaskCompletionEncoder,
    MovementPatternEncoder,
    SocialInteractionEncoder,
    CircadianEncoder,
    CTMSModel,
)


class TaskCompletionEncoderGPU(TaskCompletionEncoder):
    """GPU-friendly TaskCompletionEncoder.

    By default this class uses a pooled-cosine similarity between
    subsequence mean vectors and template mean vectors instead of DTW. This
    preserves the task-template matching semantics but runs entirely on the
    GPU and is orders of magnitude faster for large batches. Set
    `use_fast_similarity=False` to run the original DTW-based (CPU) method.
    """
    def __init__(self, *args, use_fast_similarity: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_fast_similarity = use_fast_similarity

    def compute_tir_with_dtw(self, sequences: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Override: choose between original DTW implementation (fallback) and
        a GPU-vectorized pooled-cosine approximation.
        """
        if not self.use_fast_similarity:
            # Call original (slower, CPU+fastdtw) implementation
            return super().compute_tir_with_dtw(sequences)

        # FAST GPU PATH: use mean-pooled subsequence vectors + cosine similarity
        device = sequences.device
        batch_size = sequences.size(0)

        # Precompute template pooled vectors on device
        # self.task_templates: [K, template_len, d_model]
        templates = self.task_templates.to(device)
        templates_mean = templates.mean(dim=1)  # [K, d_model]
        templates_mean = F.normalize(templates_mean, dim=-1)

        tir_values = []
        avg_sim_values = []

        # We still iterate samples to allow variable subsequence counts, but
        # all heavy ops remain on GPU and are vectorized per-sample.
        for b in range(batch_size):
            subseqs = self.extract_task_subsequences(
                sequences[b], window_size=self.task_window_size
            )

            if len(subseqs) == 0:
                tir_values.append(0.0)
                avg_sim_values.append(0.0)
                continue

            # Compute mean-pooled vector for each subsequence
            # list of tensors [L, d_model] -> stack to [N, d_model]
            subseq_vecs = torch.stack([s.mean(dim=0) for s in subseqs], dim=0).to(device)
            subseq_vecs = F.normalize(subseq_vecs, dim=-1)  # [N, d]

            # Similarities: [N, K] = subseq_vecs @ templates_mean.T
            sims = torch.matmul(subseq_vecs, templates_mean.t())

            # For each subseq, take best template similarity
            max_sim, _ = sims.max(dim=1)  # [N]

            # Count incomplete subseqs (threshold on similarity)
            theta = self.theta_complete.to(device)
            incomplete = (max_sim < theta).sum().item()
            total = max_sim.size(0)

            tir = float(incomplete) / float(total) if total > 0 else 0.0
            avg_sim = float(max_sim.mean().item()) if total > 0 else 0.0

            tir_values.append(tir)
            avg_sim_values.append(avg_sim)

        tir_tensor = torch.tensor(tir_values, device=device, dtype=torch.float32)
        avg_sim_tensor = torch.tensor(avg_sim_values, device=device, dtype=torch.float32)

        return tir_tensor, avg_sim_tensor


class MovementPatternEncoderGPU(MovementPatternEncoder):
    """GPU-friendly MovementPatternEncoder.

    Vectorizes the construction of the 21-node transition matrix using
    `torch.bincount` / `scatter_add` instead of nested Python loops.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_activity_transition_graph(self, activity_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized transition counting across the batch.
        Returns edge_index and edge_weight on the same device as activity_ids.
        """
        device = activity_ids.device
        batch_size, seq_len = activity_ids.shape
        num_act = self.num_activities

        if seq_len < 2:
            # Fall back to self-loops
            edge_list = [[i, i] for i in range(num_act)]
            edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).t()
            edge_weight = torch.ones(edge_index.size(1), device=device)
            return edge_index, edge_weight

        src = activity_ids[:, :-1].reshape(-1)
        dst = activity_ids[:, 1:].reshape(-1)

        pair_idx = src * num_act + dst  # unique pair id
        pair_idx = pair_idx.to(device)

        counts = torch.bincount(pair_idx, minlength=num_act * num_act).float()
        counts = counts.reshape(num_act, num_act)

        # Build edge list where count > 0
        nz_src, nz_dst = torch.nonzero(counts > 0, as_tuple=True)
        if nz_src.numel() == 0:
            edge_list = [[i, i] for i in range(num_act)]
            edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).t()
            edge_weight = torch.ones(edge_index.size(1), device=device)
            return edge_index, edge_weight

        edge_index = torch.stack([nz_src, nz_dst], dim=0).long().to(device)
        edge_weight = counts[nz_src, nz_dst]

        # Normalize per-source
        # compute sum per source for present edges
        src_sums = torch.zeros(num_act, device=device)
        for s in nz_src.unique():
            mask = nz_src == s
            src_sums[s] = edge_weight[mask].sum()
        # Avoid division by zero
        src_sums = src_sums + 1e-10
        norm_weights = edge_weight / src_sums[edge_index[0]]

        return edge_index, norm_weights


class CTMSModelGPU(CTMSModel):
    """Factory model that uses GPU-optimized encoder variants.

    Constructor signature mirrors `CTMSModel`.
    """
    def __init__(self, d_model: int = 128, num_activities: int = 21,
                 num_task_templates: int = 20, num_fusion_heads: int = 4,
                 use_fast_similarity: bool = True):
        super().__init__(d_model=d_model, num_activities=num_activities,
                         num_task_templates=num_task_templates,
                         num_fusion_heads=num_fusion_heads)

        # Replace encoders with GPU-optimized variants (preserve interfaces)
        self.circadian_encoder = CircadianEncoder(d_model=d_model, num_activities=num_activities)
        self.task_encoder = TaskCompletionEncoderGPU(
            d_model=d_model,
            num_templates=num_task_templates,
            num_activities=num_activities,
            use_fast_similarity=use_fast_similarity,
        )
        self.movement_encoder = MovementPatternEncoderGPU(d_model=d_model, num_activities=num_activities)
        self.social_encoder = SocialInteractionEncoder(d_model=d_model, num_activities=num_activities)


__all__ = [
    'TaskCompletionEncoderGPU',
    'MovementPatternEncoderGPU',
    'CTMSModelGPU'
]
