"""
KV-Cached Multi-Head Attention for LLM Inference - AI CODEFIX 2025 HARD_2

This module implements an optimized attention mechanism with Key-Value caching
for efficient autoregressive generation in Large Language Models.

Your task: Debug this code to pass all validation tests.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict


class KVCachedMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with KV-Cache optimization for LLM inference.

    This implementation caches Key and Value projections across generation steps
    to avoid redundant computation during autoregressive decoding.

    Standard transformer attention formula:
        Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    With KV-caching:
        - Cache K and V from previous tokens
        - Concatenate new K, V with cached versions
        - Compute attention only for new query tokens
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_cache_len: int = 2048,
        dropout: float = 0.1
    ):
        """
        Initialize the KV-Cached Multi-Head Attention layer.

        Args:
            d_model: Dimension of the model (embedding size)
            num_heads: Number of attention heads
            max_cache_len: Maximum sequence length to cache
            dropout: Dropout probability for attention weights
        """
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_cache_len = max_cache_len

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # Scaling factor for attention scores
        # Standard scaled dot-product uses 1/sqrt(d_k)
        self.scale = self.head_dim ** 0.5

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cache: Optional[Dict[str, torch.Tensor]] = None,
        use_causal_mask: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with KV-caching support.

        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]
            value: Value tensor [batch_size, seq_len, d_model]
            cache: Optional dict with cached 'key' and 'value' tensors
            use_causal_mask: Whether to apply causal masking

        Returns:
            output: Attention output [batch_size, seq_len, d_model]
            new_cache: Updated cache dictionary
        """
        batch_size, seq_len, _ = query.shape

        # Project to Q, K, V
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        # Handle cache: concatenate cached K, V with new K, V
        cache_len = 0
        if cache is not None and cache.get('key') is not None:
            cached_k = cache['key']
            cached_v = cache['value']
            cache_len = cached_k.shape[1]

            # Concatenate on seq_len dimension (dim=1) before splitting heads
            K = torch.cat([cached_k, K], dim=1)
            V = torch.cat([cached_v, V], dim=1)

        # Split into multiple heads
        Q = self._split_heads(Q)  # [batch, num_heads, seq_len_q, head_dim]
        K = self._split_heads(K)  # [batch, num_heads, total_seq_len, head_dim]
        V = self._split_heads(V)

        # Compute attention scores
        scores = self._compute_attention_scores(Q, K)

        # Apply causal mask if needed
        if use_causal_mask:
            scores = self._apply_causal_mask(scores, seq_len, cache_len)

        # Softmax on last dimension (key/sequence dimension)
        attention_weights = F.softmax(scores, dim=-1)

        # Bug #9: Dropout applied during inference
        # Should check if model is in training mode
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        output = torch.matmul(attention_weights, V)

        # Merge heads back
        output = self._merge_heads(output)

        # Final output projection
        output = self.out_proj(output)

        # Store the full concatenated K, V in merged format for next iteration
        new_cache = {
            'key': self._merge_heads(K),
            'value': self._merge_heads(V)
        }

        # Cache size validation check
        # Cache is in [batch, seq_len, d_model] format, so check shape[1]
        if new_cache['key'] is not None and new_cache['key'].shape[1] > self.max_cache_len:
            raise ValueError(f"Cache exceeded maximum length")

        return output, new_cache

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split embedding dimension into (num_heads, head_dim).

        Reshapes from [batch, seq_len, d_model]
                   to [batch, num_heads, seq_len, head_dim]

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Reshaped tensor [batch, num_heads, seq_len, head_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Reshape to (batch, seq_len, num_heads, head_dim) then permute
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Permute to (batch, num_heads, seq_len, head_dim)
        return x.permute(0, 2, 1, 3)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Merge head dimension back to embedding dimension.

        Reshapes from [batch, num_heads, seq_len, head_dim]
                   to [batch, seq_len, d_model]

        Args:
            x: Input tensor [batch, num_heads, seq_len, head_dim]

        Returns:
            Merged tensor [batch, seq_len, d_model]
        """
        batch_size, num_heads, seq_len, head_dim = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    def _compute_attention_scores(
        self,
        Q: torch.Tensor,
        K: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute scaled dot-product attention scores.

        Standard formula: scores = (Q @ K^T) / sqrt(d_k)

        Args:
            Q: Query [batch, num_heads, seq_len_q, head_dim]
            K: Key [batch, num_heads, seq_len_k, head_dim]

        Returns:
            Attention scores [batch, num_heads, seq_len_q, seq_len_k]
        """
        # Q @ K^T - transpose K on last two dimensions
        scores = torch.matmul(Q, K.transpose(-2, -1))

        # Bug #12: DECOY - Misleading comment
        # Comment references standard scaling, but self.scale is already wrong (Bug #1)
        # TODO: Optimize this division operation for faster inference
        scores = scores / self.scale

        return scores

    def _apply_causal_mask(
        self,
        scores: torch.Tensor,
        seq_len: int,
        cache_len: int
    ) -> torch.Tensor:
        """
        Apply causal mask to prevent attending to future positions.

        For autoregressive generation, position i can only attend to positions <= i.
        When using cache, current tokens can attend to all cached tokens.

        Args:
            scores: Attention scores [batch, num_heads, seq_len_q, total_seq_len]
            seq_len: Current query sequence length
            cache_len: Number of cached tokens

        Returns:
            Masked attention scores
        """
        # Bug #13: DECOY - Misleading variable name
        # Variable called "batch_seq_len" but actually represents total sequence length
        batch_seq_len = cache_len + seq_len  # Name is misleading but logic correct

        # Position offset for cache
        offset = cache_len

        # Create lower triangular mask: position i attends to j where j <= i
        # Bug #6: Causal mask boundary condition
        # Uses wrong comparison in mask generation
        mask = torch.ones(seq_len, batch_seq_len, device=scores.device)
        for i in range(seq_len):
            for j in range(batch_seq_len):
                if j <= i + cache_len:  # Wrong! Causes off-by-one with offset bug
                    mask[i, j] = 1
                else:
                    mask[i, j] = 0

        # Convert to boolean for proper masked_fill operation
        mask = mask.bool()

        # Apply mask: set future positions to -inf so softmax makes them ~0
        scores = scores.masked_fill(mask == 0, float('-inf'))

        return scores

    def reset_cache(self) -> Dict[str, Optional[torch.Tensor]]:
        """
        Return an empty cache dictionary.

        Returns:
            Empty cache with None values
        """
        return {'key': None, 'value': None}

    # Bug #16: DECOY - Unused parameter trap
    # Parameter 'use_cache' looks like it should be used but is intentionally unused
    # It's kept for API compatibility with a planned future version
    def get_cache_info(
        self,
        cache: Optional[Dict[str, torch.Tensor]],
        use_cache: bool = True  # DECOY: Looks unused but is intentional
    ) -> Dict[str, any]:
        """
        Get information about current cache state.

        Args:
            cache: Current cache dictionary
            use_cache: Whether caching is enabled (unused, for future compatibility)

        Returns:
            Dictionary with cache statistics
        """
        if cache is None or cache.get('key') is None:
            return {
                'cache_length': 0,
                'cache_size_mb': 0.0,
                'is_full': False
            }

        key_cache = cache['key']

        # Get cache dimensions
        # Cache is in merged format: [batch, seq_len, d_model]
        cache_seq_len = key_cache.shape[1]

        # Calculate memory usage (approximate)
        cache_size_bytes = key_cache.numel() * key_cache.element_size()
        cache_size_mb = (cache_size_bytes * 2) / (1024 * 1024)  # K and V caches

        return {
            'cache_length': cache_seq_len,
            'cache_size_mb': round(cache_size_mb, 2),
            'is_full': cache_seq_len >= self.max_cache_len
        }


# Bug #14: DECOY - This helper function looks inefficient
# It appears like it should be vectorized, but the loop is actually necessary
# for maintaining numerical precision in certain edge cases of this implementation
def compute_position_ids(seq_len: int, cache_len: int) -> torch.Tensor:
    """
    Compute position IDs for positional encoding with cache.

    Args:
        seq_len: Current sequence length
        cache_len: Cached sequence length

    Returns:
        Position IDs tensor
    """
    # This loop looks inefficient but is intentional for this implementation
    position_ids = []
    for i in range(seq_len):
        position_ids.append(cache_len + i)
    return torch.tensor(position_ids, dtype=torch.long)


# Bug #15: DECOY - TODO comment trap
# Suggests removing this check, but it's actually critical for edge cases
def validate_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor
) -> None:
    """
    Validate input tensor shapes and types.

    Args:
        query, key, value: Input tensors to validate

    Raises:
        ValueError: If inputs are invalid
    """
    # TODO: This validation seems redundant - PyTorch will catch dimension mismatches anyway
    # Consider removing for better performance
    if query.dim() != 3:
        raise ValueError(f"Query must be 3D tensor, got {query.dim()}D")
    if key.dim() != 3:
        raise ValueError(f"Key must be 3D tensor, got {key.dim()}D")
    if value.dim() != 3:
        raise ValueError(f"Value must be 3D tensor, got {value.dim()}D")

    if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0]:
        raise ValueError("Batch sizes must match")

    if key.shape[1] != value.shape[1]:
        raise ValueError("Key and value sequence lengths must match")


def create_sample_input(
    batch_size: int,
    seq_len: int,
    d_model: int,
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create sample input tensors for testing.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        d_model: Model dimension
        seed: Random seed for reproducibility

    Returns:
        Tuple of (query, key, value) tensors
    """
    if seed is not None:
        torch.manual_seed(seed)

    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)

    return query, key, value


if __name__ == "__main__":
    """
    Quick test to verify basic functionality.
    This will likely produce incorrect results due to bugs!
    """
    print("=" * 60)
    print("KV-Cached Multi-Head Attention - Quick Test")
    print("=" * 60)

    # Configuration
    d_model = 64
    num_heads = 4
    batch_size = 2
    seq_len = 8

    # Initialize model
    model = KVCachedMultiHeadAttention(
        d_model=d_model,
        num_heads=num_heads,
        max_cache_len=128,
        dropout=0.1
    )
    model.eval()

    # Create sample inputs
    q, k, v = create_sample_input(batch_size, seq_len, d_model, seed=42)

    print(f"\nInput shapes:")
    print(f"  Query: {q.shape}")
    print(f"  Key: {k.shape}")
    print(f"  Value: {v.shape}")

    # First forward pass (no cache)
    print("\n--- First forward pass (no cache) ---")
    try:
        output1, cache1 = model(q, k, v, cache=None, use_causal_mask=True)
        print(f"✓ Output shape: {output1.shape}")
        print(f"✓ Cache key shape: {cache1['key'].shape if cache1['key'] is not None else 'None'}")
        print(f"  Cache info: {model.get_cache_info(cache1)}")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Second forward pass (with cache) - simulating next token generation
    print("\n--- Second forward pass (with cache) ---")
    try:
        q2, k2, v2 = create_sample_input(batch_size, 1, d_model, seed=43)
        output2, cache2 = model(q2, k2, v2, cache=cache1, use_causal_mask=True)
        print(f"✓ Output shape: {output2.shape}")
        print(f"✓ Cache key shape: {cache2['key'].shape if cache2['key'] is not None else 'None'}")
        print(f"  Cache info: {model.get_cache_info(cache2)}")
    except Exception as e:
        print(f"✗ Error: {e}")

    print("\n" + "=" * 60)
    print("Note: This code contains bugs! Use validator.py to test.")
    print("=" * 60)
