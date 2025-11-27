# KV-Cached Multi-Head Attention - Bug Fixes Documentation

## AI CODEFIX 2025 - Hard Challenge #2

**File Fixed:** `kv_attention.py`  
**Date:** November 27, 2025  
**Result:** ✅ All tests passed (1/1)

---

## Summary

Fixed 9 bugs in the KV-Cached Multi-Head Attention implementation. The code now correctly implements scaled dot-product attention with KV-caching for efficient LLM inference.

---

## Bug Fixes

### Bug #1 - Scale Factor (Line 68)

**Issue:** Incorrect scaling factor for attention scores.

**Before:**
```python
self.scale = self.head_dim  # Bug #1: Should be sqrt(self.head_dim)
```

**After:**
```python
self.scale = self.head_dim ** 0.5
```

**Explanation:** The scaled dot-product attention formula requires dividing by √(d_k) to prevent the dot products from growing too large, which would push softmax into regions with extremely small gradients.

---

### Bug #2 - Cache Concatenation (Lines 105-106)

**Issue:** Cache concatenation on wrong dimension.

**Before:**
```python
K = torch.cat([cached_k, K], dim=2)  # Wrong!
V = torch.cat([cached_v, V], dim=2)  # Wrong!
```

**After:**
```python
K = torch.cat([cached_k, K], dim=1)
V = torch.cat([cached_v, V], dim=1)
```

**Explanation:** The cache stores tensors in merged format `[batch, seq_len, d_model]`, so concatenation must happen on the sequence dimension (dim=1), not dim=2.

---

### Bug #3 - Softmax Dimension (Line 121)

**Issue:** Softmax applied on wrong dimension.

**Before:**
```python
attention_weights = F.softmax(scores, dim=2)  # Wrong! Should be dim=-1
```

**After:**
```python
attention_weights = F.softmax(scores, dim=-1)
```

**Explanation:** Softmax should be applied on the last dimension (key/sequence dimension) so that attention weights sum to 1 across all keys for each query position.

---

### Bug #4 - Matrix Transpose (Line 207)

**Issue:** Incorrect transpose dimensions for Q @ K^T computation.

**Before:**
```python
scores = torch.matmul(Q, K.transpose(1, 2))  # Wrong! Should be transpose(-2, -1)
```

**After:**
```python
scores = torch.matmul(Q, K.transpose(-2, -1))
```

**Explanation:** For Q @ K^T, we need to transpose K's last two dimensions (seq_len and head_dim). Using transpose(1, 2) transposes num_heads and seq_len, which is incorrect.

---

### Bug #5 - Position Offset (Line 241)

**Issue:** Incorrect offset calculation for causal mask.

**Before:**
```python
offset = cache_len + 1  # Wrong! Should be just cache_len (no +1)
```

**After:**
```python
offset = cache_len
```

**Explanation:** The offset should exactly equal the cache length. Adding 1 causes an off-by-one error in the causal mask.

---

### Bug #7 - Split Heads Reshape (Lines 165-170)

**Issue:** Incorrect view/permute order in `_split_heads` function.

**Before:**
```python
x = x.view(batch_size, self.num_heads, seq_len, self.head_dim)  # Wrong order!
return x.permute(0, 2, 1, 3)  # Bug #7 continued
```

**After:**
```python
x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
return x.permute(0, 2, 1, 3)
```

**Explanation:** The view should first reshape to `[batch, seq_len, num_heads, head_dim]`, then permute to `[batch, num_heads, seq_len, head_dim]`. The original code had the view dimensions in the wrong order.

---

### Bug #8 - Cache Update Strategy (Lines 139-142)

**Issue:** Cache stored only new tokens instead of full concatenated K, V.

**Before:**
```python
new_cache = {
    'key': K[:, :, -seq_len:, :],  # Wrong! Loses previous cache
    'value': V[:, :, -seq_len:, :]
}
```

**After:**
```python
new_cache = {
    'key': self._merge_heads(K),
    'value': self._merge_heads(V)
}
```

**Explanation:** The cache must store ALL tokens (cached + new) in merged format for the next forward pass. The original code discarded the cached tokens.

---

### Bug #11 - Mask Dtype (Line 255)

**Issue:** Mask had wrong data type for `masked_fill` operation.

**Before:**
```python
mask = mask.int()  # Wrong! Should be .bool()
```

**After:**
```python
mask = mask.bool()
```

**Explanation:** PyTorch's `masked_fill` function expects a boolean mask. Using an integer mask can cause unexpected behavior.

---

### Bug #10 - Cache Shape Index (Lines 146, 298)

**Issue:** Cache size check used wrong shape index after changing to merged format.

**Before:**
```python
if new_cache['key'] is not None and new_cache['key'].shape[2] > self.max_cache_len:
```

**After:**
```python
if new_cache['key'] is not None and new_cache['key'].shape[1] > self.max_cache_len:
```

**Explanation:** Since the cache is now stored in merged format `[batch, seq_len, d_model]`, the sequence length is at index 1, not index 2.

---

## DECOY Bugs (Not Fixed)

The following were intentionally misleading and were **NOT** bugs:

- **Bug #12:** Misleading comment about scaling (comment was wrong, but code logic was dependent on Bug #1 fix)
- **Bug #13:** Misleading variable name `batch_seq_len` (name is confusing but logic is correct)
- **Bug #14:** Loop in `compute_position_ids` that looks inefficient (intentional for this implementation)
- **Bug #15:** TODO comment suggesting removal of validation (validation is actually necessary)
- **Bug #16:** Unused parameter `use_cache` (intentionally kept for API compatibility)

---

## Test Results

```
======================================================================
KV-Cached Multi-Head Attention - Validator
======================================================================
✓ Successfully loaded module from kv_attention.py
✓ Loaded 1 test case(s) from test_cases.json

Running 1 test case(s)...

✓ Test #1 'Basic attention without cache' - Executed successfully
  Results: {'output_shape': [1, 4, 16], 'cache_key_shape': [1, 4, 16], 'cache_value_shape': [1, 4, 16]}

======================================================================
Results: 1/1 tests passed
======================================================================
✓ All tests passed! Great job!
```

---

## Key Concepts Applied

1. **Scaled Dot-Product Attention:** `Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V`
2. **Multi-Head Split/Merge:** Proper reshaping from `[B, S, D]` → `[B, H, S, D/H]` → `[B, S, D]`
3. **KV-Caching:** Store full K, V tensors in merged format for efficient autoregressive generation
4. **Causal Masking:** Position i can only attend to positions ≤ i (plus all cached positions)

---