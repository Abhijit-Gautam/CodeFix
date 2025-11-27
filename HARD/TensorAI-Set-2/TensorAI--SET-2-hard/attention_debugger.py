"""
AI Attention Debugger - BONUS Challenge (+10%)

Implement an AI agent that can automatically detect and suggest fixes
for bugs in attention mechanism implementations.

This is an advanced challenge for those who want to demonstrate
deep understanding of transformers and debugging techniques.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
import ast
import inspect
import re


class AttentionBugDetector:
    """
    An AI agent that analyzes attention mechanism code and detects bugs.

    Implements methods to automatically detect common bugs in
    transformer attention implementations.
    """

    def __init__(self):
        """Initialize the bug detector with known bug patterns."""
        self.bug_patterns = [
            'scaling_factor',
            'softmax_dimension',
            'cache_concatenation',
            'transpose_dimensions',
            'mask_dtype',
            'cache_update',
            'head_reshape'
        ]
        self.detected_bugs = []

    def analyze_code(self, module) -> List[Dict[str, any]]:
        """
        Analyze a module and detect potential bugs.

        Args:
            module: Python module containing KVCachedMultiHeadAttention class

        Returns:
            List of detected bugs with metadata
        """
        self.detected_bugs = []
        
        # Get the source code
        try:
            source = inspect.getsource(module)
        except:
            return []
        
        # Run all bug detection checks
        bug = self.check_scaling_factor_from_source(source)
        if bug:
            self.detected_bugs.append(bug)
        
        bug = self.check_softmax_dimension_from_source(source)
        if bug:
            self.detected_bugs.append(bug)
        
        bug = self.check_cache_concatenation_from_source(source)
        if bug:
            self.detected_bugs.append(bug)
        
        bug = self.check_transpose_dimensions_from_source(source)
        if bug:
            self.detected_bugs.append(bug)
        
        bug = self.check_mask_dtype_from_source(source)
        if bug:
            self.detected_bugs.append(bug)
        
        bug = self.check_split_heads_from_source(source)
        if bug:
            self.detected_bugs.append(bug)
        
        bugs = self.check_cache_update_from_source(source)
        self.detected_bugs.extend(bugs)
        
        return self.detected_bugs
    
    def check_scaling_factor_from_source(self, source: str) -> Optional[Dict]:
        """Check for wrong scaling factor in source code."""
        # Look for self.scale = self.head_dim without sqrt (and not in a comment)
        lines = source.split('\n')
        for i, line in enumerate(lines, 1):
            # Skip comments
            stripped = line.strip()
            if stripped.startswith('#'):
                continue
            
            # Check for wrong pattern: self.scale = self.head_dim without ** 0.5
            if 'self.scale' in line and '=' in line and 'self.head_dim' in line:
                # Check if it has the sqrt operation
                if '** 0.5' not in line and 'sqrt' not in line and '**0.5' not in line:
                    return {
                        'type': 'scaling_factor',
                        'severity': 'critical',
                        'location': f'line {i}',
                        'description': 'Wrong scaling factor: using head_dim instead of sqrt(head_dim)',
                        'suggestion': 'Change to: self.scale = self.head_dim ** 0.5'
                    }
        return None
    
    def check_softmax_dimension_from_source(self, source: str) -> Optional[Dict]:
        """Check for wrong softmax dimension."""
        # Look for softmax with wrong dimension
        patterns = [
            (r'F\.softmax\([^,]+,\s*dim\s*=\s*(-2|2|1|0)\)', 'F.softmax'),
            (r'\.softmax\(\s*dim\s*=\s*(-2|2|1|0)\)', '.softmax')
        ]
        
        for pattern, func_name in patterns:
            match = re.search(pattern, source)
            if match:
                line_num = source[:match.start()].count('\n') + 1
                wrong_dim = match.group(1)
                return {
                    'type': 'softmax_dimension',
                    'severity': 'critical',
                    'location': f'line {line_num}',
                    'description': f'Softmax applied on wrong dimension (dim={wrong_dim})',
                    'suggestion': f'Change to: {func_name}(..., dim=-1) for attention over keys'
                }
        return None
    
    def check_cache_concatenation_from_source(self, source: str) -> Optional[Dict]:
        """Check for wrong cache concatenation dimension."""
        # Look for torch.cat with dim=2 for cache (should be dim=1 for merged format)
        pattern = r'torch\.cat\(\[cached_[kv],\s*[KV]\],\s*dim\s*=\s*2\)'
        match = re.search(pattern, source)
        
        if match:
            line_num = source[:match.start()].count('\n') + 1
            return {
                'type': 'cache_concatenation',
                'severity': 'critical',
                'location': f'line {line_num}',
                'description': 'Cache concatenation on wrong dimension (dim=2)',
                'suggestion': 'Use dim=1 for sequence dimension in merged format [batch, seq, d_model]'
            }
        return None
    
    def check_transpose_dimensions_from_source(self, source: str) -> Optional[Dict]:
        """Check for wrong transpose dimensions in attention score computation."""
        # Look for K.transpose(1, 2) instead of K.transpose(-2, -1)
        pattern = r'K\.transpose\(\s*1\s*,\s*2\s*\)'
        match = re.search(pattern, source)
        
        if match:
            line_num = source[:match.start()].count('\n') + 1
            return {
                'type': 'transpose_dimensions',
                'severity': 'critical',
                'location': f'line {line_num}',
                'description': 'Wrong transpose dimensions for K in Q @ K^T',
                'suggestion': 'Use K.transpose(-2, -1) to transpose last two dimensions'
            }
        return None
    
    def check_mask_dtype_from_source(self, source: str) -> Optional[Dict]:
        """Check for wrong mask dtype."""
        # Look for mask.int() instead of mask.bool()
        pattern = r'mask\s*=\s*mask\.int\(\)'
        match = re.search(pattern, source)
        
        if match:
            line_num = source[:match.start()].count('\n') + 1
            return {
                'type': 'mask_dtype',
                'severity': 'high',
                'location': f'line {line_num}',
                'description': 'Mask converted to int instead of bool',
                'suggestion': 'Use mask.bool() for masked_fill operation'
            }
        return None
    
    def check_split_heads_from_source(self, source: str) -> Optional[Dict]:
        """Check for wrong reshape order in split_heads."""
        # Look for wrong view order: (batch, num_heads, seq_len, head_dim) before permute
        pattern = r'\.view\([^)]*,\s*self\.num_heads\s*,\s*seq_len'
        match = re.search(pattern, source)
        
        if match:
            line_num = source[:match.start()].count('\n') + 1
            return {
                'type': 'split_heads_reshape',
                'severity': 'critical',
                'location': f'line {line_num}',
                'description': 'Wrong reshape order in split_heads: num_heads before seq_len',
                'suggestion': 'Reshape to (batch, seq_len, num_heads, head_dim) then permute'
            }
        return None
    
    def check_cache_update_from_source(self, source: str) -> List[Dict]:
        """Check for wrong cache update strategy."""
        bugs = []
        
        # Look for cache slicing that loses previous tokens
        pattern = r"'key'\s*:\s*K\[\s*:\s*,\s*:\s*,\s*-seq_len\s*:"
        match = re.search(pattern, source)
        
        if match:
            line_num = source[:match.start()].count('\n') + 1
            bugs.append({
                'type': 'cache_update',
                'severity': 'critical',
                'location': f'line {line_num}',
                'description': 'Cache update discards previous cached tokens',
                'suggestion': 'Store full concatenated K, V using _merge_heads()'
            })
        
        # Check for wrong cache shape index
        pattern = r"new_cache\['key'\]\.shape\[2\]\s*>"
        match = re.search(pattern, source)
        
        if match:
            line_num = source[:match.start()].count('\n') + 1
            bugs.append({
                'type': 'cache_shape_index',
                'severity': 'high',
                'location': f'line {line_num}',
                'description': 'Cache length check uses wrong shape index (2 instead of 1)',
                'suggestion': 'Use shape[1] for merged format [batch, seq_len, d_model]'
            })
        
        return bugs

    def check_scaling_factor(self, model_class) -> Optional[Dict]:
        """
        Check if attention scaling factor is correct.

        Correct: scores / sqrt(d_k)
        Wrong: scores / d_k

        Returns:
            Bug dict if found, None otherwise
        """
        try:
            # Check if model has scale attribute
            if hasattr(model_class, 'scale') and hasattr(model_class, 'head_dim'):
                # Create instance to check
                instance = model_class(d_model=64, num_heads=4)
                expected_scale = instance.head_dim ** 0.5
                if abs(instance.scale - expected_scale) > 1e-6:
                    return {
                        'type': 'scaling_factor',
                        'severity': 'critical',
                        'location': '__init__',
                        'description': f'Scale is {instance.scale}, expected {expected_scale}',
                        'suggestion': 'Use self.scale = self.head_dim ** 0.5'
                    }
        except Exception:
            pass
        return None

    def check_softmax_dimension(self, model_class) -> Optional[Dict]:
        """
        Check if softmax is applied on correct dimension.

        Correct: F.softmax(scores, dim=-1)  # Last dimension
        Wrong: F.softmax(scores, dim=-2)    # Wrong dimension

        Returns:
            Bug dict if found, None otherwise
        """
        try:
            source = inspect.getsource(model_class)
            return self.check_softmax_dimension_from_source(source)
        except Exception:
            pass
        return None

    def check_cache_concatenation(self, model_class) -> Optional[Dict]:
        """
        Check if cache concatenation uses correct dimension.

        After projection: [batch, seq_len, d_model]
        Correct: torch.cat([cached_k, K], dim=1)  # Sequence dimension
        Wrong: torch.cat([cached_k, K], dim=2)    # Model dimension

        Returns:
            Bug dict if found, None otherwise
        """
        try:
            source = inspect.getsource(model_class)
            return self.check_cache_concatenation_from_source(source)
        except Exception:
            pass
        return None

    def check_dropout_during_inference(self, model_class) -> Optional[Dict]:
        """
        Check if dropout is incorrectly applied during inference.

        Correct: if self.training: dropout(x)
        Wrong: Always applying dropout

        Returns:
            Bug dict if found, None otherwise
        """
        try:
            source = inspect.getsource(model_class)
            
            # Check if dropout is used without training check
            has_dropout = 'self.dropout(' in source
            has_training_check = 'if self.training' in source or 'self.training:' in source
            
            if has_dropout and not has_training_check:
                return {
                    'type': 'dropout_inference',
                    'severity': 'medium',
                    'location': 'forward method',
                    'description': 'Dropout applied without checking training mode',
                    'suggestion': 'Wrap dropout in: if self.training: x = self.dropout(x)'
                }
        except Exception:
            pass
        return None

    def check_tensor_dimensions(self, model_class) -> List[Dict]:
        """
        Check for dimension errors in tensor operations.

        Common issues:
        - Wrong reshape order
        - Incorrect transpose dimensions
        - Dimension mismatch in matmul

        Returns:
            List of dimension-related bugs
        """
        bugs = []
        try:
            source = inspect.getsource(model_class)
            
            # Check transpose
            bug = self.check_transpose_dimensions_from_source(source)
            if bug:
                bugs.append(bug)
            
            # Check split_heads reshape
            bug = self.check_split_heads_from_source(source)
            if bug:
                bugs.append(bug)
                
        except Exception:
            pass
        return bugs

    def suggest_fix(self, bug: Dict[str, any]) -> str:
        """
        Generate a detailed fix suggestion for a detected bug.

        Args:
            bug: Bug dictionary from detection

        Returns:
            Detailed fix suggestion with code examples
        """
        fix_templates = {
            'scaling_factor': """
## Fix for Scaling Factor Bug

**Problem:** The attention scores are divided by `head_dim` instead of `sqrt(head_dim)`.

**Solution:** Change the scale initialization:

```python
# Before (wrong):
self.scale = self.head_dim

# After (correct):
self.scale = self.head_dim ** 0.5
```

**Explanation:** The scaled dot-product attention formula is:
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

Dividing by sqrt(d_k) prevents the dot products from becoming too large,
which would push the softmax into regions with vanishing gradients.
""",
            'softmax_dimension': """
## Fix for Softmax Dimension Bug

**Problem:** Softmax is applied on the wrong dimension.

**Solution:** Use dim=-1 for attention:

```python
# Before (wrong):
attention_weights = F.softmax(scores, dim=-2)  # or dim=2

# After (correct):
attention_weights = F.softmax(scores, dim=-1)
```

**Explanation:** Softmax should normalize across the key dimension so that
attention weights for each query sum to 1.0 across all keys.
""",
            'cache_concatenation': """
## Fix for Cache Concatenation Bug

**Problem:** Cache is concatenated on the wrong dimension.

**Solution:** Concatenate on sequence dimension (dim=1):

```python
# Before (wrong):
K = torch.cat([cached_k, K], dim=2)

# After (correct):
K = torch.cat([cached_k, K], dim=1)
```

**Explanation:** Cache tensors are in merged format [batch, seq_len, d_model].
Concatenation must happen on the sequence dimension (dim=1).
""",
            'transpose_dimensions': """
## Fix for Transpose Dimensions Bug

**Problem:** K is transposed on wrong dimensions for Q @ K^T.

**Solution:** Transpose the last two dimensions:

```python
# Before (wrong):
scores = torch.matmul(Q, K.transpose(1, 2))

# After (correct):
scores = torch.matmul(Q, K.transpose(-2, -1))
```

**Explanation:** Q @ K^T requires transposing K's last two dimensions
(seq_len and head_dim). Using (1, 2) transposes num_heads and seq_len.
""",
            'mask_dtype': """
## Fix for Mask Dtype Bug

**Problem:** Mask is converted to int instead of bool.

**Solution:** Use bool for masked_fill:

```python
# Before (wrong):
mask = mask.int()

# After (correct):
mask = mask.bool()
```

**Explanation:** PyTorch's masked_fill expects a boolean mask tensor.
""",
            'split_heads_reshape': """
## Fix for Split Heads Reshape Bug

**Problem:** Wrong reshape order in _split_heads.

**Solution:** Reshape to (batch, seq_len, num_heads, head_dim) first:

```python
# Before (wrong):
x = x.view(batch_size, self.num_heads, seq_len, self.head_dim)
return x.permute(0, 2, 1, 3)

# After (correct):
x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
return x.permute(0, 2, 1, 3)
```

**Explanation:** The input is [batch, seq_len, d_model]. We first split d_model
into (num_heads, head_dim), keeping seq_len in position 1, then permute.
""",
            'cache_update': """
## Fix for Cache Update Bug

**Problem:** Cache update discards previous cached tokens.

**Solution:** Store the full concatenated tensors:

```python
# Before (wrong):
new_cache = {
    'key': K[:, :, -seq_len:, :],
    'value': V[:, :, -seq_len:, :]
}

# After (correct):
new_cache = {
    'key': self._merge_heads(K),
    'value': self._merge_heads(V)
}
```

**Explanation:** The cache must store ALL tokens (cached + new) for the
next iteration. The original code only kept the new tokens.
"""
        }
        
        bug_type = bug.get('type', '')
        if bug_type in fix_templates:
            return fix_templates[bug_type]
        
        return f"""
## Fix for {bug_type}

**Location:** {bug.get('location', 'unknown')}
**Problem:** {bug.get('description', 'Unknown issue')}
**Solution:** {bug.get('suggestion', 'Review the implementation')}
"""

    def run_analysis(self, module) -> None:
        """
        Run complete analysis and print report.

        Args:
            module: Module to analyze
        """
        print("=" * 70)
        print("AI Attention Debugger - Analysis Report")
        print("=" * 70)

        bugs = self.analyze_code(module)

        if not bugs:
            print("\n✓ No bugs detected!")
            return

        print(f"\n✗ Found {len(bugs)} potential bug(s):\n")

        for i, bug in enumerate(bugs, 1):
            severity_icon = {
                'critical': '🔴',
                'high': '🟠',
                'medium': '🟡',
                'low': '🟢'
            }.get(bug['severity'], '⚪')

            print(f"{i}. {severity_icon} [{bug['severity'].upper()}] {bug['type']}")
            print(f"   Location: {bug.get('location', 'unknown')}")
            print(f"   Issue: {bug['description']}")
            print(f"   Fix: {bug['suggestion']}\n")

        print("=" * 70)


class AttentionValidator:
    """
    Validates attention mechanism correctness through runtime checks.

    Your task: Implement validators that check attention computation
    properties during execution.
    """

    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize validator.

        Args:
            tolerance: Numerical tolerance for comparisons
        """
        self.tolerance = tolerance

    def validate_attention_weights(self, attention_weights: torch.Tensor) -> Tuple[bool, str]:
        """
        Validate that attention weights sum to 1.0.

        Attention weights after softmax should sum to 1.0 along the key dimension.

        Args:
            attention_weights: [batch, num_heads, seq_len_q, seq_len_k]

        Returns:
            (is_valid, message)
        """
        # Check that weights sum to 1.0 along the key dimension (last dim)
        sums = attention_weights.sum(dim=-1)
        ones = torch.ones_like(sums)
        
        if torch.allclose(sums, ones, atol=self.tolerance):
            return True, "Attention weights correctly sum to 1.0"
        
        max_diff = (sums - ones).abs().max().item()
        return False, f"Attention weights do not sum to 1.0 (max diff: {max_diff:.6f})"

    def validate_cache_shapes(
        self,
        cache: Dict[str, torch.Tensor],
        expected_seq_len: int
    ) -> Tuple[bool, str]:
        """
        Validate cache tensor shapes are correct.

        Args:
            cache: Cache dictionary with 'key' and 'value'
            expected_seq_len: Expected sequence length

        Returns:
            (is_valid, message)
        """
        if cache is None or cache.get('key') is None:
            return True, "Cache is empty (valid for first pass)"
        
        key_cache = cache['key']
        value_cache = cache['value']
        
        # Check key cache shape
        if key_cache.dim() != 3:
            return False, f"Cache key should be 3D, got {key_cache.dim()}D"
        
        # Check sequence length
        if key_cache.shape[1] != expected_seq_len:
            return False, f"Cache seq_len is {key_cache.shape[1]}, expected {expected_seq_len}"
        
        # Check key and value have same shape
        if key_cache.shape != value_cache.shape:
            return False, f"Key cache {key_cache.shape} != Value cache {value_cache.shape}"
        
        return True, f"Cache shapes valid: {key_cache.shape}"

    def validate_output_shape(
        self,
        output: torch.Tensor,
        query: torch.Tensor
    ) -> Tuple[bool, str]:
        """
        Validate output shape matches query shape.

        Output should be [batch, seq_len_q, d_model], same as query.

        Args:
            output: Model output
            query: Query input

        Returns:
            (is_valid, message)
        """
        if output.shape == query.shape:
            return True, f"Output shape matches query: {output.shape}"
        
        return False, f"Output shape {output.shape} != Query shape {query.shape}"

    def validate_causal_mask(
        self,
        attention_weights: torch.Tensor,
        seq_len: int
    ) -> Tuple[bool, str]:
        """
        Validate that causal mask is correctly applied.

        For causal attention, position i should have ~0 weight for positions > i.

        Args:
            attention_weights: [batch, num_heads, seq_len, seq_len]
            seq_len: Sequence length

        Returns:
            (is_valid, message)
        """
        if attention_weights.shape[-1] != attention_weights.shape[-2]:
            return True, "Non-square attention (likely with cache) - skipping causal check"
        
        # Check upper triangular part (excluding diagonal) is near zero
        batch, heads, q_len, k_len = attention_weights.shape
        
        # Create upper triangular mask (positions that should be ~0)
        future_mask = torch.triu(torch.ones(q_len, k_len), diagonal=1).bool()
        
        # Get the values that should be near zero
        future_weights = attention_weights[:, :, future_mask].abs()
        
        if future_weights.max() < self.tolerance:
            return True, "Causal mask correctly applied (future positions are ~0)"
        
        max_future = future_weights.max().item()
        return False, f"Causal mask not applied correctly (max future weight: {max_future:.6f})"
    
    def validate_no_nan_inf(self, tensor: torch.Tensor, name: str = "tensor") -> Tuple[bool, str]:
        """Check tensor contains no NaN or Inf values."""
        if torch.isnan(tensor).any():
            return False, f"{name} contains NaN values"
        if torch.isinf(tensor).any():
            return False, f"{name} contains Inf values"
        return True, f"{name} is numerically stable"
    
    def run_full_validation(
        self,
        model: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cache: Optional[Dict] = None,
        use_causal_mask: bool = True
    ) -> List[Tuple[bool, str]]:
        """Run all validation checks on a forward pass."""
        results = []
        
        model.eval()
        with torch.no_grad():
            output, new_cache = model(query, key, value, cache=cache, use_causal_mask=use_causal_mask)
        
        # Validate output
        results.append(self.validate_output_shape(output, query))
        results.append(self.validate_no_nan_inf(output, "Output"))
        
        # Validate cache
        expected_cache_len = query.shape[1] + (cache['key'].shape[1] if cache and cache.get('key') is not None else 0)
        results.append(self.validate_cache_shapes(new_cache, expected_cache_len))
        
        return results


def main():
    """
    Main entry point for the AI debugger.

    Usage:
        python attention_debugger.py
    """
    print("=" * 70)
    print("AI Attention Debugger - Automated Bug Detection")
    print("=" * 70)
    
    # Import the module to analyze
    try:
        import kv_attention
        print("\n✓ Successfully loaded kv_attention module")
    except ImportError as e:
        print(f"\n✗ Failed to import kv_attention: {e}")
        return
    
    # Initialize detector
    detector = AttentionBugDetector()
    
    # Analyze the code
    print("\n" + "-" * 70)
    print("STATIC CODE ANALYSIS")
    print("-" * 70)
    
    bugs = detector.analyze_code(kv_attention)
    
    if not bugs:
        print("\n✓ No bugs detected in source code!")
    else:
        print(f"\n✗ Found {len(bugs)} potential bug(s):\n")
        for i, bug in enumerate(bugs, 1):
            severity_icon = {
                'critical': '🔴',
                'high': '🟠',
                'medium': '🟡',
                'low': '🟢'
            }.get(bug['severity'], '⚪')
            
            print(f"{i}. {severity_icon} [{bug['severity'].upper()}] {bug['type']}")
            print(f"   Location: {bug.get('location', 'unknown')}")
            print(f"   Issue: {bug['description']}")
            print(f"   Fix: {bug['suggestion']}\n")
    
    # Runtime validation
    print("-" * 70)
    print("RUNTIME VALIDATION")
    print("-" * 70)
    
    try:
        import torch
        
        # Create model and test inputs
        model = kv_attention.KVCachedMultiHeadAttention(
            d_model=64, num_heads=4, max_cache_len=128, dropout=0.0
        )
        model.eval()
        
        torch.manual_seed(42)
        q = torch.randn(1, 8, 64)
        k = torch.randn(1, 8, 64)
        v = torch.randn(1, 8, 64)
        
        # Run validation
        validator = AttentionValidator()
        results = validator.run_full_validation(model, q, k, v)
        
        print()
        all_passed = True
        for is_valid, message in results:
            status = "✓" if is_valid else "✗"
            print(f"  {status} {message}")
            if not is_valid:
                all_passed = False
        
        # Test with cache
        print("\n  Testing with cache...")
        output, cache = model(q, k, v)
        q2 = torch.randn(1, 1, 64)
        k2 = torch.randn(1, 1, 64)
        v2 = torch.randn(1, 1, 64)
        
        results2 = validator.run_full_validation(model, q2, k2, v2, cache=cache)
        for is_valid, message in results2:
            status = "✓" if is_valid else "✗"
            print(f"  {status} {message}")
            if not is_valid:
                all_passed = False
        
        if all_passed:
            print("\n✓ All runtime validations passed!")
        else:
            print("\n✗ Some runtime validations failed!")
            
    except Exception as e:
        print(f"\n✗ Runtime validation error: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if not bugs:
        print("\n🎉 The kv_attention.py implementation appears to be correct!")
        print("   All bug patterns checked - no issues found.")
    else:
        print(f"\n⚠️  Found {len(bugs)} bug(s) that should be fixed.")
        print("\nDetailed fix suggestions:")
        for bug in bugs:
            print(detector.suggest_fix(bug))
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
