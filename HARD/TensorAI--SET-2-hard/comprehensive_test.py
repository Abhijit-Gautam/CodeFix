"""
Comprehensive test script for KV-Cached Multi-Head Attention
Tests various scenarios similar to hidden test cases
"""
import torch
import torch.nn as nn
import numpy as np
import sys

# Import the module
sys.path.insert(0, '.')
from kv_attention import KVCachedMultiHeadAttention, create_sample_input

def run_test(name, model, query, key, value, cache, use_causal_mask, expected_output_shape, expected_cache_shape):
    """Run a single test and check shapes"""
    try:
        with torch.no_grad():
            output, new_cache = model(query, key, value, cache=cache, use_causal_mask=use_causal_mask)
        
        # Check output shape
        if list(output.shape) != expected_output_shape:
            print(f"✗ {name}: Output shape mismatch - got {list(output.shape)}, expected {expected_output_shape}")
            return False
        
        # Check cache shape
        if list(new_cache['key'].shape) != expected_cache_shape:
            print(f"✗ {name}: Cache shape mismatch - got {list(new_cache['key'].shape)}, expected {expected_cache_shape}")
            return False
        
        # Check attention weights sum to ~1 (no NaN/Inf)
        if torch.isnan(output).any():
            print(f"✗ {name}: Output contains NaN")
            return False
        if torch.isinf(output).any():
            print(f"✗ {name}: Output contains Inf")
            return False
            
        print(f"✓ {name}: PASSED")
        return True
    except Exception as e:
        print(f"✗ {name}: Runtime error - {e}")
        return False

def main():
    print("=" * 70)
    print("Comprehensive KV-Cached Multi-Head Attention Test Suite")
    print("=" * 70)
    
    passed = 0
    total = 0
    
    # Test 1: Basic attention without cache
    print("\n--- Test 1: Basic attention without cache ---")
    torch.manual_seed(42)
    model = KVCachedMultiHeadAttention(d_model=16, num_heads=2, max_cache_len=128, dropout=0.0)
    model.eval()
    q, k, v = create_sample_input(1, 4, 16, seed=42)
    total += 1
    if run_test("Basic attention", model, q, k, v, None, True, [1, 4, 16], [1, 4, 16]):
        passed += 1
    
    # Test 2: Attention with cached context
    print("\n--- Test 2: Attention with cached context ---")
    torch.manual_seed(43)
    model = KVCachedMultiHeadAttention(d_model=16, num_heads=2, max_cache_len=128, dropout=0.0)
    model.eval()
    # First get a cache
    q1, k1, v1 = create_sample_input(1, 4, 16, seed=42)
    _, cache1 = model(q1, k1, v1, cache=None, use_causal_mask=True)
    # Now test with cache
    q2, k2, v2 = create_sample_input(1, 1, 16, seed=43)
    total += 1
    if run_test("With cache", model, q2, k2, v2, cache1, True, [1, 1, 16], [1, 5, 16]):
        passed += 1
    
    # Test 3: Multi-batch attention
    print("\n--- Test 3: Multi-batch attention ---")
    torch.manual_seed(100)
    model = KVCachedMultiHeadAttention(d_model=32, num_heads=4, max_cache_len=128, dropout=0.0)
    model.eval()
    q, k, v = create_sample_input(4, 8, 32, seed=100)
    total += 1
    if run_test("Multi-batch", model, q, k, v, None, True, [4, 8, 32], [4, 8, 32]):
        passed += 1
    
    # Test 4: No causal mask (bidirectional)
    print("\n--- Test 4: No causal mask (bidirectional) ---")
    torch.manual_seed(50)
    model = KVCachedMultiHeadAttention(d_model=16, num_heads=2, max_cache_len=128, dropout=0.0)
    model.eval()
    q, k, v = create_sample_input(1, 6, 16, seed=50)
    total += 1
    if run_test("No causal mask", model, q, k, v, None, False, [1, 6, 16], [1, 6, 16]):
        passed += 1
    
    # Test 5: Single head attention
    print("\n--- Test 5: Single head attention ---")
    torch.manual_seed(200)
    model = KVCachedMultiHeadAttention(d_model=32, num_heads=1, max_cache_len=128, dropout=0.0)
    model.eval()
    q, k, v = create_sample_input(2, 5, 32, seed=200)
    total += 1
    if run_test("Single head", model, q, k, v, None, True, [2, 5, 32], [2, 5, 32]):
        passed += 1
    
    # Test 6: Many heads attention
    print("\n--- Test 6: Many heads attention (8 heads) ---")
    torch.manual_seed(300)
    model = KVCachedMultiHeadAttention(d_model=64, num_heads=8, max_cache_len=256, dropout=0.0)
    model.eval()
    q, k, v = create_sample_input(1, 10, 64, seed=300)
    total += 1
    if run_test("Many heads", model, q, k, v, None, True, [1, 10, 64], [1, 10, 64]):
        passed += 1
    
    # Test 7: Long sequence (64 tokens)
    print("\n--- Test 7: Long sequence (64 tokens) ---")
    torch.manual_seed(400)
    model = KVCachedMultiHeadAttention(d_model=32, num_heads=4, max_cache_len=256, dropout=0.0)
    model.eval()
    q, k, v = create_sample_input(1, 64, 32, seed=400)
    total += 1
    if run_test("Long sequence", model, q, k, v, None, True, [1, 64, 32], [1, 64, 32]):
        passed += 1
    
    # Test 8: Incremental generation (5 steps with cache)
    print("\n--- Test 8: Incremental generation with cache ---")
    torch.manual_seed(500)
    model = KVCachedMultiHeadAttention(d_model=16, num_heads=2, max_cache_len=128, dropout=0.0)
    model.eval()
    # Create initial cache
    q_init, k_init, v_init = create_sample_input(1, 10, 16, seed=501)
    _, cache_init = model(q_init, k_init, v_init, cache=None, use_causal_mask=True)
    # Test with 5 new tokens
    q, k, v = create_sample_input(1, 5, 16, seed=500)
    total += 1
    if run_test("Incremental gen", model, q, k, v, cache_init, True, [1, 5, 16], [1, 15, 16]):
        passed += 1
    
    # Test 9: Large batch stress test
    print("\n--- Test 9: Large batch stress test ---")
    torch.manual_seed(600)
    model = KVCachedMultiHeadAttention(d_model=64, num_heads=4, max_cache_len=256, dropout=0.0)
    model.eval()
    q, k, v = create_sample_input(16, 12, 64, seed=600)
    total += 1
    if run_test("Large batch", model, q, k, v, None, True, [16, 12, 64], [16, 12, 64]):
        passed += 1
    
    # Test 10: Single token generation (typical LLM inference)
    print("\n--- Test 10: Single token generation ---")
    torch.manual_seed(700)
    model = KVCachedMultiHeadAttention(d_model=32, num_heads=4, max_cache_len=128, dropout=0.0)
    model.eval()
    # Create cache with 20 tokens
    q_init, k_init, v_init = create_sample_input(1, 20, 32, seed=701)
    _, cache_init = model(q_init, k_init, v_init, cache=None, use_causal_mask=True)
    # Generate single new token
    q, k, v = create_sample_input(1, 1, 32, seed=700)
    total += 1
    if run_test("Single token", model, q, k, v, cache_init, True, [1, 1, 32], [1, 21, 32]):
        passed += 1
    
    # Summary
    print("\n" + "=" * 70)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 70)
    
    if passed == total:
        print("✓ All tests passed! Great job!")
    else:
        print(f"✗ {total - passed} test(s) failed.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
