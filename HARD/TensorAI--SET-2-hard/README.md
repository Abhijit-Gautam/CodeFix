# 🔥 HARD Challenge #2: KV-Cached Multi-Head Attention Debugger

## AI CODEFIX 2025 - Gen AI / Deep Learning Edition

---

## 🎯 Challenge Overview

Welcome to the **hardest challenge** of AI CODEFIX 2025!

You've been handed a broken implementation of **KV-Cached Multi-Head Attention** - the core mechanism powering modern Large Language Models (GPT, Claude, LLaMA). This implementation is designed for **efficient inference** by caching Key and Value computations across autoregressive generation steps.

**The problem**: This code contains bugs. Your task is to find and fix them all.

**Time limit**: 75-90 minutes
**Difficulty**: ⚡ VERY HARD (Harder than Hard Round)
**Domain**: Gen AI, Deep Learning, Transformer Architectures

---

## 📖 Background: Multi-Head Attention with KV-Caching

### What is Multi-Head Attention?

Multi-head attention is the fundamental building block of Transformer models. It allows the model to attend to different parts of the input sequence simultaneously.

**Standard attention formula**:
```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

Where:
- **Q** (Query): What we're looking for
- **K** (Key): What we're looking at
- **V** (Value): The actual information
- **d_k**: Dimension of key vectors (head_dim)

### Why KV-Caching?

In autoregressive generation (like ChatGPT generating text token-by-token), we:
1. Generate one token at a time
2. Use all previous tokens as context
3. Recompute attention over entire sequence

**Problem**: This is wasteful! K and V for previous tokens never change.

**Solution**: KV-Caching
- Store (cache) K and V from previous steps
- For new tokens: only compute new K, V
- Concatenate new K, V with cached versions
- Compute attention using cached context

**Result**: 10-100x faster inference!

### Multi-Head Attention

Instead of one attention mechanism, we use multiple "heads" in parallel:
- Split d_model into num_heads × head_dim
- Each head learns different patterns
- Concatenate head outputs
- Project back to d_model

---

## 🧩 Problem Statement

You are given `kv_attention.py` containing class `KVCachedMultiHeadAttention`.

**Your task**: Debug this implementation to make it work correctly.

### Key Components

```python
class KVCachedMultiHeadAttention:
    def forward(query, key, value, cache, use_causal_mask):
        # 1. Project Q, K, V
        # 2. Concatenate with cache (if exists)
        # 3. Split into multiple heads
        # 4. Compute attention scores
        # 5. Apply causal mask (prevent future positions)
        # 6. Softmax to get attention weights
        # 7. Apply attention to V
        # 8. Merge heads
        # 9. Output projection
        # 10. Update cache
```

### Input/Output Specification

**Inputs**:
- `query`: [batch_size, seq_len_q, d_model]
- `key`: [batch_size, seq_len_k, d_model]
- `value`: [batch_size, seq_len_v, d_model]
- `cache`: Optional dict with cached 'key' and 'value' tensors
- `use_causal_mask`: Boolean (True for autoregressive)

**Outputs**:
- `output`: [batch_size, seq_len_q, d_model]
- `new_cache`: Dict with updated 'key' and 'value' caches

### Expected Tensor Shapes

```
Input:        [batch, seq_len, d_model]
After proj:   [batch, seq_len, d_model]
After split:  [batch, num_heads, seq_len, head_dim]
Scores:       [batch, num_heads, seq_len_q, seq_len_k]
After attn:   [batch, num_heads, seq_len_q, head_dim]
After merge:  [batch, seq_len_q, d_model]
```

---

## 🔨 Your Task

### 1. Analyze the Code

Read through `kv_attention.py` carefully:
- Understand the overall flow
- Identify each major operation
- Trace tensor dimensions through the pipeline

### 2. Identify Bugs

The code contains bugs. Find them all.

⚠️ **WARNING**: Not all "suspicious" code is buggy! Some patterns may be intentional. Don't break working code.

### 3. Fix All Bugs

Make minimal, targeted fixes:
- Fix only actual bugs
- Don't refactor or "improve" working code
- Maintain the original structure
- Add comments if needed

### 4. Validate Your Solution

```bash
python validator.py --file kv_attention.py
```

The validator will run test cases and report pass/fail.

---

## 📋 Files Provided

```
hard_2/
├── README.md              # This file
├── kv_attention.py        # BUGGY implementation (FIX THIS!)
├── validator.py           # Test runner
├── test_cases.json        # 1 visible test (no expected output)
├── attention_debugger.py  # Optional AI agent skeleton (BONUS)
└── requirements.txt       # Dependencies (torch, numpy)
```

---

## 🚀 Getting Started

### Step 1: Setup Environment

```bash
cd hard_2
pip install -r requirements.txt
```

### Step 2: Understand the Problem

1. Read this entire README
2. Study the attention mechanism background
3. Review the code structure in `kv_attention.py`

### Step 3: Run Initial Test

```bash
python kv_attention.py
```

This will run a quick test. It may crash or produce wrong results due to bugs.

### Step 4: Debug Systematically

**Recommended approach**:

1. **Understand the algorithm**
   - Review attention mechanism theory
   - Trace through the code flow
   - Verify your understanding with diagrams

2. **Add debugging instrumentation**
   ```python
   print(f"Q shape: {Q.shape}")  # Add debugging prints
   print(f"K shape: {K.shape}")
   print(f"Attention weights sum: {attention_weights.sum(dim=-1)}")
   ```

3. **Test incrementally**
   - Fix one issue at a time
   - Run tests after each change
   - Verify you haven't broken anything

4. **Test different scenarios**
   - Basic attention (no cache)
   - With cache
   - Different batch sizes
   - Different sequence lengths
   - Edge cases

### Step 5: Validate

```bash
python validator.py --file kv_attention.py --verbose
```

---

## 📊 Evaluation Criteria

Your solution will be graded on:

### Automatic Testing (70%)
- **Visible test** (10%): Basic functionality
- **Hidden tests 1-4** (25%): Core features
- **Hidden tests 5-8** (25%): Edge cases
- **Hidden tests 9-10** (10%): Stress tests

### Manual Code Review (30%)
- **Bug fixes** (15%): Correctness of fixes
- **Code quality** (10%): Clean fixes, no new bugs
- **AI Debugger bonus** (+10%): Implement `attention_debugger.py` (optional)

### Partial Credit Available

You can earn partial credit based on progress made.

---

## 🧠 Key Concepts to Review

### 1. Scaled Dot-Product Attention

```
scores = (Q @ K^T) / sqrt(d_k)
attention = softmax(scores)
output = attention @ V
```

### 2. Multi-Head Mechanism

```
Split:  [B, S, D] → [B, S, H, D/H] → [B, H, S, D/H]
Merge:  [B, H, S, D/H] → [B, S, H, D/H] → [B, S, D]
```

Where: B=batch, S=seq_len, D=d_model, H=num_heads

### 3. Causal Masking

For position i, can only attend to positions <= i:
```
Mask matrix (4x4):
[[1, 0, 0, 0],
 [1, 1, 0, 0],
 [1, 1, 1, 0],
 [1, 1, 1, 1]]
```

With cache: current tokens attend to all cached tokens + current tokens up to position i.

### 4. KV-Cache Management

```python
# First generation step
output1, cache1 = model(q, k, v, cache=None)
# cache1['key'].shape = [batch, seq_len, d_model]

# Next token
output2, cache2 = model(q_new, k_new, v_new, cache=cache1)
# cache2['key'].shape = [batch, seq_len + 1, d_model]
```

---

## 🔍 Debugging Strategy

### Useful Debugging Techniques

```python
# Check shapes
print(f"Tensor shape: {tensor.shape}")

# Check values
print(f"Min: {tensor.min()}, Max: {tensor.max()}, Mean: {tensor.mean()}")

# Check for NaN/Inf
assert not torch.isnan(tensor).any(), "Found NaN!"
assert not torch.isinf(tensor).any(), "Found Inf!"

# Verify attention properties
# Attention weights should sum to 1.0 after softmax
sums = attention_weights.sum(dim=-1)
print(f"Attention weight sums: {sums}")  # Should be all ~1.0
```

### Questions to Ask Yourself

- Are tensor dimensions correct at each step?
- Are matrix operations using the right dimensions?
- Is the cache being managed correctly?
- Are mathematical formulas implemented correctly?
- Does it handle edge cases (batch=1, single head, etc.)?
- Does it work with and without cache?

---

## ⚠️ Important Notes

1. **Do NOT use high-level libraries**
   - No `transformers`, `fairseq`, etc.
   - Implement from scratch using PyTorch primitives

2. **Some code may be intentionally written a certain way**
   - Not all "suspicious" code is buggy
   - Don't fix what isn't broken
   - Test before assuming something is wrong

3. **Focus on correctness first**
   - Fix bugs before optimizing
   - Keep the original structure
   - Minimal changes only

4. **Test incrementally**
   - Fix one thing at a time
   - Validate after each fix
   - Don't batch changes without testing

5. **AI tools may not help**
   - This is a custom implementation
   - ChatGPT/Gemini may suggest incorrect fixes
   - You must understand the algorithm

---

## 🎁 Bonus Challenge (Optional +10%)

Implement `attention_debugger.py` - an AI agent that can:
1. Automatically detect bugs in attention implementations
2. Suggest fixes with explanations
3. Validate fixes before applying

This demonstrates advanced understanding!

---

## 📚 Resources

### Attention Mechanism
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Original Transformer paper)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)

### KV-Caching
- [Accelerating Large Language Model Decoding with KV-Cache](https://arxiv.org/abs/2311.17035)
- [LLM Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/)

### PyTorch Tensor Operations
- [PyTorch Documentation](https://pytorch.org/docs/stable/tensors.html)

---

## 💡 Tips

1. **Read the code thoroughly** before making changes
2. **Draw diagrams** of tensor transformations
3. **Add debug prints** to understand what's happening
4. **Test incrementally** - one change at a time
5. **Think about edge cases** (small batches, single heads, etc.)
6. **Verify against theory** - does the implementation match the math?
7. **Don't trust everything** - verify assumptions
8. **Watch for common pitfalls** in tensor operations
9. **Check both training and inference** scenarios
10. **Stay systematic** - don't make random changes

---

## 🏆 Submission Checklist

Before submitting:
- [ ] Code runs without crashing
- [ ] Validator passes tests
- [ ] Fixed code is documented
- [ ] No new bugs introduced
- [ ] Cache management works
- [ ] Tested with different configurations
- [ ] Edge cases handled
- [ ] (Bonus) Implemented AI debugger

---

## ❓ FAQ

**Q: How many bugs are there?**
A: We won't tell you - finding them all is part of the challenge!

**Q: Can I use ChatGPT/Gemini/Claude?**
A: Yes, you have full internet access. However, this is a custom implementation and AI tools may not provide correct fixes. Use them wisely.

**Q: Should I refactor the code?**
A: No! Only fix bugs. Minimal targeted fixes only.

**Q: What if I can't fix all bugs?**
A: Partial credit is available. Do your best!

**Q: Can I add new functions/methods?**
A: Avoid it if possible. Only if absolutely necessary.

**Q: How do I know if code is intentional or buggy?**
A: Test it! If tests fail or results are wrong, it's likely a bug.

**Q: The visible test doesn't show expected output?**
A: Correct! This prevents reverse-engineering. Focus on understanding the algorithm.

---

## 🚨 Important Reminders

- This is a **debugging challenge** - fix existing code, don't rewrite
- **Not all suspicious code is buggy** - test before changing
- **Test frequently** - validate each change
- **Time management** - 75-90 minutes is limited!
- **Stay systematic** - understand before fixing

---

Good luck, and may your gradients flow smoothly! 🚀

**Remember**: This is challenging by design. Do your best, learn from the experience, and have fun debugging!

---

*AI CODEFIX 2025 - Where Code Meets Intelligence*
