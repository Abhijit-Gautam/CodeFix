"""
Generate actual test case values for test_cases_hidden.json
Uses fixed model seeds for reproducibility
"""
import torch
import json
import sys
sys.path.insert(0, '.')
from kv_attention import KVCachedMultiHeadAttention, create_sample_input

def tensor_to_list(t):
    return t.detach().cpu().numpy().tolist()

def create_model_with_seed(d_model, num_heads, max_cache_len, dropout, model_seed):
    """Create model with deterministic weights by setting seed before model creation"""
    torch.manual_seed(model_seed)
    model = KVCachedMultiHeadAttention(d_model=d_model, num_heads=num_heads, 
                                        max_cache_len=max_cache_len, dropout=dropout)
    model.eval()
    return model

test_cases = []

# Test 1: Basic attention without cache
print("Generating Test 1...")
model_seed_1 = 1000
model = create_model_with_seed(16, 2, 128, 0.0, model_seed_1)
q, k, v = create_sample_input(1, 4, 16, seed=42)
with torch.no_grad():
    output, cache = model(q, k, v, cache=None, use_causal_mask=True)
test_cases.append({
    "id": 1,
    "name": "Basic attention without cache",
    "model_seed": model_seed_1,
    "config": {"d_model": 16, "num_heads": 2, "max_cache_len": 128, "dropout": 0.0},
    "seed": 42,
    "inputs": {
        "query": {"shape": [1, 4, 16], "values": tensor_to_list(q)},
        "key": {"shape": [1, 4, 16], "values": tensor_to_list(k)},
        "value": {"shape": [1, 4, 16], "values": tensor_to_list(v)},
        "cache": None,
        "use_causal_mask": True
    },
    "expected": {
        "output": {"shape": [1, 4, 16], "values": tensor_to_list(output)},
        "cache": {
            "key": {"shape": list(cache['key'].shape), "values": tensor_to_list(cache['key'])},
            "value": {"shape": list(cache['value'].shape), "values": tensor_to_list(cache['value'])}
        }
    }
})

# Test 2: Attention with cached context (reuses model from test 1)
print("Generating Test 2...")
q2, k2, v2 = create_sample_input(1, 1, 16, seed=43)
with torch.no_grad():
    output2, cache2 = model(q2, k2, v2, cache=cache, use_causal_mask=True)
test_cases.append({
    "id": 2,
    "name": "Attention with cached context",
    "model_seed": model_seed_1,
    "config": {"d_model": 16, "num_heads": 2, "max_cache_len": 128, "dropout": 0.0},
    "seed": 43,
    "inputs": {
        "query": {"shape": [1, 1, 16], "values": tensor_to_list(q2)},
        "key": {"shape": [1, 1, 16], "values": tensor_to_list(k2)},
        "value": {"shape": [1, 1, 16], "values": tensor_to_list(v2)},
        "cache": {
            "key": {"shape": list(cache['key'].shape), "values": tensor_to_list(cache['key'])},
            "value": {"shape": list(cache['value'].shape), "values": tensor_to_list(cache['value'])}
        },
        "use_causal_mask": True
    },
    "expected": {
        "output": {"shape": [1, 1, 16], "values": tensor_to_list(output2)},
        "cache": {
            "key": {"shape": list(cache2['key'].shape), "values": tensor_to_list(cache2['key'])},
            "value": {"shape": list(cache2['value'].shape), "values": tensor_to_list(cache2['value'])}
        }
    }
})

# Test 3: Multi-batch attention
print("Generating Test 3...")
model_seed_3 = 1001
model3 = create_model_with_seed(32, 4, 128, 0.0, model_seed_3)
q3, k3, v3 = create_sample_input(4, 8, 32, seed=100)
with torch.no_grad():
    output3, cache3 = model3(q3, k3, v3, cache=None, use_causal_mask=True)
test_cases.append({
    "id": 3,
    "name": "Multi-batch attention",
    "model_seed": model_seed_3,
    "config": {"d_model": 32, "num_heads": 4, "max_cache_len": 128, "dropout": 0.0},
    "seed": 100,
    "inputs": {
        "query": {"shape": [4, 8, 32], "values": tensor_to_list(q3)},
        "key": {"shape": [4, 8, 32], "values": tensor_to_list(k3)},
        "value": {"shape": [4, 8, 32], "values": tensor_to_list(v3)},
        "cache": None,
        "use_causal_mask": True
    },
    "expected": {
        "output": {"shape": [4, 8, 32], "values": tensor_to_list(output3)},
        "cache": {
            "key": {"shape": list(cache3['key'].shape), "values": tensor_to_list(cache3['key'])},
            "value": {"shape": list(cache3['value'].shape), "values": tensor_to_list(cache3['value'])}
        }
    }
})

# Test 4: No causal mask
print("Generating Test 4...")
model_seed_4 = 1002
model4 = create_model_with_seed(16, 2, 128, 0.0, model_seed_4)
q4, k4, v4 = create_sample_input(1, 6, 16, seed=50)
with torch.no_grad():
    output4, cache4 = model4(q4, k4, v4, cache=None, use_causal_mask=False)
test_cases.append({
    "id": 4,
    "name": "No causal mask",
    "model_seed": model_seed_4,
    "config": {"d_model": 16, "num_heads": 2, "max_cache_len": 128, "dropout": 0.0},
    "seed": 50,
    "inputs": {
        "query": {"shape": [1, 6, 16], "values": tensor_to_list(q4)},
        "key": {"shape": [1, 6, 16], "values": tensor_to_list(k4)},
        "value": {"shape": [1, 6, 16], "values": tensor_to_list(v4)},
        "cache": None,
        "use_causal_mask": False
    },
    "expected": {
        "output": {"shape": [1, 6, 16], "values": tensor_to_list(output4)},
        "cache": {
            "key": {"shape": list(cache4['key'].shape), "values": tensor_to_list(cache4['key'])},
            "value": {"shape": list(cache4['value'].shape), "values": tensor_to_list(cache4['value'])}
        }
    }
})

# Test 5: Single head attention
print("Generating Test 5...")
model_seed_5 = 1003
model5 = create_model_with_seed(32, 1, 128, 0.0, model_seed_5)
q5, k5, v5 = create_sample_input(2, 5, 32, seed=200)
with torch.no_grad():
    output5, cache5 = model5(q5, k5, v5, cache=None, use_causal_mask=True)
test_cases.append({
    "id": 5,
    "name": "Single head attention",
    "model_seed": model_seed_5,
    "config": {"d_model": 32, "num_heads": 1, "max_cache_len": 128, "dropout": 0.0},
    "seed": 200,
    "inputs": {
        "query": {"shape": [2, 5, 32], "values": tensor_to_list(q5)},
        "key": {"shape": [2, 5, 32], "values": tensor_to_list(k5)},
        "value": {"shape": [2, 5, 32], "values": tensor_to_list(v5)},
        "cache": None,
        "use_causal_mask": True
    },
    "expected": {
        "output": {"shape": [2, 5, 32], "values": tensor_to_list(output5)},
        "cache": {
            "key": {"shape": list(cache5['key'].shape), "values": tensor_to_list(cache5['key'])},
            "value": {"shape": list(cache5['value'].shape), "values": tensor_to_list(cache5['value'])}
        }
    }
})

# Test 6: Many heads attention
print("Generating Test 6...")
model_seed_6 = 1004
model6 = create_model_with_seed(64, 8, 256, 0.0, model_seed_6)
q6, k6, v6 = create_sample_input(1, 10, 64, seed=300)
with torch.no_grad():
    output6, cache6 = model6(q6, k6, v6, cache=None, use_causal_mask=True)
test_cases.append({
    "id": 6,
    "name": "Many heads attention",
    "model_seed": model_seed_6,
    "config": {"d_model": 64, "num_heads": 8, "max_cache_len": 256, "dropout": 0.0},
    "seed": 300,
    "inputs": {
        "query": {"shape": [1, 10, 64], "values": tensor_to_list(q6)},
        "key": {"shape": [1, 10, 64], "values": tensor_to_list(k6)},
        "value": {"shape": [1, 10, 64], "values": tensor_to_list(v6)},
        "cache": None,
        "use_causal_mask": True
    },
    "expected": {
        "output": {"shape": [1, 10, 64], "values": tensor_to_list(output6)},
        "cache": {
            "key": {"shape": list(cache6['key'].shape), "values": tensor_to_list(cache6['key'])},
            "value": {"shape": list(cache6['value'].shape), "values": tensor_to_list(cache6['value'])}
        }
    }
})

# Test 7: Long sequence
print("Generating Test 7...")
model_seed_7 = 1005
model7 = create_model_with_seed(32, 4, 256, 0.0, model_seed_7)
q7, k7, v7 = create_sample_input(1, 64, 32, seed=400)
with torch.no_grad():
    output7, cache7 = model7(q7, k7, v7, cache=None, use_causal_mask=True)
test_cases.append({
    "id": 7,
    "name": "Long sequence",
    "model_seed": model_seed_7,
    "config": {"d_model": 32, "num_heads": 4, "max_cache_len": 256, "dropout": 0.0},
    "seed": 400,
    "inputs": {
        "query": {"shape": [1, 64, 32], "values": tensor_to_list(q7)},
        "key": {"shape": [1, 64, 32], "values": tensor_to_list(k7)},
        "value": {"shape": [1, 64, 32], "values": tensor_to_list(v7)},
        "cache": None,
        "use_causal_mask": True
    },
    "expected": {
        "output": {"shape": [1, 64, 32], "values": tensor_to_list(output7)},
        "cache": {
            "key": {"shape": list(cache7['key'].shape), "values": tensor_to_list(cache7['key'])},
            "value": {"shape": list(cache7['value'].shape), "values": tensor_to_list(cache7['value'])}
        }
    }
})

# Test 8: Incremental generation with cache
print("Generating Test 8...")
model_seed_8 = 1006
model8 = create_model_with_seed(16, 2, 128, 0.0, model_seed_8)
# First create cache
q8_init, k8_init, v8_init = create_sample_input(1, 10, 16, seed=501)
with torch.no_grad():
    _, cache8_init = model8(q8_init, k8_init, v8_init, cache=None, use_causal_mask=True)
# Then test with new tokens
q8, k8, v8 = create_sample_input(1, 5, 16, seed=500)
with torch.no_grad():
    output8, cache8 = model8(q8, k8, v8, cache=cache8_init, use_causal_mask=True)
test_cases.append({
    "id": 8,
    "name": "Incremental generation",
    "model_seed": model_seed_8,
    "config": {"d_model": 16, "num_heads": 2, "max_cache_len": 128, "dropout": 0.0},
    "seed": 500,
    "inputs": {
        "query": {"shape": [1, 5, 16], "values": tensor_to_list(q8)},
        "key": {"shape": [1, 5, 16], "values": tensor_to_list(k8)},
        "value": {"shape": [1, 5, 16], "values": tensor_to_list(v8)},
        "cache": {
            "key": {"shape": list(cache8_init['key'].shape), "values": tensor_to_list(cache8_init['key'])},
            "value": {"shape": list(cache8_init['value'].shape), "values": tensor_to_list(cache8_init['value'])}
        },
        "use_causal_mask": True
    },
    "expected": {
        "output": {"shape": [1, 5, 16], "values": tensor_to_list(output8)},
        "cache": {
            "key": {"shape": list(cache8['key'].shape), "values": tensor_to_list(cache8['key'])},
            "value": {"shape": list(cache8['value'].shape), "values": tensor_to_list(cache8['value'])}
        }
    }
})

# Write to JSON
output_data = {
    "description": "Hidden test cases for KV-Cached Multi-Head Attention",
    "test_cases": test_cases
}

with open('test_cases_hidden.json', 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"\nGenerated {len(test_cases)} test cases to test_cases_hidden.json")
