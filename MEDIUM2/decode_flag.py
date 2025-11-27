import h5py
import numpy as np

f = h5py.File('fashion_classifier (1).h5', 'r')

print("=== Looking for ASCII encoded in float32 bytes ===\n")

# The dense_1 layer has 10 outputs - could be interesting
dense1_bias = f['model_weights/dense_1/sequential/dense_1/bias'][()]
dense1_kernel = f['model_weights/dense_1/sequential/dense_1/kernel'][()]

print("dense_1 bias:", dense1_bias)
print("dense_1 bias as bytes:", dense1_bias.tobytes())

# Check if values when multiplied by some factor give ASCII
for multiplier in [1, 10, 100, 1000, 10000]:
    vals = (dense1_bias * multiplier).astype(int)
    if all(32 <= v <= 126 for v in vals if v != 0):
        print(f"Multiplier {multiplier}: {[chr(v) for v in vals]}")

# Look for the flag in hex representation
print("\n=== Checking hex patterns ===")
raw_bytes = dense1_bias.tobytes()
hex_str = raw_bytes.hex()
print(f"Bias hex: {hex_str}")

# Try to decode pairs as ASCII
ascii_from_hex = ''.join(chr(b) for b in raw_bytes if 32 <= b <= 126)
print(f"ASCII chars from bytes: {ascii_from_hex}")

# Check the file for "secret" or "flag" in any form
print("\n=== Searching raw file for keywords ===")
with open('fashion_classifier (1).h5', 'rb') as raw:
    content = raw.read()
    
# Search for variations
keywords = [b'secret', b'Secret', b'SECRET', b'flag', b'Flag', b'FLAG', b'hidden', b'Hidden', b'HIDDEN', b'key', b'pass']
for kw in keywords:
    pos = content.find(kw)
    if pos != -1:
        print(f"Found '{kw.decode()}' at position {pos}")
        context = content[pos:pos+100]
        print(f"  Context: {context}")

# Also look for patterns like tensor{, TensorAI{, etc.
patterns = [b'ensor', b'ENSOR', b'lag{', b'LAG{', b'ecret', b'ECRET']
for p in patterns:
    pos = content.find(p)
    if pos != -1:
        print(f"Found pattern '{p.decode()}' at {pos}: {content[max(0,pos-5):pos+50]}")

# Check metadata strings more carefully
print("\n=== Full model_config attribute ===")
model_config = f.attrs['model_config']
print(model_config)

# Check training_config
print("\n=== Full training_config attribute ===")
training_config = f.attrs['training_config']
print(training_config)

f.close()
