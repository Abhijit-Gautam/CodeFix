import h5py
import numpy as np

# Read the h5 file and look for flag in weights
f = h5py.File('fashion_classifier (1).h5', 'r')

# Read the final dense layer bias (10 values)
bias = f['model_weights/dense_1/sequential/dense_1/bias'][:]
print("Dense_1 bias values:")
print(bias)

# Try interpreting as ASCII
print("\nBias as raw bytes:")
raw_bytes = bias.tobytes()
print(raw_bytes[:100])

# Try reading as int and converting to ASCII
print("\nTrying to find hidden data in bias...")
# Round to nearest int
bias_int = np.round(bias * 100).astype(int)
print(f"Bias * 100 rounded: {bias_int}")

# Check dense kernel for hidden message
kernel = f['model_weights/dense_1/sequential/dense_1/kernel'][:]
print(f"\nDense_1 kernel shape: {kernel.shape}")

# Check first row
first_row = kernel[0, :]
print(f"First row: {first_row}")

# Maybe the flag is hidden in specific weight indices?
# Check weight values that could be ASCII
for layer_name in ['conv2d', 'conv2d_1', 'conv2d_2', 'dense', 'dense_1']:
    layer_path = f'model_weights/{layer_name}'
    if layer_path in f:
        layer = f[layer_path]
        for name in ['sequential']:
            sublayer = layer.get(name, {})
            if hasattr(sublayer, 'keys'):
                for subname in sublayer.keys():
                    for wname in sublayer[subname].keys():
                        data = sublayer[subname][wname][:]
                        # Check if any values are near ASCII range (65-122)
                        flat = data.flatten()
                        in_ascii = flat[(flat > 32) & (flat < 127)]
                        if len(in_ascii) > 0:
                            print(f"\n{layer_name}/{subname}/{wname}: {len(in_ascii)} values in ASCII range")
                            if len(in_ascii) < 50:
                                ascii_chars = ''.join([chr(int(x)) for x in in_ascii])
                                print(f"  As ASCII: {ascii_chars}")

f.close()

# Let me also check if the file has any embedded strings that look like a flag
print("\n\n=== Searching for flag patterns in raw file ===")
with open('fashion_classifier (1).h5', 'rb') as raw:
    content = raw.read()
    
    # Search for TensorAI{ or FLAG{ or CTF{
    patterns = [b'TensorAI{', b'FLAG{', b'CTF{', b'flag{', b'secret{', b'SECRET']
    for pattern in patterns:
        idx = content.find(pattern)
        if idx != -1:
            # Found! Print surrounding context
            print(f"FOUND '{pattern.decode()}' at offset {idx}!")
            print(f"Context: {content[idx:idx+100]}")
    
    # Also search for just curly braces with content
    import re
    # Find all {...} patterns
    text_content = content.decode('latin-1', errors='ignore')
    brace_patterns = re.findall(r'\{[A-Za-z0-9_]+\}', text_content)
    if brace_patterns:
        print(f"\nPatterns with braces found: {brace_patterns[:20]}")
