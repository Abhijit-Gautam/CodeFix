import re

# Read the entire file as binary
with open('fashion_classifier (1).h5', 'rb') as f:
    data = f.read()

print(f"File size: {len(data)} bytes")

# Extract all printable ASCII strings of length 4+
strings = []
current = b''
for byte in data:
    if 32 <= byte <= 126:
        current += bytes([byte])
    else:
        if len(current) >= 4:
            strings.append(current.decode('ascii'))
        current = b''

# Also add the last one
if len(current) >= 4:
    strings.append(current.decode('ascii'))

print(f"Found {len(strings)} strings of length 4+")

# Look for any unusual strings (not standard keras/tensorflow stuff)
keywords = ['keras', 'tensorflow', 'sequential', 'adam', 'float32', 'conv2d', 
            'dense', 'batch_norm', 'kernel', 'bias', 'weight', 'momentum', 
            'velocity', 'moving', 'gamma', 'beta', 'optimizer', 'layer',
            'config', 'class_name', 'trainable', 'dtype', 'module', 'initializer',
            'regularizer', 'constraint', 'activation', 'relu', 'softmax',
            'pool', 'dropout', 'flatten', 'input', 'shape', 'null', 'true', 'false',
            'iteration', 'learning', 'rate', 'loss', 'accuracy', 'metrics']

print("\n=== Unusual strings (not matching keras/tf keywords) ===")
for s in strings:
    s_lower = s.lower()
    is_standard = any(kw in s_lower for kw in keywords)
    # Also skip pure numbers or very short strings
    if not is_standard and len(s) > 4 and not s.replace('.','').replace('-','').replace('+','').replace('e','').isdigit():
        print(f"  '{s}'")

# Specifically look for flag-like patterns
print("\n=== Looking for flag patterns ===")
patterns = [
    r'[Ff][Ll][Aa][Gg]',
    r'[Ss][Ee][Cc][Rr][Ee][Tt]',
    r'[Hh][Ii][Dd][Dd][Ee][Nn]',
    r'[Kk][Ee][Yy][:=]',
    r'[Pp][Aa][Ss][Ss]',
    r'\{[A-Za-z0-9_]+\}',  # Anything in braces
    r'CTF',
    r'TensorAI',
]

text = data.decode('latin-1')
for pattern in patterns:
    matches = list(re.finditer(pattern, text))
    if matches:
        for m in matches:
            start = max(0, m.start() - 20)
            end = min(len(text), m.end() + 50)
            print(f"Pattern '{pattern}' found at {m.start()}: ...{text[start:end]}...")
