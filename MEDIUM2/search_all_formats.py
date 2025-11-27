import h5py
import re
import base64

# Read entire file as bytes
with open('fashion_classifier (1).h5', 'rb') as f:
    data = f.read()

text = data.decode('latin-1', errors='replace')

print("=== Searching for various flag formats ===\n")

# Different flag patterns
patterns = [
    r'flag\{[^}]+\}',
    r'FLAG\{[^}]+\}',
    r'CTF\{[^}]+\}',
    r'ctf\{[^}]+\}',
    r'TensorAI\{[^}]+\}',
    r'TENSORAI\{[^}]+\}',
    r'tensorai\{[^}]+\}',
    r'Tensor_AI\{[^}]+\}',
    r'AI_?CODEFIX[^}]*\}',
    r'AICODEFIX[^}]*\}',
    r'codefix\{[^}]+\}',
    r'CODEFIX\{[^}]+\}',
    # Underscores and dashes
    r'flag_[a-zA-Z0-9_]+',
    r'FLAG_[a-zA-Z0-9_]+',
    r'secret_[a-zA-Z0-9_]+',
    # With brackets
    r'\[[A-Za-z0-9_]+\]',
    # Hex patterns
    r'0x[a-fA-F0-9]{8,}',
    # UUID-like
    r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}',
]

for pattern in patterns:
    matches = re.findall(pattern, text, re.IGNORECASE)
    if matches:
        print(f"Pattern '{pattern}':")
        for m in matches[:10]:
            print(f"  {m}")

# Check for common CTF words
print("\n=== Searching for CTF-related words ===")
words = ['flag', 'secret', 'hidden', 'password', 'key', 'token', 'ctf', 'challenge', 'winner', 'congrat', 'found', 'correct']
for word in words:
    if word.encode() in data.lower():
        # Find position and context
        pos = data.lower().find(word.encode())
        context = data[max(0, pos-30):pos+50]
        print(f"Found '{word}' at {pos}: {context}")

# Check h5 file attributes more carefully
print("\n=== Checking ALL h5 attributes ===")
f = h5py.File('fashion_classifier (1).h5', 'r')

def check_all_attrs(name, obj):
    for key, value in obj.attrs.items():
        val_str = str(value)
        # Check for anything that looks like a flag
        if '{' in val_str and '}' in val_str:
            print(f"[{name}].{key} contains braces: {value}")
        if re.search(r'[A-Z]{3,}_', val_str):
            print(f"[{name}].{key} has uppercase pattern: {value[:200]}")

# Check file-level
for key, value in f.attrs.items():
    val_str = str(value)
    if '{' in val_str and '}' in val_str and 'class_name' not in val_str:
        print(f"[FILE].{key} (non-class braces): {value[:500]}")

f.visititems(check_all_attrs)

# Check for custom/unusual attribute names
print("\n=== Unusual attribute names ===")
def find_unusual(name, obj):
    for key in obj.attrs.keys():
        if key not in ['weight_names', 'layer_names', 'backend', 'keras_version']:
            print(f"[{name}] has unusual attr: {key} = {obj.attrs[key]}")

f.visititems(find_unusual)

# Check file-level for unusual attrs
standard_attrs = ['backend', 'keras_version', 'model_config', 'training_config']
for key in f.attrs.keys():
    if key not in standard_attrs:
        print(f"[FILE] unusual attr: {key} = {f.attrs[key]}")

f.close()

# Look for base64 encoded flags
print("\n=== Checking for base64 encoded flags ===")
b64_pattern = r'[A-Za-z0-9+/]{16,}={0,2}'
b64_matches = re.findall(b64_pattern, text)
for m in b64_matches[:30]:
    try:
        decoded = base64.b64decode(m).decode('utf-8', errors='replace')
        if any(c.isalpha() for c in decoded) and not decoded.startswith('\x00'):
            print(f"B64: {m[:50]} -> {decoded[:100]}")
    except:
        pass

# Check for hex-encoded strings
print("\n=== Checking for hex-encoded strings ===")
hex_pattern = r'[0-9a-fA-F]{20,}'
hex_matches = re.findall(hex_pattern, text)
for m in hex_matches[:20]:
    try:
        decoded = bytes.fromhex(m).decode('utf-8', errors='replace')
        if any(c.isalpha() for c in decoded):
            print(f"HEX: {m[:40]} -> {decoded[:50]}")
    except:
        pass
