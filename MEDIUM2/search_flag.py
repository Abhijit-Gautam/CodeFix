import re

# Read the h5 file as binary
with open('fashion_classifier (1).h5', 'rb') as f:
    data = f.read()

# Convert to string for searching (ignoring decode errors)
text = data.decode('latin-1')

# Search for common flag patterns
patterns = [
    r'flag\{[^}]+\}',
    r'FLAG\{[^}]+\}',
    r'CTF\{[^}]+\}',
    r'ctf\{[^}]+\}',
    r'TensorAI\{[^}]+\}',
    r'TENSORAI\{[^}]+\}',
    r'tensor[_-]?ai',
    r'secret[_:]',
    r'hidden[_:]',
    r'password',
]

print("=== Searching for flag patterns ===")
for pattern in patterns:
    matches = re.findall(pattern, text, re.IGNORECASE)
    if matches:
        print(f"Pattern '{pattern}': {matches}")

# Also look for any printable strings that might be interesting
print("\n=== All long printable strings (>20 chars) ===")
strings = re.findall(r'[\x20-\x7e]{20,}', text)
for s in strings[:50]:  # First 50
    if not any(x in s.lower() for x in ['keras', 'tensorflow', 'config', 'sequential', 'adam', 'class_name', 'initializer']):
        print(s)

# Look for base64 encoded strings
print("\n=== Potential Base64 strings ===")
b64_pattern = r'[A-Za-z0-9+/]{20,}={0,2}'
b64_matches = re.findall(b64_pattern, text)
for m in b64_matches[:20]:
    print(m)
    try:
        import base64
        decoded = base64.b64decode(m)
        if decoded.isascii():
            print(f"  -> Decoded: {decoded}")
    except:
        pass
