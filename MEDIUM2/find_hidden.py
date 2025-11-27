import h5py
import re

# Read entire file as bytes
with open('fashion_classifier (1).h5', 'rb') as f:
    data = f.read()

# Check the last 10KB of the file
print("=== LAST 5KB OF FILE (looking for hidden data) ===")
tail = data[-5000:]
text = tail.decode('latin-1', errors='replace')

# Print printable strings
strings = re.findall(r'[\x20-\x7e]{5,}', text)
for s in strings:
    print(s)

# Also check raw bytes pattern like "TensorAI" or "flag"
print("\n=== Searching entire file for specific patterns ===")
patterns = [b'TensorAI', b'TENSORAI', b'flag', b'FLAG', b'CTF', b'ctf', b'secret', b'hidden']
for p in patterns:
    pos = data.find(p)
    if pos != -1:
        print(f"Found '{p.decode()}' at position {pos}")
        # Show context around it
        context = data[max(0, pos-20):pos+100]
        print(f"  Context: {context}")

# Check for any unusual patterns - ASCII text after weights
print("\n=== Looking for ASCII text after the last known data structure ===")
# Find all ASCII printable regions
regions = []
current_start = None
for i, b in enumerate(data):
    if 32 <= b < 127:
        if current_start is None:
            current_start = i
    else:
        if current_start is not None and i - current_start >= 10:
            regions.append((current_start, i, data[current_start:i]))
        current_start = None

# Print regions that don't look like standard keras metadata
print(f"\nFound {len(regions)} text regions")
for start, end, content in regions[-30:]:  # Last 30 regions
    text = content.decode('latin-1', errors='replace')
    # Skip obvious keras stuff
    if not any(x in text.lower() for x in ['keras', 'tensorflow', 'sequential', 'adam', 'float32', 'kernel', 'bias', 'momentum', 'velocity', 'batch_norm', 'conv2d', 'dense', 'moving']):
        print(f"\nOffset {start}-{end}: {text[:200]}")
