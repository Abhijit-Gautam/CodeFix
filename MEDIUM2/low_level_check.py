import h5py
import numpy as np

f = h5py.File('fashion_classifier (1).h5', 'r')

print("=== CHECKING LOW LEVEL H5 STRUCTURE ===\n")

# Get the low-level file ID
fid = f.id

# List all objects in the file
print("All objects in file:")
def visit_all(name):
    print(f"  {name}")
    return None

f.visit(visit_all)

# Check for any "soft links" or "external links"
print("\n=== Checking for links ===")
def check_links(name, obj):
    link = f.get(name, getlink=True)
    link_type = type(link).__name__
    if 'Soft' in link_type or 'External' in link_type:
        print(f"  {name}: {link_type} -> {link}")

f.visititems(check_links)

# Check each dataset's raw data for strings
print("\n=== Checking dataset contents for embedded strings ===")
def check_dataset_contents(name, obj):
    if isinstance(obj, h5py.Dataset):
        try:
            data = obj[()]
            # Convert to bytes and look for printable strings
            if data.dtype == np.float32 or data.dtype == np.float64:
                raw_bytes = data.tobytes()
                # Look for printable ASCII sequences
                import re
                text = raw_bytes.decode('latin-1', errors='replace')
                strings = re.findall(r'[\x20-\x7e]{8,}', text)
                if strings:
                    for s in strings[:5]:
                        if not any(x in s.lower() for x in ['nan', 'inf']):
                            print(f"  {name}: {s}")
        except Exception as e:
            pass

f.visititems(check_dataset_contents)

# Check for any comments in the HDF5 file
print("\n=== Checking object comments ===")
def check_comments(name, obj):
    try:
        comment = f.id.get_comment(name.encode())
        if comment:
            print(f"  {name}: {comment}")
    except:
        pass

f.visititems(check_comments)

# Try to access any custom metadata
print("\n=== Full attr dump with types ===")
for key in f.attrs.keys():
    val = f.attrs[key]
    print(f"{key}: type={type(val)}, dtype={getattr(val, 'dtype', 'N/A')}")
    if isinstance(val, bytes):
        print(f"  bytes: {val}")
    elif isinstance(val, str):
        # Check for hidden chars
        if any(ord(c) < 32 or ord(c) > 126 for c in val if c not in '\n\r\t'):
            print(f"  Contains non-printable chars!")
            print(f"  Raw: {[ord(c) for c in val[:50]]}")

f.close()

# Also check using h5dump-like approach
print("\n=== RAW ATTRIBUTE BYTES ===")
import struct

with open('fashion_classifier (1).h5', 'rb') as raw:
    content = raw.read()
    
# Search for attribute markers and their content
# Looking for patterns like attribute name followed by data
patterns_to_find = [
    b'flag',
    b'FLAG', 
    b'secret',
    b'hidden',
    b'TensorAI',
    b'TENSORAI',
    b'tensor_ai',
    b'AI_CODEFIX',
    b'AICODEFIX',
    b'key',
    b'password',
    b'token',
]

print("Searching for byte patterns:")
for pattern in patterns_to_find:
    if pattern in content:
        idx = content.find(pattern)
        print(f"Found {pattern} at offset {idx}")
        print(f"  Context: {content[idx-20:idx+50]}")
