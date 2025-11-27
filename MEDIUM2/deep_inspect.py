import h5py
import numpy as np

f = h5py.File('fashion_classifier (1).h5', 'r')

print("=== HDF5 File Properties ===")
print(f"Driver: {f.driver}")
print(f"Filename: {f.filename}")
print(f"Mode: {f.mode}")
print(f"User block size: {f.userblock_size}")
print(f"ID: {f.id}")

# Check if there's a userblock
if f.userblock_size > 0:
    print(f"\n=== USER BLOCK DATA ({f.userblock_size} bytes) ===")
    with open('fashion_classifier (1).h5', 'rb') as raw:
        userblock = raw.read(f.userblock_size)
        print(userblock)

# Check all items recursively and print everything
print("\n=== DEEP INSPECTION OF ALL ITEMS ===")
def deep_inspect(name, obj):
    print(f"\n--- {name} ---")
    print(f"  Type: {type(obj).__name__}")
    
    # Print all attributes
    if len(obj.attrs) > 0:
        print(f"  Attributes:")
        for k, v in obj.attrs.items():
            if isinstance(v, bytes):
                v = v.decode('utf-8', errors='replace')
            elif isinstance(v, np.ndarray):
                if v.dtype.kind in ['S', 'U', 'O']:
                    v = [x.decode('utf-8', errors='replace') if isinstance(x, bytes) else str(x) for x in v.flatten()]
            print(f"    {k}: {v}")
    
    # If it's a dataset, print some info
    if isinstance(obj, h5py.Dataset):
        print(f"  Shape: {obj.shape}")
        print(f"  Dtype: {obj.dtype}")
        # Check for string data
        if obj.dtype.kind in ['S', 'U', 'O']:
            print(f"  String Data: {obj[()]}")

f.visititems(deep_inspect)

# Check file-level attrs again more carefully
print("\n=== FILE LEVEL ATTRIBUTES (FULL) ===")
for k, v in f.attrs.items():
    if isinstance(v, bytes):
        v = v.decode('utf-8', errors='replace')
    print(f"{k}:")
    print(f"  Type: {type(v)}")
    print(f"  Value: {v}")
    print()

f.close()

# Also check the raw file for the first few KB
print("\n=== RAW FILE HEADER (first 2KB) ===")
with open('fashion_classifier (1).h5', 'rb') as raw:
    header = raw.read(2048)
    # Print hex dump
    for i in range(0, min(512, len(header)), 16):
        hex_part = ' '.join(f'{b:02x}' for b in header[i:i+16])
        ascii_part = ''.join(chr(b) if 32 <= b < 127 else '.' for b in header[i:i+16])
        print(f"{i:04x}: {hex_part:<48} {ascii_part}")
