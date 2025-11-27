import h5py
import numpy as np

f = h5py.File('fashion_classifier (1).h5', 'r')

print("=== Checking all datasets for hidden data ===\n")

def check_all_datasets(name, obj):
    if isinstance(obj, h5py.Dataset):
        data = obj[()]
        
        # Check if data could encode ASCII characters
        if data.dtype in [np.float32, np.float64]:
            # Check if values look like ASCII codes
            flat = data.flatten()
            
            # Method 1: Check if values are close to integers in ASCII range
            rounded = np.round(flat).astype(int)
            if np.all((rounded >= 32) & (rounded <= 126)):
                ascii_str = ''.join(chr(int(x)) for x in rounded if 32 <= x <= 126)
                if len(ascii_str) > 3:
                    print(f"{name}: Possible ASCII: {ascii_str[:100]}")
            
            # Method 2: Check first few bytes of raw data
            raw_bytes = data.tobytes()[:100]
            printable = ''.join(chr(b) if 32 <= b < 127 else '.' for b in raw_bytes)
            if any(c.isalpha() for c in printable[:20]):
                print(f"{name}: Raw bytes start: {printable}")

f.visititems(check_all_datasets)

# Also check specific small datasets that might contain flags
print("\n=== Small datasets (potential flag storage) ===")
def find_small_datasets(name, obj):
    if isinstance(obj, h5py.Dataset):
        if obj.size <= 128:  # Small enough to be a flag
            data = obj[()]
            print(f"\n{name}:")
            print(f"  Shape: {obj.shape}, Dtype: {obj.dtype}")
            print(f"  Data: {data}")
            
            # Try interpreting as bytes
            if data.dtype == np.float32:
                raw = data.tobytes()
                # Check for printable strings
                printable = ''.join(chr(b) if 32 <= b < 127 else '.' for b in raw)
                print(f"  As bytes: {printable[:200]}")

f.visititems(find_small_datasets)

f.close()
