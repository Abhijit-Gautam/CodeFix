import h5py

f = h5py.File('fashion_classifier (1).h5', 'r')

# Try to access 'secret' or 'flag' directly as keys
print("=== Trying to access 'secret' and 'flag' keys directly ===")
try:
    print(f"f['secret'] = {f['secret']}")
except KeyError as e:
    print(f"No 'secret' key: {e}")

try:
    print(f"f['flag'] = {f['flag']}")
except KeyError as e:
    print(f"No 'flag' key: {e}")

# List all top-level keys
print("\n=== All top-level keys ===")
for key in f.keys():
    print(f"  {key}")

# Try common hidden key patterns
hidden_patterns = ['_secret', '__secret', 'secret_', '.secret', 'hidden', '_flag', '__flag', 'flag_', '.flag', '_hidden', '__hidden']
print("\n=== Trying hidden key patterns ===")
for pattern in hidden_patterns:
    try:
        val = f[pattern]
        print(f"FOUND: f['{pattern}'] = {val}")
    except:
        pass

# Check for any attributes with unusual names at file level
print("\n=== All file-level attribute names (raw) ===")
for key in f.attrs.keys():
    val = f.attrs[key]
    if isinstance(val, bytes):
        val = val.decode('utf-8', errors='replace')
    # Show first 100 chars
    val_str = str(val)[:100]
    print(f"  '{key}': {val_str}...")

# Also scan all groups for any unusual keys
print("\n=== All group/dataset keys at each level ===")
def list_keys(name, obj):
    if hasattr(obj, 'keys'):
        keys = list(obj.keys())
        if keys:
            # Look for non-standard keys
            for k in keys:
                if k not in ['sequential', 'adam'] and not k.startswith(('conv2d', 'batch_norm', 'dense', 'max_pool', 'dropout', 'flatten', 'top_level', 'iteration', 'learning')):
                    print(f"  {name}/{k}")

f.visititems(list_keys)

f.close()

# Also search for "secret" or "flag" as substrings in ANY attribute value
print("\n=== Searching for 'secret' or 'flag' in all attribute VALUES ===")
f = h5py.File('fashion_classifier (1).h5', 'r')

def search_in_values(name, obj):
    for key, val in obj.attrs.items():
        if isinstance(val, bytes):
            val = val.decode('utf-8', errors='replace')
        val_str = str(val).lower()
        if 'secret' in val_str or 'flag' in val_str:
            print(f"FOUND in {name}.{key}: {val}")

# File level
for key, val in f.attrs.items():
    if isinstance(val, bytes):
        val = val.decode('utf-8', errors='replace')
    val_str = str(val).lower()
    if 'secret' in val_str or 'flag' in val_str:
        print(f"FOUND at file level {key}: {val}")

f.visititems(search_in_values)
f.close()
