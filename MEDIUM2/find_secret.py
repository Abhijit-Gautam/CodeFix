import h5py

f = h5py.File('fashion_classifier (1).h5', 'r')

print("=== Searching for 'secret' or 'flag' attributes ===\n")

# Check file-level attributes
print("FILE-LEVEL ATTRIBUTES:")
for key in f.attrs.keys():
    print(f"  {key}")
    if 'secret' in key.lower() or 'flag' in key.lower():
        print(f"    *** FOUND: {f.attrs[key]}")

# Check all groups and datasets
def search_attrs(name, obj):
    for key in obj.attrs.keys():
        if 'secret' in key.lower() or 'flag' in key.lower():
            val = obj.attrs[key]
            if isinstance(val, bytes):
                val = val.decode('utf-8')
            print(f"FOUND in {name}: {key} = {val}")

f.visititems(search_attrs)

# Also list ALL attribute keys to see if there's anything unusual
print("\n=== ALL UNIQUE ATTRIBUTE KEYS ===")
all_keys = set(f.attrs.keys())

def collect_keys(name, obj):
    for key in obj.attrs.keys():
        all_keys.add(key)

f.visititems(collect_keys)

for key in sorted(all_keys):
    print(f"  {key}")

# Check for any datasets that might contain a flag string
print("\n=== Checking for string/object datasets ===")
def check_datasets(name, obj):
    if isinstance(obj, h5py.Dataset):
        # Check if it could contain string data
        if obj.size < 100:  # Small datasets might be flags
            try:
                data = obj[()]
                print(f"{name}: shape={obj.shape}, dtype={obj.dtype}, data={data}")
            except:
                pass

f.visititems(check_datasets)

f.close()
