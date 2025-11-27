import h5py

print("=== Complete H5 attribute dump ===")
print()

f = h5py.File('fashion_classifier (1).h5', 'r')

# File level
print("FILE LEVEL ATTRIBUTES:")
for key in f.attrs.keys():
    val = f.attrs[key]
    if isinstance(val, bytes):
        val = val.decode('utf-8')
    val_str = str(val)
    # Truncate long values but show first 100 chars
    if len(val_str) > 100:
        val_str = val_str[:100] + "..."
    print(f"  '{key}': {val_str}")

print()

# All groups and datasets
def dump_attrs(name, obj):
    if len(obj.attrs) > 0:
        print(f"\n{name}:")
        for key in obj.attrs.keys():
            val = obj.attrs[key]
            if isinstance(val, bytes):
                val = val.decode('utf-8')
            val_str = str(val)
            if len(val_str) > 100:
                val_str = val_str[:100] + "..."
            print(f"  '{key}': {val_str}")

f.visititems(dump_attrs)

# List ALL attribute names found anywhere
print("\n\n=== ALL UNIQUE ATTRIBUTE NAMES IN FILE ===")
all_attrs = set(f.attrs.keys())
def collect_attrs(name, obj):
    for key in obj.attrs.keys():
        all_attrs.add(key)
f.visititems(collect_attrs)
print(sorted(all_attrs))

f.close()
