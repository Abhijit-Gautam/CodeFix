import h5py

f = h5py.File('fashion_classifier (1).h5', 'r')

print('=== COMPLETE DUMP OF ALL ATTRIBUTES ===\n')

# File-level attributes
print('--- FILE LEVEL ATTRS ---')
for key, value in f.attrs.items():
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    print(f'[FILE].{key} = {value[:500] if len(str(value)) > 500 else value}')
    print()

print('\n--- ALL GROUP/DATASET ATTRS ---')
def dump_all(name, obj):
    for key, value in obj.attrs.items():
        if isinstance(value, bytes):
            value = value.decode('utf-8')
        elif hasattr(value, 'tobytes'):
            # numpy array - convert to string
            try:
                value = value.astype(str).tolist()
            except:
                value = str(value)
        print(f'[{name}].{key} = {value}')
    print()

f.visititems(dump_all)

# Also look for any hidden/custom datasets that might contain text
print('\n--- CHECKING ALL DATASETS FOR STRING DATA ---')
def check_datasets(name, obj):
    if isinstance(obj, h5py.Dataset):
        try:
            data = obj[()]
            if data.dtype.kind in ['S', 'O', 'U']:  # String types
                print(f'[{name}] STRING DATA: {data}')
        except:
            pass

f.visititems(check_datasets)

# Check for any groups/datasets with suspicious names
print('\n--- ALL GROUP/DATASET NAMES ---')
def list_all(name, obj):
    print(name)

f.visititems(list_all)

f.close()
