import h5py

f = h5py.File('fashion_classifier (1).h5', 'r')

print('=== File-level Attributes ===')
for key, value in f.attrs.items():
    print(f'{key}: {value}')

print('\n=== All Groups and Datasets with Attributes ===')
def print_attrs(name, obj):
    if len(obj.attrs) > 0:
        print(f'\n{name}:')
        for key, value in obj.attrs.items():
            if isinstance(value, bytes):
                value = value.decode('utf-8')
            print(f'  {key}: {value}')

f.visititems(print_attrs)

print('\n=== Looking for flag in all attributes ===')
def find_flag(name, obj):
    for key, value in obj.attrs.items():
        if isinstance(value, bytes):
            value = value.decode('utf-8')
        val_str = str(value)
        if 'flag' in val_str.lower() or 'ctf' in val_str.lower() or 'tensor' in val_str.lower():
            print(f'Found in {name}.{key}: {value}')

f.visititems(find_flag)

# Also check file-level attrs
for key, value in f.attrs.items():
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    val_str = str(value)
    if 'flag' in val_str.lower() or 'ctf' in val_str.lower() or 'tensor' in val_str.lower():
        print(f'Found in file.{key}: {value}')

f.close()
