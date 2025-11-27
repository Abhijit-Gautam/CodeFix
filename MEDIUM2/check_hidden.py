import h5py

f = h5py.File('fashion_classifier (1).h5', 'r')

# Check top_level_model_weights specifically
print("=== Checking top_level_model_weights ===")
if 'model_weights/top_level_model_weights' in f:
    tlmw = f['model_weights/top_level_model_weights']
    print(f"Type: {type(tlmw)}")
    print(f"Attributes: {list(tlmw.attrs.keys())}")
    for key in tlmw.attrs.keys():
        val = tlmw.attrs[key]
        if isinstance(val, bytes):
            print(f"  {key}: {val.decode('utf-8')}")
        else:
            print(f"  {key}: {val}")
    
    # If it's a group, list contents
    if hasattr(tlmw, 'keys'):
        print(f"Contents: {list(tlmw.keys())}")
        for item in tlmw.keys():
            print(f"  - {item}")

# Check ALL groups and their attributes more carefully
print("\n\n=== ALL group attributes ===")
def check_all_attrs(name, obj):
    if len(obj.attrs) > 0:
        print(f"\n{name}:")
        for key in obj.attrs.keys():
            val = obj.attrs[key]
            if isinstance(val, bytes):
                val_str = val.decode('utf-8')
            else:
                val_str = str(val)
            # Only print if not empty
            if val_str.strip():
                print(f"  {key}: {val_str[:200]}")

f.visititems(check_all_attrs)

# Check if there's any hidden data in dataset descriptions
print("\n\n=== Dataset descriptions ===")
def check_datasets(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"{name}: dtype={obj.dtype}, shape={obj.shape}")
        # Check for any non-standard attributes
        for attr in obj.attrs:
            if attr not in ['DIMENSION_LIST']:  # Standard HDF5 attrs
                print(f"  Custom attr: {attr} = {obj.attrs[attr]}")

f.visititems(check_datasets)

f.close()

# Also let's look at what keys exist at each level
print("\n\n=== Exploring h5 structure ===")
f = h5py.File('fashion_classifier (1).h5', 'r')
print(f"Root keys: {list(f.keys())}")
print(f"Root attrs: {list(f.attrs.keys())}")

# Check model_weights
mw = f['model_weights']
print(f"\nmodel_weights keys: {list(mw.keys())}")
print(f"model_weights attrs: {list(mw.attrs.keys())}")

# Check optimizer_weights
ow = f['optimizer_weights']
print(f"\noptimizer_weights keys: {list(ow.keys())}")
print(f"optimizer_weights attrs: {list(ow.attrs.keys())}")

# Check adam
adam = ow['adam']
print(f"\nadam keys: {list(adam.keys())}")
print(f"adam attrs: {list(adam.attrs.keys())}")

f.close()
