import h5py

f = h5py.File('fashion_classifier (1).h5', 'r')

# Print ALL file-level attributes with their full values
print("=== File-level attributes (Full dump) ===")
for key in f.attrs.keys():
    val = f.attrs[key]
    if isinstance(val, bytes):
        val_str = val.decode('utf-8')
    else:
        val_str = str(val)
    print(f"\n--- {key} ---")
    print(val_str[:5000] if len(val_str) > 5000 else val_str)

# Look for any attribute containing 'flag' or 'secret' in its VALUE
print("\n\n=== Searching for 'flag', 'secret', 'CTF' in attribute values ===")
for key in f.attrs.keys():
    val = f.attrs[key]
    if isinstance(val, bytes):
        val_str = val.decode('utf-8').lower()
    else:
        val_str = str(val).lower()
    
    for search_term in ['flag', 'secret', 'ctf', 'tensor', 'hidden', 'challenge']:
        if search_term in val_str:
            print(f"Found '{search_term}' in attribute '{key}':")
            if isinstance(f.attrs[key], bytes):
                print(f.attrs[key].decode('utf-8'))
            else:
                print(f.attrs[key])

# Maybe it's in the model_config or training_config JSON
import json

print("\n\n=== Parsing model_config ===")
model_config = f.attrs.get('model_config')
if model_config is not None:
    if isinstance(model_config, bytes):
        model_config = model_config.decode('utf-8')
    try:
        config_dict = json.loads(model_config)
        print(json.dumps(config_dict, indent=2))
        
        # Search recursively for any key containing flag/secret
        def find_key(d, search):
            if isinstance(d, dict):
                for k, v in d.items():
                    if search in k.lower():
                        print(f"FOUND KEY '{k}': {v}")
                    find_key(v, search)
            elif isinstance(d, list):
                for item in d:
                    find_key(item, search)
        
        print("\nSearching model_config for 'flag':")
        find_key(config_dict, 'flag')
        print("\nSearching model_config for 'secret':")
        find_key(config_dict, 'secret')
    except:
        print(model_config)

print("\n\n=== Parsing training_config ===")
training_config = f.attrs.get('training_config')
if training_config is not None:
    if isinstance(training_config, bytes):
        training_config = training_config.decode('utf-8')
    try:
        config_dict = json.loads(training_config)
        print(json.dumps(config_dict, indent=2))
        
        def find_key(d, search):
            if isinstance(d, dict):
                for k, v in d.items():
                    if search in k.lower():
                        print(f"FOUND KEY '{k}': {v}")
                    find_key(v, search)
            elif isinstance(d, list):
                for item in d:
                    find_key(item, search)
        
        print("\nSearching training_config for 'flag':")
        find_key(config_dict, 'flag')
        print("\nSearching training_config for 'secret':")
        find_key(config_dict, 'secret')
    except:
        print(training_config)

f.close()
