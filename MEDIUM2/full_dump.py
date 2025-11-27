import h5py

f = h5py.File('fashion_classifier (1).h5', 'r')

print("=== Complete dump of ALL attributes at every level ===\n")

# First, list ALL attributes at file level
print("FILE LEVEL ATTRS:")
for k in f.attrs.keys():
    v = f.attrs[k]
    print(f"  Key: '{k}'")
    print(f"  Type: {type(v)}")
    if isinstance(v, str):
        print(f"  Length: {len(v)}")
    print()

# Now go through ALL objects and list their attrs
print("\n=== ALL OBJECT ATTRIBUTES ===")
count = 0
def list_all_attrs(name, obj):
    global count
    if len(obj.attrs) > 0:
        count += 1
        print(f"\n[{name}]")
        for k in obj.attrs.keys():
            v = obj.attrs[k]
            print(f"  {k}: ", end="")
            if isinstance(v, bytes):
                print(v.decode('utf-8', errors='replace')[:100])
            elif hasattr(v, '__len__') and len(v) > 10:
                print(f"[array of {len(v)} items]")
            else:
                print(v)

f.visititems(list_all_attrs)
print(f"\nTotal objects with attrs: {count}")

# Try accessing by index
print("\n=== Trying to access items by numeric index ===")
try:
    for i in range(10):
        print(f"f[{i}]: ", end="")
        try:
            print(f[str(i)])
        except:
            print("not found")
except Exception as e:
    print(f"Error: {e}")

# List all keys at all levels
print("\n=== Complete key hierarchy ===")
def list_all_keys(name, obj):
    print(name)

f.visititems(list_all_keys)

f.close()

# Now try reading the file with low-level h5py access
print("\n\n=== Low-level file inspection ===")
import h5py.h5f as h5f
import h5py.h5g as h5g
import h5py.h5a as h5a

fid = h5py.h5f.open(b'fashion_classifier (1).h5')
root = h5py.h5g.open(fid, b'/')

# Get number of attrs at root
n_attrs = h5a.get_num_attrs(root)
print(f"Number of root attrs: {n_attrs}")

# List them
for i in range(n_attrs):
    aid = h5a.open(root, index=i)
    name = h5a.get_name(aid)
    print(f"  Attr {i}: {name}")
