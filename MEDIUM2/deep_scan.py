import h5py
import struct

f = h5py.File('fashion_classifier (1).h5', 'r')

# Try to get ALL object names including soft/hard links
print("=== All objects including linked ===")
def visit_all(name):
    print(name)

f.visit(visit_all)

# Check if there are any soft links or external links
print("\n=== Checking for soft/external links ===")
def check_links(name, obj):
    link = f.get(name, getlink=True)
    if link is not None:
        print(f"{name}: link type = {type(link)}")

f.visititems(check_links)

f.close()

# Now let's check the raw file structure more carefully
# HDF5 files have a superblock at the start
print("\n=== HDF5 Superblock and File Structure ===")
with open('fashion_classifier (1).h5', 'rb') as raw:
    # Read first 8 bytes (HDF5 signature)
    signature = raw.read(8)
    print(f"Signature: {signature}")
    
    # Read superblock
    raw.seek(0)
    header = raw.read(1024)
    
    # Look for any readable strings in the header
    strings_in_header = []
    current = b''
    for b in header:
        if 32 <= b <= 126:
            current += bytes([b])
        else:
            if len(current) > 3:
                strings_in_header.append(current.decode('ascii'))
            current = b''
    print(f"Strings in header: {strings_in_header}")

# Check if there's anything at the very end of the file
print("\n=== End of file ===")
with open('fashion_classifier (1).h5', 'rb') as raw:
    raw.seek(-2000, 2)  # Last 2000 bytes
    tail = raw.read()
    
    # Look for strings
    strings_in_tail = []
    current = b''
    for b in tail:
        if 32 <= b <= 126:
            current += bytes([b])
        else:
            if len(current) > 3:
                strings_in_tail.append(current.decode('ascii'))
            current = b''
    
    print(f"Strings in last 2000 bytes:")
    for s in strings_in_tail:
        if not any(x in s.lower() for x in ['tensorflow', 'keras', 'conv', 'batch', 'dense', 'adam', 'sequential']):
            print(f"  '{s}'")
