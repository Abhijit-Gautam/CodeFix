import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from PIL import Image
import glob

# Load the model
model = keras.models.load_model('fashion_classifier (1).h5')

# Load class names
with open('class_names (1).txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# Get all images
image_files = sorted(glob.glob('images/*.png'))
print(f"Found {len(image_files)} images")

# Extract the image numbers from filenames
# Format: class_X_img_Y.png
print("\nFilename analysis:")
for img_path in image_files:
    filename = os.path.basename(img_path)
    parts = filename.replace('.png', '').split('_')
    class_num = int(parts[1])
    img_num = int(parts[3])
    print(f"  {filename}: class={class_num}, img={img_num}")

# Try to see if img_num forms ASCII
print("\n\nTrying different orderings to find flag:")

# Sort by class, then by img number
images_data = []
for img_path in image_files:
    filename = os.path.basename(img_path)
    parts = filename.replace('.png', '').split('_')
    class_num = int(parts[1])
    img_num = int(parts[3])
    images_data.append({'file': filename, 'class': class_num, 'img': img_num})

# Sort by class
by_class = sorted(images_data, key=lambda x: (x['class'], x['img']))
img_nums_by_class = [d['img'] for d in by_class]
print(f"Image nums sorted by class: {img_nums_by_class}")

# Try ASCII decode
try:
    # Maybe img numbers are ASCII offsets?
    ascii_str = ''.join([chr(n + 65) for n in img_nums_by_class])  # A=0
    print(f"As ASCII (A=0): {ascii_str}")
    ascii_str2 = ''.join([chr(n + 97) for n in img_nums_by_class])  # a=0
    print(f"As ASCII (a=0): {ascii_str2}")
except:
    pass

# Maybe the class numbers spell out something with specific eps attacks
print("\n\nClass pattern analysis:")
classes = [d['class'] for d in by_class]
print(f"Classes in order: {classes}")

# Maybe img_num is direct ASCII?
try:
    direct_ascii = ''.join([chr(d['img']) for d in by_class if d['img'] >= 32])
    print(f"Img nums as direct ASCII (printable only): {direct_ascii}")
except:
    pass

# Let me check if there are exactly certain images that together form something
print("\n\nImg numbers in original order:")
img_nums = [d['img'] for d in images_data]
print(f"Img nums: {img_nums}")

# Sort by image number
by_img = sorted(images_data, key=lambda x: x['img'])
classes_by_img = [d['class'] for d in by_img]
print(f"Classes sorted by img num: {classes_by_img}")

# Maybe classes form ASCII when sorted by img
try:
    ascii_from_class = ''.join([chr(c + 65) for c in classes_by_img])
    print(f"Classes as ASCII (A=0): {ascii_from_class}")
except:
    pass

# Look for flag pattern in combinations
# Maybe first letter of class name + something
print("\n\nFirst letters of class names by img number order:")
class_names_clean = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
first_letters = ''.join([class_names_clean[c][0] for c in classes_by_img])
print(f"First letters: {first_letters}")

# Try combining class number and image number
print("\n\nCombined patterns:")
for d in sorted(images_data, key=lambda x: x['img']):
    print(f"  img{d['img']:02d}: class {d['class']} ({class_names_clean[d['class']]})")
