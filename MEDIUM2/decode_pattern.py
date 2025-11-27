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

# Fashion MNIST class mapping (short names)
short_names = ['T', 'R', 'P', 'D', 'C', 'S', 'H', 'N', 'B', 'A']
# 0=T-shirt, 1=tRouser, 2=Pullover, 3=Dress, 4=Coat, 5=Sandal, 6=sHirt, 7=sNeaker, 8=Bag, 9=Ankle

def fgsm_attack(model, image, label, epsilon=0.1):
    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    label_tensor = tf.convert_to_tensor([label], dtype=tf.int32)
    
    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        prediction = model(image_tensor)
        loss = keras.losses.sparse_categorical_crossentropy(label_tensor, prediction)
    
    gradient = tape.gradient(loss, image_tensor)
    signed_grad = tf.sign(gradient)
    adversarial_image = image_tensor + epsilon * signed_grad
    adversarial_image = tf.clip_by_value(adversarial_image, 0, 1)
    
    return adversarial_image.numpy()

# Get all images sorted by image number
image_files = sorted(glob.glob('images/*.png'))
images_data = []
for img_path in image_files:
    filename = os.path.basename(img_path)
    parts = filename.replace('.png', '').split('_')
    class_num = int(parts[1])
    img_num = int(parts[3])
    images_data.append({'file': filename, 'path': img_path, 'class': class_num, 'img': img_num})

# Sort by image number
images_data = sorted(images_data, key=lambda x: x['img'])

print("Testing FGSM attack on each image (sorted by img number):")
print("="*70)

eps = 0.1  # Standard epsilon

# Collect attack results
attack_results = []
for data in images_data:
    img = Image.open(data['path']).convert('L')
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    
    # Get original prediction
    orig_pred = model.predict(img_array, verbose=0)
    orig_class = np.argmax(orig_pred)
    
    # Perform attack
    adv_image = fgsm_attack(model, img_array, data['class'], epsilon=eps)
    adv_pred = model.predict(adv_image, verbose=0)
    adv_class = np.argmax(adv_pred)
    
    attack_results.append({
        'img_num': data['img'],
        'original': data['class'],
        'predicted': orig_class,
        'attacked': adv_class
    })
    
    success = "SUCCESS" if adv_class != orig_class else "FAIL"
    print(f"img{data['img']:02d}: class {data['class']} -> predicted {orig_class} -> attacked {adv_class} [{success}]")

print("\n" + "="*70)
print("Looking for patterns in attacked classes:\n")

# Get sequence of attacked classes
attacked_classes = [r['attacked'] for r in attack_results]
print(f"Attacked class sequence: {attacked_classes}")

# Try as direct numbers
print(f"As string: {''.join(map(str, attacked_classes))}")

# Original classes in img order
original_classes = [r['original'] for r in attack_results]
print(f"Original class sequence: {original_classes}")
print(f"As string: {''.join(map(str, original_classes))}")

# Img numbers
img_nums = [r['img_num'] for r in attack_results]
print(f"Img number sequence: {img_nums}")

# Maybe flag is encoded using class differences or XOR
diffs = [attacked_classes[i] - original_classes[i] for i in range(len(attacked_classes))]
print(f"Difference (attacked - original): {diffs}")

# XOR
xors = [attacked_classes[i] ^ original_classes[i] for i in range(len(attacked_classes))]
print(f"XOR (attacked ^ original): {xors}")

# Try as hex
try:
    hex_str = ''.join([f'{x:x}' for x in attacked_classes])
    print(f"Attacked as hex: {hex_str}")
except:
    pass

# Maybe combine img_num with attacked class
combined = []
for r in attack_results:
    combined.append(r['img_num'] * 10 + r['attacked'])
print(f"Combined (img*10 + attacked): {combined}")

# Try various ASCII interpretations
print("\n\nASCII interpretations:")

# img_num as ASCII
try:
    ascii1 = ''.join([chr(n + 65) for n in img_nums])  
    print(f"Img nums + 65 (A=0): {ascii1}")
except:
    pass

# original classes as ASCII
try:
    ascii2 = ''.join([chr(n + 65) for n in original_classes])
    print(f"Original classes + 65: {ascii2}")
except:
    pass

# Maybe it's TensorAI flag format
print("\n\nLooking for 'TensorAI{' pattern...")
# T=0, e=?, n=?, s=?, o=?, r=?, A=?, I=?
# Using class first letters: 0=T, 5=S, 9=A, etc

# Let's try to decode "TensorAI" using classes
tensor_chars = []
for c in original_classes:
    # Map class to letter based on class name first letter
    names = ['T', 'T', 'P', 'D', 'C', 'S', 'S', 'S', 'B', 'A']  # First letters
    tensor_chars.append(names[c])
print(f"Original classes as first letters: {''.join(tensor_chars)}")

tensor_chars2 = []
for c in attacked_classes:
    names = ['T', 'T', 'P', 'D', 'C', 'S', 'S', 'S', 'B', 'A']
    tensor_chars2.append(names[c])
print(f"Attacked classes as first letters: {''.join(tensor_chars2)}")
