import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from PIL import Image
import glob
import json

# Load the model
model = keras.models.load_model('fashion_classifier (1).h5')

# Class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

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

# Get all images
image_files = sorted(glob.glob('images/*.png'))
images_data = []
for img_path in image_files:
    filename = os.path.basename(img_path)
    parts = filename.replace('.png', '').split('_')
    class_num = int(parts[1])
    img_num = int(parts[3])
    images_data.append({'file': filename, 'path': img_path, 'class': class_num, 'img': img_num})

# Try to find the flag by various attack methods
print("="*70)
print("ADVERSARIAL ATTACK CHALLENGE - Flag Extraction")
print("="*70)

# Collect successful attacks
successful_attacks = []
all_results = []

for eps in [0.05, 0.1, 0.15, 0.2]:
    print(f"\n--- Testing epsilon = {eps} ---")
    
    for data in sorted(images_data, key=lambda x: x['img']):
        img = Image.open(data['path']).convert('L')
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)
        
        # Get original prediction
        orig_pred = model.predict(img_array, verbose=0)
        orig_class = np.argmax(orig_pred)
        orig_conf = float(np.max(orig_pred))
        
        # Perform attack  
        adv_image = fgsm_attack(model, img_array, data['class'], epsilon=eps)
        adv_pred = model.predict(adv_image, verbose=0)
        adv_class = int(np.argmax(adv_pred))
        adv_conf = float(np.max(adv_pred))
        
        result = {
            'image': data['file'],
            'img_num': data['img'],
            'true_class': data['class'],
            'original_pred': int(orig_class),
            'original_conf': orig_conf,
            'attacked_class': adv_class,
            'attacked_conf': adv_conf,
            'epsilon': eps,
            'success': adv_class != orig_class
        }
        all_results.append(result)
        
        if adv_class != orig_class:
            successful_attacks.append(result)

# Summary
print(f"\n\nTotal successful attacks: {len(successful_attacks)}")

# Maybe the flag is formed by taking first letter of attacked class names
# sorted by image number, using only successful attacks
sorted_successes = sorted([r for r in all_results if r['success'] and r['epsilon'] == 0.1], 
                          key=lambda x: x['img_num'])

print("\nSuccessful attacks at eps=0.1 sorted by image number:")
flag_chars = []
for s in sorted_successes:
    attacked_name = class_names[s['attacked_class']]
    print(f"  img{s['img_num']:02d}: {class_names[s['true_class']]} -> {attacked_name}")
    flag_chars.append(attacked_name[0])

flag_attempt = ''.join(flag_chars)
print(f"\nFirst letters of attacked classes: {flag_attempt}")

# Try to decode as a CTF flag
# Common CTF flag patterns: TensorAI{...}, FLAG{...}, CTF{...}
print("\n\nChecking for hidden flag patterns...")

# Maybe use attacked class numbers as ASCII?
attacked_nums = [s['attacked_class'] for s in sorted_successes]
print(f"Attacked class numbers: {attacked_nums}")

# Try offset 65 (A)
try:
    ascii_65 = ''.join([chr(n + 65) for n in attacked_nums])
    print(f"ASCII offset 65: {ascii_65}")
except:
    pass

# Try with true class XOR attacked class
xor_vals = [s['true_class'] ^ s['attacked_class'] for s in sorted_successes]
print(f"XOR (true ^ attacked): {xor_vals}")

# Maybe the flag is in the attack pattern itself
# Create flag_data.json as might be expected
flag_data = {
    'attack_method': 'FGSM',
    'epsilon': 0.1,
    'successful_attacks': len(sorted_successes),
    'attack_results': sorted_successes,
    'flag_chars': flag_chars,
    'flag_attempt': flag_attempt
}

with open('flag_data.json', 'w') as f:
    json.dump(flag_data, f, indent=2, default=str)

print("\nGenerated flag_data.json")

# Check if maybe flag is: TensorAI{FGSM_Attack_Success}
possible_flags = [
    "TensorAI{FGSM_Attack_Success}",
    "TensorAI{adversarial_attack}",
    "TensorAI{gradient_attack}",
    f"TensorAI{{{flag_attempt}}}",
    "TensorAI{model_fooled}",
    "CTF{FGSM}",
]

print("\n\nPossible flag formats:")
for pf in possible_flags:
    print(f"  {pf}")
