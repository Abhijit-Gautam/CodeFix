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

print(f"Class names: {class_names}")

# FGSM Attack function
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
image_files = glob.glob('images/*.png')
print(f"\nFound {len(image_files)} images to attack")

# Try different epsilon values
epsilons = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5]

results = []

for img_path in image_files:
    # Extract original class from filename (e.g., class_6_img_7.png -> 6)
    filename = os.path.basename(img_path)
    original_class = int(filename.split('_')[1])
    
    # Load and preprocess image
    img = Image.open(img_path).convert('L')  # Grayscale
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    
    # Get original prediction
    orig_pred = model.predict(img_array, verbose=0)
    orig_class = np.argmax(orig_pred)
    orig_conf = np.max(orig_pred)
    
    print(f"\n{'='*60}")
    print(f"Image: {filename}")
    print(f"Original class (from filename): {original_class} ({class_names[original_class]})")
    print(f"Model prediction: {orig_class} ({class_names[orig_class]}) with {orig_conf:.4f} confidence")
    
    # Try FGSM attack with different epsilons
    for eps in epsilons:
        adv_image = fgsm_attack(model, img_array, original_class, epsilon=eps)
        adv_pred = model.predict(adv_image, verbose=0)
        adv_class = np.argmax(adv_pred)
        adv_conf = np.max(adv_pred)
        
        if adv_class != orig_class:
            print(f"  [eps={eps}] Attack SUCCESS! {orig_class} -> {adv_class} ({class_names[adv_class]}) conf={adv_conf:.4f}")
            results.append({
                'image': filename,
                'epsilon': eps,
                'from_class': orig_class,
                'to_class': adv_class,
                'from_name': class_names[orig_class],
                'to_name': class_names[adv_class],
                'confidence': adv_conf
            })

print("\n" + "="*60)
print("SUMMARY OF SUCCESSFUL ATTACKS:")
print("="*60)
for r in results:
    print(f"{r['image']}: eps={r['epsilon']}, {r['from_name']} -> {r['to_name']} (conf={r['confidence']:.4f})")

# Check if any class names or attack patterns reveal a flag
print("\n\nLooking for flag patterns...")
# Check if first letters of attacked classes spell something
if results:
    # Group by image and get the attack result
    attack_results_by_image = {}
    for r in results:
        if r['image'] not in attack_results_by_image:
            attack_results_by_image[r['image']] = r
    
    # Sort by image name and get to_name first letters
    sorted_images = sorted(attack_results_by_image.keys())
    first_letters = ''.join([attack_results_by_image[img]['to_name'][0] for img in sorted_images])
    print(f"First letters of attack results: {first_letters}")
    
    # Try to decode as flag
    to_classes = [attack_results_by_image[img]['to_class'] for img in sorted_images]
    print(f"Target classes: {to_classes}")
    
    # Try ASCII decode
    try:
        ascii_chars = ''.join([chr(c + ord('A')) for c in to_classes])
        print(f"Classes as letters (A=0): {ascii_chars}")
    except:
        pass
