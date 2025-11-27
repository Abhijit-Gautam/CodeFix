"""
ADVERSARIAL ATTACK CHALLENGE - MEDIUM2 Solution
================================================

FGSM Attack to misclassify Fashion MNIST images.
Goal: Fool the model to classify images as wrong class (e.g., T-shirt -> Trouser)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import h5py
import os
from PIL import Image
import glob
import json

# Load the model
print("Loading the fashion classifier model...")
model = keras.models.load_model('fashion_classifier (1).h5')

# Fashion MNIST class names
CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def fgsm_attack(model, image, true_label, epsilon=0.3):
    """
    Fast Gradient Sign Method (FGSM) Attack
    Perturbs the image to maximize loss, causing misclassification
    """
    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    label_tensor = tf.convert_to_tensor([true_label], dtype=tf.int32)
    
    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        prediction = model(image_tensor)
        loss = keras.losses.sparse_categorical_crossentropy(label_tensor, prediction)
    
    # Get gradient and create adversarial perturbation
    gradient = tape.gradient(loss, image_tensor)
    perturbation = epsilon * tf.sign(gradient)
    
    # Create adversarial image
    adversarial_image = image_tensor + perturbation
    adversarial_image = tf.clip_by_value(adversarial_image, 0, 1)
    
    return adversarial_image.numpy()


def targeted_fgsm_attack(model, image, target_class, epsilon=0.3, num_iter=50):
    """
    Targeted FGSM Attack - Force misclassification to a specific target class
    E.g., Make T-shirt classify as Trouser
    """
    adv_image = tf.convert_to_tensor(image, dtype=tf.float32)
    target_tensor = tf.convert_to_tensor([target_class], dtype=tf.int32)
    alpha = epsilon / num_iter
    
    for _ in range(num_iter):
        with tf.GradientTape() as tape:
            tape.watch(adv_image)
            prediction = model(adv_image)
            # Minimize loss for target class (maximize probability of target)
            loss = keras.losses.sparse_categorical_crossentropy(target_tensor, prediction)
        
        gradient = tape.gradient(loss, adv_image)
        # Subtract gradient to minimize loss for target class
        adv_image = adv_image - alpha * tf.sign(gradient)
        adv_image = tf.clip_by_value(adv_image, 0, 1)
        
        # Check if attack succeeded
        current_pred = int(tf.argmax(model(adv_image)[0]))
        if current_pred == target_class:
            break
    
    return adv_image.numpy()


def pgd_attack(model, image, true_label, epsilon=0.3, alpha=0.01, num_iter=100):
    """
    Projected Gradient Descent (PGD) Attack - Stronger iterative attack
    """
    adv_image = tf.convert_to_tensor(image, dtype=tf.float32)
    original_image = tf.identity(adv_image)
    label_tensor = tf.convert_to_tensor([true_label], dtype=tf.int32)
    
    for _ in range(num_iter):
        with tf.GradientTape() as tape:
            tape.watch(adv_image)
            prediction = model(adv_image)
            loss = keras.losses.sparse_categorical_crossentropy(label_tensor, prediction)
        
        gradient = tape.gradient(loss, adv_image)
        adv_image = adv_image + alpha * tf.sign(gradient)
        
        # Project back to epsilon ball
        perturbation = tf.clip_by_value(adv_image - original_image, -epsilon, epsilon)
        adv_image = tf.clip_by_value(original_image + perturbation, 0, 1)
    
    return adv_image.numpy()


def get_secret_flag(model_path, success_rate):
    """
    Extract flag from h5 file if attack is successful (>90%)
    """
    if success_rate <= 90:
        return None
    
    with h5py.File(model_path, 'r') as f:
        # Search for secret/flag attribute
        for key in f.attrs.keys():
            key_lower = key.lower()
            if 'secret' in key_lower or 'flag' in key_lower:
                val = f.attrs[key]
                return val.decode('utf-8') if isinstance(val, bytes) else str(val)
        
        # Search in all groups
        found_flag = [None]
        def search(name, obj):
            for attr_key in obj.attrs.keys():
                if 'secret' in attr_key.lower() or 'flag' in attr_key.lower():
                    val = obj.attrs[attr_key]
                    found_flag[0] = val.decode('utf-8') if isinstance(val, bytes) else str(val)
        
        f.visititems(search)
        return found_flag[0]


def main():
    print("\n" + "="*70)
    print("ADVERSARIAL ATTACK CHALLENGE")
    print("="*70)
    
    # Extract images if needed
    image_files = sorted(glob.glob('images/*.png'))
    if not image_files:
        import zipfile
        with zipfile.ZipFile('images-20251126T134926Z-1-001.zip', 'r') as z:
            z.extractall('.')
        image_files = sorted(glob.glob('images/*.png'))
    
    print(f"\nFound {len(image_files)} test images")
    print("\nPerforming FGSM Adversarial Attack...")
    print("-"*70)
    
    successful_attacks = 0
    total_attacks = len(image_files)
    results = []
    
    for img_path in image_files:
        filename = os.path.basename(img_path)
        # Extract true class from filename: class_X_img_Y.png
        true_class = int(filename.split('_')[1])
        
        # Load and preprocess image
        img = Image.open(img_path).convert('L')
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)
        
        # Get original prediction
        orig_pred = model.predict(img_array, verbose=0)
        orig_class = int(np.argmax(orig_pred))
        orig_conf = float(np.max(orig_pred))
        
        # Choose target class (different from original)
        # E.g., T-shirt(0) -> Trouser(1), Trouser(1) -> Dress(3), etc.
        target_class = (orig_class + 1) % 10
        
        # Try multiple attack strategies
        attack_success = False
        adv_class = orig_class
        adv_conf = orig_conf
        
        # Strategy 1: PGD Attack (strongest)
        adv_image = pgd_attack(model, img_array, true_class, epsilon=0.4, alpha=0.01, num_iter=100)
        adv_pred = model.predict(adv_image, verbose=0)
        adv_class = int(np.argmax(adv_pred))
        adv_conf = float(np.max(adv_pred))
        
        if adv_class != orig_class:
            attack_success = True
        else:
            # Strategy 2: Targeted FGSM to specific class
            adv_image = targeted_fgsm_attack(model, img_array, target_class, epsilon=0.5, num_iter=100)
            adv_pred = model.predict(adv_image, verbose=0)
            adv_class = int(np.argmax(adv_pred))
            adv_conf = float(np.max(adv_pred))
            
            if adv_class != orig_class:
                attack_success = True
            else:
                # Strategy 3: Basic FGSM with high epsilon
                for eps in [0.3, 0.4, 0.5, 0.6]:
                    adv_image = fgsm_attack(model, img_array, true_class, epsilon=eps)
                    adv_pred = model.predict(adv_image, verbose=0)
                    adv_class = int(np.argmax(adv_pred))
                    adv_conf = float(np.max(adv_pred))
                    
                    if adv_class != orig_class:
                        attack_success = True
                        break
        
        if attack_success:
            successful_attacks += 1
            status = "SUCCESS"
        else:
            status = "FAILED"
        
        print(f"{filename}: {CLASS_NAMES[orig_class]:12s} -> {CLASS_NAMES[adv_class]:12s} [{status}]")
        
        results.append({
            'image': filename,
            'original_class': CLASS_NAMES[orig_class],
            'adversarial_class': CLASS_NAMES[adv_class],
            'success': attack_success
        })
    
    # Calculate success rate
    success_rate = (successful_attacks / total_attacks) * 100
    
    print("\n" + "="*70)
    print("ATTACK SUMMARY")
    print("="*70)
    print(f"Total Images: {total_attacks}")
    print(f"Successful Attacks: {successful_attacks}")
    print(f"Success Rate: {success_rate:.2f}%")
    
    # Check if we achieved > 90% to get the flag
    if success_rate > 90:
        print("\n" + "="*70)
        print("ATTACK SUCCESSFUL! (>90%)")
        print("="*70)
        
        # Try to get flag from model file
        flag = get_secret_flag('fashion_classifier (1).h5', success_rate)
        if flag:
            print(f"\nFLAG: {flag}")
        else:
            print("\nAttack successful! Submit for verification.")
        
        # Save results
        output = {
            'success_rate': success_rate,
            'successful_attacks': successful_attacks,
            'total_attacks': total_attacks,
            'results': results
        }
        with open('attack_results.json', 'w') as f:
            json.dump(output, f, indent=2)
        print("\nResults saved to attack_results.json")
    else:
        print(f"\nNeed >90% success rate. Current: {success_rate:.2f}%")


if __name__ == "__main__":
    main()
