#!/usr/bin/env python3
"""
Fashion MNIST Image Classifier - AI CODEFIX 2025
Uses a pre-trained model to classify fashion images.
"""

import os
import sys
import numpy as np
from PIL import Image
import argparse

# Try to import tensorflow/keras
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Error: TensorFlow not installed. Please install with: pip install tensorflow")
    sys.exit(1)

# Class names for Fashion MNIST
CLASS_NAMES = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}


def load_model(model_path):
    """Load the pre-trained Keras model."""
    try:
        model = keras.models.load_model(model_path)
        print(f"✓ Model loaded from: {model_path}")
        return model
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None


def preprocess_image(image_path):
    """
    Preprocess an image for Fashion MNIST classification.
    Fashion MNIST images are 28x28 grayscale images.
    """
    try:
        # Load image
        img = Image.open(image_path)
        
        # Convert to grayscale if needed
        if img.mode != 'L':
            img = img.convert('L')
        
        # Resize to 28x28 (Fashion MNIST size)
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Normalize to [0, 1]
        img_array = img_array.astype('float32') / 255.0
        
        # Reshape for model input (batch_size, height, width, channels)
        # Some models expect (28, 28, 1), others expect (28, 28)
        img_array = img_array.reshape(1, 28, 28, 1)
        
        return img_array
    except Exception as e:
        print(f"✗ Error preprocessing image {image_path}: {e}")
        return None


def classify_image(model, image_path):
    """Classify a single image."""
    # Preprocess
    img_array = preprocess_image(image_path)
    if img_array is None:
        return None, None
    
    # Predict
    predictions = model.predict(img_array, verbose=0)
    
    # Get class and confidence
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    return predicted_class, confidence


def classify_directory(model, directory_path):
    """Classify all images in a directory."""
    results = []
    
    # Get all image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif']
    image_files = [f for f in os.listdir(directory_path) 
                   if os.path.splitext(f)[1].lower() in image_extensions]
    
    print(f"\nClassifying {len(image_files)} images...\n")
    print("-" * 70)
    print(f"{'Image':<35} {'Predicted Class':<20} {'Confidence':<15}")
    print("-" * 70)
    
    correct = 0
    total = 0
    
    for image_file in sorted(image_files):
        image_path = os.path.join(directory_path, image_file)
        predicted_class, confidence = classify_image(model, image_path)
        
        if predicted_class is not None:
            class_name = CLASS_NAMES.get(predicted_class, "Unknown")
            
            # Try to extract true class from filename (e.g., class_0_img_19.png)
            true_class = None
            if "class_" in image_file:
                try:
                    true_class = int(image_file.split("class_")[1].split("_")[0])
                except:
                    pass
            
            # Check if prediction is correct
            is_correct = ""
            if true_class is not None:
                total += 1
                if true_class == predicted_class:
                    correct += 1
                    is_correct = "✓"
                else:
                    is_correct = f"✗ (True: {CLASS_NAMES.get(true_class, 'Unknown')})"
            
            print(f"{image_file:<35} {class_name:<20} {confidence*100:.2f}% {is_correct}")
            
            results.append({
                'image': image_file,
                'predicted_class': predicted_class,
                'class_name': class_name,
                'confidence': confidence,
                'true_class': true_class
            })
    
    print("-" * 70)
    
    if total > 0:
        accuracy = correct / total * 100
        print(f"\nAccuracy: {correct}/{total} = {accuracy:.2f}%")
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Fashion MNIST Image Classifier - AI CODEFIX 2025')
    parser.add_argument('--model', default='fashion_classifier (1).h5', 
                        help='Path to the pre-trained model file')
    parser.add_argument('--image', help='Path to a single image to classify')
    parser.add_argument('--directory', default='images', 
                        help='Path to directory containing images to classify')
    args = parser.parse_args()
    
    print("=" * 60)
    print("  FASHION MNIST IMAGE CLASSIFIER - AI CODEFIX 2025")
    print("=" * 60)
    print()
    
    # Load model
    print("Loading model...")
    model = load_model(args.model)
    if model is None:
        sys.exit(1)
    
    # Print model summary
    print("\nModel Summary:")
    model.summary()
    print()
    
    # Classify
    if args.image:
        # Single image
        print(f"\nClassifying single image: {args.image}")
        predicted_class, confidence = classify_image(model, args.image)
        if predicted_class is not None:
            class_name = CLASS_NAMES.get(predicted_class, "Unknown")
            print(f"\nResult: {class_name} ({confidence*100:.2f}% confidence)")
    elif os.path.isdir(args.directory):
        # Directory of images
        classify_directory(model, args.directory)
    else:
        print(f"✗ Directory not found: {args.directory}")
        print("Use --image to classify a single image, or --directory for a folder of images")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("  CLASSIFICATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
