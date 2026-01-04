"""
Image Processing Utilities for Face Recognition
utils/image_processing.py

Based on FaceNet implementation
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
from PIL import Image

# Set to channels_first format (required for the Inception model built from scratch)
K.set_image_data_format('channels_first')


def img_to_encoding(image_path, model):
    """
    Convert an image to its 128-dimensional encoding using FaceNet model.
    
    The Inception model (built from inception_blocks_v2) expects 96x96x3 images
    in channels-first format (3, 96, 96).
    
    Args:
        image_path: Path to the image file
        model: Pre-trained FaceNet model
        
    Returns:
        Normalized encoding vector of shape (1, 128)
        The encoding is L2 normalized to unit length.
    """
    # Load and resize image to 96x96 (required input size for the model)
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(96, 96))
    
    # Convert to numpy array and normalize to [0, 1]
    # Round to 12 decimals to match the notebook implementation
    img_array = np.around(np.array(img) / 255.0, decimals=12)
    
    # Convert to channels-first format (3, 96, 96)
    # PIL loads as (96, 96, 3) so we need to transpose
    if img_array.ndim == 3 and img_array.shape[2] == 3:
        img_array = np.transpose(img_array, (2, 0, 1))
    
    # Add batch dimension: (3, 96, 96) -> (1, 3, 96, 96)
    x_train = np.expand_dims(img_array, axis=0)
    
    # Get the embedding from the model
    embedding = model.predict_on_batch(x_train)
    
    # L2 normalize the embedding to unit length
    # This ensures embeddings are on the unit hypersphere
    return embedding / np.linalg.norm(embedding, ord=2)


def load_and_preprocess_image(image_path, target_size=(96, 96)):
    """
    Load and preprocess an image for display or further processing
    
    Args:
        image_path: Path to the image
        target_size: Target size for the image (default: 96x96)
        
    Returns:
        Preprocessed image array
    """
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = np.array(img)
    return img_array


def visualize_image(image_path):
    """
    Load and return PIL image for visualization
    
    Args:
        image_path: Path to the image
        
    Returns:
        PIL Image object
    """
    return Image.open(image_path)


def preprocess_batch(image_paths, model, batch_size=32):
    """
    Preprocess a batch of images and return their encodings
    
    Args:
        image_paths: List of image paths
        model: Pre-trained FaceNet model
        batch_size: Number of images to process at once
        
    Returns:
        Array of encodings with shape (num_images, 128)
    """
    encodings = []
    
    for image_path in image_paths:
        encoding = img_to_encoding(image_path, model)
        encodings.append(encoding)
    
    return np.vstack(encodings) if encodings else np.array([])
