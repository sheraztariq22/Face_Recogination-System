"""
Image Processing Utilities
utils/image_processing.py
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K


K.set_image_data_format('channels_first')


def img_to_encoding(image_path, model):
    """
    Convert an image to its 128-dimensional encoding
    
    Args:
        image_path: Path to the image file
        model: Pre-trained FaceNet model
        
    Returns:
        Normalized encoding vector (1, 128)
    """
    # Load image and resize to 96x96 (channels-first format)
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(96, 96))
    
    # Convert to numpy array and normalize
    img_array = np.array(img) / 255.0
    img_array = np.around(img_array, decimals=12)
    
    # Convert to channels-first format (3, 96, 96)
    # PIL loads as (96, 96, 3) so we need to transpose
    if img_array.ndim == 3 and img_array.shape[2] == 3:
        img_array = np.transpose(img_array, (2, 0, 1))
    
    # Add batch dimension -> (1, 3, 96, 96)
    x_train = np.expand_dims(img_array, axis=0)
    
    # Get encoding
    embedding = model.predict_on_batch(x_train)
    
    # Normalize the encoding
    normalized_embedding = embedding / np.linalg.norm(embedding, ord=2)
    
    return normalized_embedding


def load_and_preprocess_image(image_path, target_size=(160, 160)):
    """
    Load and preprocess an image for display or further processing
    
    Args:
        image_path: Path to the image
        target_size: Target size for the image
        
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
    from PIL import Image
    return Image.open(image_path)