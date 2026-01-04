"""
Face Recognition System - Main Application

Based on FaceNet (https://arxiv.org/pdf/1503.03832.pdf)
This implementation includes face verification and recognition using triplet loss.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Lambda
from tensorflow.keras import backend as K
from PIL import Image

# Set to channels_first format (required for the Inception model built from scratch)
K.set_image_data_format('channels_first')

# Import custom modules
from models.facenet_model import load_facenet_model
from database.db_manager import create_database


# ======================== UTILITY FUNCTIONS ========================

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


def triplet_loss(y_true, y_pred, alpha=0.2):
    """
    Implementation of the triplet loss as defined by FaceNet
    
    The triplet loss minimizes the distance between an anchor and positive image
    while maximizing the distance between anchor and negative image.
    
    Args:
        y_true: True labels (required by Keras, not used in this implementation)
        y_pred: List containing [anchor_encoding, positive_encoding, negative_encoding]
        alpha: Margin parameter (default: 0.2)
        
    Returns:
        Scalar loss value
    """
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    # Step 1: Compute the squared L2 distance between anchor and positive
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    
    # Step 2: Compute the squared L2 distance between anchor and negative
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    
    # Step 3: Compute the triplet loss
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    
    # Step 4: Take the maximum with 0 and sum over the batch
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    
    return loss


def verify(image_path, identity, database, model, threshold=0.05):
    """
    Verify if the person in the image matches the claimed identity (1:1 matching)
    
    Args:
        image_path: Path to the image to verify
        identity: Name of the person claiming to be in the image
        database: Dictionary mapping names to their face encodings
        model: Pre-trained FaceNet model
        threshold: Distance threshold for verification (default: 0.05)
        
    Returns:
        tuple: (distance, verified_flag)
            - distance: L2 distance between encodings
            - verified_flag: True if verified, False otherwise
    """
    # Step 1: Compute the encoding of the input image
    encoding = img_to_encoding(image_path, model)
    
    # Step 2: Compute the L2 distance between the input encoding and database encoding
    dist = np.linalg.norm(encoding - database[identity])
    
    # Step 3: Check if distance is below threshold
    if dist < threshold:
        print(f"It's {identity}, welcome in!")
        door_open = True
    else:
        print(f"It's not {identity}, please go away")
        door_open = False
    
    return dist, door_open


def who_is_it(image_path, database, model, threshold=0.7):
    """
    Identify who the person in the image is (1:K matching)
    
    Args:
        image_path: Path to the image to recognize
        database: Dictionary mapping names to their face encodings
        model: Pre-trained FaceNet model
        threshold: Distance threshold for identification (default: 0.7)
        
    Returns:
        tuple: (min_distance, identity)
            - min_distance: Minimum distance found
            - identity: Name of the identified person (or None if not in database)
    """
    # Step 1: Compute the encoding for the input image
    encoding = img_to_encoding(image_path, model)
    
    # Step 2: Find the closest match in the database
    min_dist = 100  # Initialize with large value
    identity = None
    
    # Loop through all people in the database
    for (name, db_enc) in database.items():
        # Compute L2 distance
        dist = np.linalg.norm(encoding - db_enc)
        
        # Update if this is the closest match so far
        if dist < min_dist:
            min_dist = dist
            identity = name
    
    # Step 3: Check if the best match is within threshold
    if min_dist > threshold:
        print("Not in the database.")
    else:
        print(f"It's {identity}, the distance is {min_dist:.4f}")
    
    return min_dist, identity


# ======================== FACE RECOGNITION SYSTEM CLASS ========================

class FaceRecognitionSystem:
    """Main class for face recognition operations"""
    
    def __init__(self, model_path='keras-facenet-h5'):
        """
        Initialize the face recognition system
        
        Args:
            model_path: Path to the pre-trained FaceNet model
        """
        self.model = load_facenet_model(model_path)
        self.database = {}
        print("Face Recognition System initialized successfully!")
    
    def build_database(self, images_folder='images'):
        """
        Build the face database from images in the specified folder
        
        Args:
            images_folder: Path to folder containing authorized person images
        """
        print("Building face database...")
        self.database = create_database(images_folder, self.model)
        print(f"Database created with {len(self.database)} entries")
        return self.database
    
    def verify_identity(self, image_path, claimed_identity, threshold=0.05):
        """
        Verify if the person in the image matches the claimed identity
        
        Args:
            image_path: Path to the image to verify
            claimed_identity: Name of the person claiming to be in the image
            threshold: Distance threshold (default: 0.05)
            
        Returns:
            tuple: (distance, door_open_flag)
        """
        return verify(image_path, claimed_identity, self.database, self.model, threshold)
    
    def recognize_person(self, image_path, threshold=0.7):
        """
        Identify who the person in the image is
        
        Args:
            image_path: Path to the image to recognize
            threshold: Distance threshold (default: 0.7)
            
        Returns:
            tuple: (min_distance, identity)
        """
        return who_is_it(image_path, self.database, self.model, threshold)
    
    def add_person_to_database(self, name, image_path):
        """
        Add a new person to the database
        
        Args:
            name: Name of the person
            image_path: Path to their reference image
        """
        encoding = img_to_encoding(image_path, self.model)
        self.database[name] = encoding
        print(f"Added {name} to database with encoding shape {encoding.shape}")
    
    def remove_person_from_database(self, name):
        """
        Remove a person from the database
        
        Args:
            name: Name of the person to remove
        """
        if name in self.database:
            del self.database[name]
            print(f"Removed {name} from database")
        else:
            print(f"{name} not found in database")


# ======================== MAIN FUNCTION ========================

def main():
    """Main function to demonstrate the face recognition system"""
    
    # Initialize system
    system = FaceRecognitionSystem()
    
    # Build database from images
    system.build_database('images')
    
    # Example 1: Verify Younes
    print("\n" + "="*50)
    print("FACE VERIFICATION TEST")
    print("="*50)
    print("\nTest 1: Younes claiming to be Younes")
    dist, door_open = system.verify_identity("images/camera_0.jpg", "younes")
    print(f"Result - Distance: {dist:.4f}, Door Open: {door_open}")
    
    # Example 2: Verify Kian with Benoit's image (should fail)
    print("\nTest 2: Benoit claiming to be Kian (should fail)")
    dist, door_open = system.verify_identity("images/camera_2.jpg", "kian")
    print(f"Result - Distance: {dist:.4f}, Door Open: {door_open}")
    
    # Example 3: Face Recognition
    print("\n" + "="*50)
    print("FACE RECOGNITION TEST")
    print("="*50)
    print("\nTest 1: Identifying person in camera_0.jpg")
    min_dist, identity = system.recognize_person("images/camera_0.jpg")
    print(f"Result - Identified as: {identity}, Distance: {min_dist:.4f}")
    
    # Example 4: Test with multiple images
    print("\nTest 2: Identifying person in younes.jpg")
    min_dist, identity = system.recognize_person("images/younes.jpg")
    print(f"Result - Identified as: {identity}, Distance: {min_dist:.4f}")


if __name__ == "__main__":
    main()