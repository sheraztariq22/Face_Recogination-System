"""
Face Recognition System - Main Application
"""

import os
import numpy as np
from tensorflow.keras.models import model_from_json
import tensorflow as tf
from PIL import Image

# Import custom modules
from models.facenet_model import load_facenet_model
from utils.image_processing import img_to_encoding
from core.verification import verify
from core.recognition import who_is_it
from database.db_manager import create_database


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
    
    def verify_identity(self, image_path, claimed_identity):
        """
        Verify if the person in the image matches the claimed identity
        
        Args:
            image_path: Path to the image to verify
            claimed_identity: Name of the person claiming to be in the image
            
        Returns:
            tuple: (distance, door_open_flag)
        """
        return verify(image_path, claimed_identity, self.database, self.model)
    
    def recognize_person(self, image_path):
        """
        Identify who the person in the image is
        
        Args:
            image_path: Path to the image to recognize
            
        Returns:
            tuple: (min_distance, identity)
        """
        return who_is_it(image_path, self.database, self.model)
    
    def add_person_to_database(self, name, image_path):
        """
        Add a new person to the database
        
        Args:
            name: Name of the person
            image_path: Path to their reference image
        """
        encoding = img_to_encoding(image_path, self.model)
        self.database[name] = encoding
        print(f"Added {name} to database")
    
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


def main():
    """Main function to demonstrate the face recognition system"""
    
    # Initialize system
    system = FaceRecognitionSystem()
    
    # Build database from images
    system.build_database('images')
    
    # Example 1: Verify Younes
    print("\n=== Face Verification Test ===")
    print("Testing with Younes image...")
    dist, door_open = system.verify_identity("images/camera_0.jpg", "younes")
    print(f"Distance: {dist:.4f}, Door Open: {door_open}")
    
    # Example 2: Verify Kian with Benoit's image (should fail)
    print("\nTesting with wrong person (Benoit claiming to be Kian)...")
    dist, door_open = system.verify_identity("images/camera_2.jpg", "kian")
    print(f"Distance: {dist:.4f}, Door Open: {door_open}")
    
    # Example 3: Face Recognition
    print("\n=== Face Recognition Test ===")
    print("Identifying person in camera_0.jpg...")
    min_dist, identity = system.recognize_person("images/camera_0.jpg")
    print(f"Identified as: {identity}, Distance: {min_dist:.4f}")
    
    # Example 4: Test with multiple images
    print("\n=== Multiple Recognition Tests ===")
    test_images = ["images/camera_0.jpg", "images/camera_1.jpg"]
    for img_path in test_images:
        if os.path.exists(img_path):
            min_dist, identity = system.recognize_person(img_path)


if __name__ == "__main__":
    main()