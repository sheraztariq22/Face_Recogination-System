"""
Database Manager for Face Encodings
database/db_manager.py
"""

import os
import pickle
import json
import numpy as np
from utils.image_processing import img_to_encoding


def create_database(images_folder, model, extensions=('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')):
    """
    Create a face encoding database from images in a folder
    
    Args:
        images_folder: Path to folder containing images
        model: Pre-trained FaceNet model
        extensions: Tuple of valid image extensions
        
    Returns:
        Dictionary mapping names to encodings
    """
    database = {}
    
    if not os.path.exists(images_folder):
        raise FileNotFoundError(f"Images folder not found: {images_folder}")
    
    # Get all image files
    image_files = [f for f in os.listdir(images_folder) 
                   if f.lower().endswith(tuple(ext.lower() for ext in extensions))]
    
    if not image_files:
        print(f"Warning: No image files found in {images_folder}")
        return database
    
    print(f"\n{'='*50}")
    print(f"Building Face Database")
    print(f"{'='*50}")
    print(f"Processing {len(image_files)} images from {images_folder}...\n")
    
    for i, image_file in enumerate(image_files, 1):
        # Extract name from filename (remove extension)
        name = os.path.splitext(image_file)[0]
        
        # Skip camera images (these are for testing)
        if name.startswith('camera'):
            print(f"  [{i}/{len(image_files)}] ⊘ Skipping test image: {name}")
            continue
        
        # Get full path
        image_path = os.path.join(images_folder, image_file)
        
        try:
            # Generate encoding
            encoding = img_to_encoding(image_path, model)
            database[name] = encoding
            print(f"  [{i}/{len(image_files)}] ✓ Added: {name:<20} (encoding shape: {encoding.shape})")
        except Exception as e:
            print(f"  [{i}/{len(image_files)}] ✗ Error processing {name}: {str(e)}")
    
    print(f"\n{'='*50}")
    print(f"Database created with {len(database)} entries")
    print(f"{'='*50}\n")
    
    return database


def save_database(database, filepath='database/face_database.pkl'):
    """
    Save face encoding database to disk
    
    Args:
        database: Dictionary of face encodings
        filepath: Path to save the database
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(database, f)
    
    print(f"Database saved to {filepath}")


def load_database(filepath='database/face_database.pkl'):
    """
    Load face encoding database from disk
    
    Args:
        filepath: Path to the saved database
        
    Returns:
        Dictionary of face encodings
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Database file not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        database = pickle.load(f)
    
    print(f"Database loaded from {filepath}")
    print(f"Contains {len(database)} entries")
    
    return database


def add_to_database(database, name, image_path, model):
    """
    Add a new person to the database
    
    Args:
        database: Existing database dictionary
        name: Name of the person
        image_path: Path to their image
        model: FaceNet model
        
    Returns:
        Updated database
    """
    encoding = img_to_encoding(image_path, model)
    database[name] = encoding
    print(f"Added {name} to database")
    return database


def remove_from_database(database, name):
    """
    Remove a person from the database
    
    Args:
        database: Existing database dictionary
        name: Name of the person to remove
        
    Returns:
        Updated database
    """
    if name in database:
        del database[name]
        print(f"Removed {name} from database")
    else:
        print(f"{name} not found in database")
    return database


def list_database_entries(database):
    """
    List all entries in the database
    
    Args:
        database: Face encoding database
    """
    print(f"\nDatabase contains {len(database)} entries:")
    for i, name in enumerate(database.keys(), 1):
        print(f"  {i}. {name}")