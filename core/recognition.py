"""
Face Recognition Module
core/recognition.py

Based on FaceNet implementation
Provides 1:K matching for face recognition
"""

import numpy as np
from utils.image_processing import img_to_encoding


def who_is_it(image_path, database, model, threshold=0.7):
    """
    Identify who the person in the image is.
    
    This is a 1:K matching problem - identify which person from the database
    matches the input image.
    
    Args:
        image_path: Path to the image to recognize
        database: Dictionary mapping names to their face encodings
        model: Pre-trained FaceNet model
        threshold: Maximum distance for positive identification (default: 0.7)
        
    Returns:
        tuple: (min_distance, identity)
            - min_distance: Smallest distance found
            - identity: Name of the identified person (or None if not found)
    """
    # Step 1: Compute encoding for the target image
    encoding = img_to_encoding(image_path, model)
    
    # Step 2: Find the closest match in the database
    min_dist = 100  # Initialize with large value
    identity = None
    
    # Loop through all people in the database
    for name, db_enc in database.items():
        # Compute L2 distance
        dist = np.linalg.norm(encoding - db_enc)
        
        # Update if this is the closest match so far
        if dist < min_dist:
            min_dist = dist
            identity = name
    
    # Step 3: Check if the best match is within threshold
    if min_dist > threshold:
        print("Not in the database.")
        print(f"  Closest match: {identity} (distance: {min_dist:.4f})")
        print(f"  Threshold: {threshold}")
        return min_dist, None
    else:
        print(f"It's {identity}!")
        print(f"  Distance: {min_dist:.4f}")
        return min_dist, identity


def recognize_with_confidence(image_path, database, model, threshold=0.7):
    """
    Recognize person with confidence score
    
    Args:
        image_path: Path to the image to recognize
        database: Dictionary mapping names to their face encodings
        model: Pre-trained FaceNet model
        threshold: Distance threshold for identification
        
    Returns:
        dict: Contains identity, distance, and confidence score
    """
    min_dist, identity = who_is_it(image_path, database, model, threshold)
    
    if identity:
        confidence = max(0, (1 - min_dist / threshold) * 100)
    else:
        confidence = 0
    
    return {
        'identity': identity,
        'distance': min_dist,
        'confidence': confidence,
        'verified': identity is not None
    }


def get_top_k_matches(image_path, database, model, k=3):
    """
    Get the top K closest matches from the database
    
    Args:
        image_path: Path to the image to recognize
        database: Dictionary mapping names to their face encodings
        model: Pre-trained FaceNet model
        k: Number of top matches to return (default: 3)
        
    Returns:
        List of tuples (name, distance) sorted by distance
    """
    # Compute encoding for the input image
    encoding = img_to_encoding(image_path, model)
    
    # Calculate distances to all people
    distances = []
    for name, db_enc in database.items():
        dist = np.linalg.norm(encoding - db_enc)
        distances.append((name, dist))
    
    # Sort by distance and return top K
    distances.sort(key=lambda x: x[1])
    return distances[:k]