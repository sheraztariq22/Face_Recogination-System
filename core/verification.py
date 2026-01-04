"""
Face Verification Module
core/verification.py

Based on FaceNet implementation
Provides 1:1 matching for face verification
"""

import numpy as np
from utils.image_processing import img_to_encoding


def verify(image_path, identity, database, model, threshold=0.05):
    """
    Verify if the person in the image matches the claimed identity.
    
    This is a 1:1 matching problem - verify that the person is who they claim to be.
    
    Args:
        image_path: Path to the image to verify
        identity: Name of the person claiming to be in the image
        database: Dictionary mapping names to their face encodings
        model: Pre-trained FaceNet model
        threshold: Maximum distance for positive verification (default: 0.05)
        
    Returns:
        tuple: (distance, door_open_flag)
            - distance: L2 distance between encodings
            - door_open_flag: True if verified, False otherwise
    """
    # Step 1: Compute encoding for the input image
    encoding = img_to_encoding(image_path, model)
    
    # Step 2: Check if identity exists in database
    if identity not in database:
        print(f"Error: {identity} is not in the database")
        return float('inf'), False
    
    # Step 3: Compute L2 distance between encodings
    dist = np.linalg.norm(encoding - database[identity])
    
    # Step 4: Compare distance with threshold
    if dist < threshold:
        print(f"It's {identity}, welcome in!")
        door_open = True
    else:
        print(f"It's not {identity}, please go away")
        print(f"  Distance: {dist:.4f} (threshold: {threshold})")
        door_open = False
    
    return dist, door_open


def batch_verify(image_paths, identities, database, model, threshold=0.05):
    """
    Verify multiple images at once
    
    Args:
        image_paths: List of image paths
        identities: List of claimed identities
        database: Face encoding database
        model: FaceNet model
        threshold: Verification threshold
        
    Returns:
        List of verification results
    """
    results = []
    for img_path, identity in zip(image_paths, identities):
        dist, verified = verify(img_path, identity, database, model, threshold)
        results.append({
            'image': img_path,
            'identity': identity,
            'distance': dist,
            'verified': verified
        })
    return results
