"""
Models Package for Face Recognition System

This package contains:
- facenet_model.py: Pre-trained model loader (JSON/H5)
- inception_blocks_v2.py: Custom Inception architecture
- fr_utils.py: Utilities for weight loading and preprocessing
"""

# Don't import anything here to avoid circular imports
# Import directly in your code when needed:
# from models import fr_utils
# from models.facenet_model import load_facenet_model
# from models.inception_blocks_v2 import faceRecoModel

__all__ = ['facenet_model', 'inception_blocks_v2', 'fr_utils']