"""
Models Package for Face Recognition System

This package contains:
- facenet_model.py: Pre-trained model loader (JSON/H5)
- inception_blocks_v2.py: Custom Inception architecture
- fr_utils.py: Utilities for weight loading and preprocessing
"""

# Import key utilities
from models import fr_utils

__all__ = ['fr_utils']