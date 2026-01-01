"""
FaceNet Model Loading Module
models/facenet_model.py
"""

import os
from tensorflow.keras.models import model_from_json
from tensorflow.keras import backend as K


def load_facenet_model(model_path='keras-facenet-h5'):
    """
    Load the pre-trained FaceNet model
    
    Args:
        model_path: Path to the directory containing model.json and model.h5
        
    Returns:
        Loaded Keras model
    """
    K.set_image_data_format('channels_last')
    
    # Load model architecture
    json_path = os.path.join(model_path, 'model.json')
    weights_path = os.path.join(model_path, 'model.h5')
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Model JSON file not found: {json_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights file not found: {weights_path}")
    
    # Load model
    with open(json_path, 'r') as json_file:
        loaded_model_json = json_file.read()
    
    model = model_from_json(loaded_model_json)
    model.load_weights(weights_path)
    
    print(f"Model loaded successfully from {model_path}")
    print(f"Input shape: {model.inputs}")
    print(f"Output shape: {model.outputs}")
    
    return model


def get_model_summary(model):
    """
    Get a summary of the model architecture
    
    Args:
        model: Keras model
    """
    return model.summary()