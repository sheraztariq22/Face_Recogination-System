"""
FaceNet Model Loading Module
models/facenet_model.py
"""

import os
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras import backend as K
import tensorflow as tf


def load_facenet_model(model_path='keras-facenet-h5'):
    """
    Load the pre-trained FaceNet model (FIXED VERSION)
    
    Args:
        model_path: Path to the directory containing model.json and model.h5
        
    Returns:
        Loaded Keras model
    """
    K.set_image_data_format('channels_last')
    
    # Method 1: Try loading with custom objects (RECOMMENDED)
    try:
        print(f"🔄 Attempting to load model from {model_path}...")
        
        # Path to model files
        json_path = os.path.join(model_path, 'model.json')
        weights_path = os.path.join(model_path, 'model.h5')
        
        # Check if files exist
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"❌ Model JSON file not found: {json_path}")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"❌ Model weights file not found: {weights_path}")
        
        # Try loading with custom objects to handle 'Functional' class
        with open(json_path, 'r') as json_file:
            loaded_model_json = json_file.read()
        
        # Create custom objects for compatibility
        custom_objects = {
            'tf': tf,
            'Functional': tf.keras.Model  # Map Functional to Model
        }
        
        # Load model architecture
        model = model_from_json(loaded_model_json, custom_objects=custom_objects)
        
        # Load weights
        model.load_weights(weights_path)
        
        print("✅ Model loaded successfully using method 1 (JSON + H5)")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        
        return model
        
    except Exception as e1:
        print(f"⚠️ Method 1 failed: {str(e1)}")
        
        # Method 2: Try loading entire model directly
        try:
            print("🔄 Attempting method 2: Loading entire model...")
            
            # Try loading the entire model
            h5_path = os.path.join(model_path, 'model.h5')
            
            model = load_model(h5_path, compile=False, custom_objects=custom_objects)
            
            print("✅ Model loaded successfully using method 2 (Direct H5)")
            print(f"   Input shape: {model.input_shape}")
            print(f"   Output shape: {model.output_shape}")
            
            return model
            
        except Exception as e2:
            print(f"⚠️ Method 2 failed: {str(e2)}")
            
            # Method 3: Build model from scratch using inception blocks
            try:
                print("🔄 Attempting method 3: Building from scratch...")
                from models.inception_blocks_v2 import faceRecoModel
                
                # Build model with correct input shape for channels_last
                model = faceRecoModel(input_shape=(3, 160, 160))
                
                # Try to load weights
                weights_path = os.path.join(model_path, 'model.h5')
                model.load_weights(weights_path)
                
                print("✅ Model loaded successfully using method 3 (Inception blocks)")
                return model
                
            except Exception as e3:
                print(f"❌ All methods failed!")
                print(f"   Error 1: {str(e1)}")
                print(f"   Error 2: {str(e2)}")
                print(f"   Error 3: {str(e3)}")
                raise RuntimeError(
                    "Could not load FaceNet model. Please ensure:\n"
                    "1. keras-facenet-h5/model.json exists\n"
                    "2. keras-facenet-h5/model.h5 exists\n"
                    "3. Both files are valid and not corrupted\n"
                    f"Last error: {str(e3)}"
                )


def get_model_summary(model):
    """
    Get a summary of the model architecture
    
    Args:
        model: Keras model
    """
    return model.summary()


def test_model_loading(model_path='keras-facenet-h5'):
    """
    Test if model can be loaded successfully
    
    Args:
        model_path: Path to model directory
        
    Returns:
        True if successful, False otherwise
    """
    try:
        model = load_facenet_model(model_path)
        print("✅ Model test successful!")
        print(f"   Model type: {type(model)}")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        return True
    except Exception as e:
        print(f"❌ Model test failed: {str(e)}")
        return False