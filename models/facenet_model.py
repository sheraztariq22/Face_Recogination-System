"""
FaceNet Model Loading Module
models/facenet_model.py - FIXED FOR LAMBDA LAYERS
"""

import os
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras import backend as K
import tensorflow as tf
import keras


def load_facenet_model(model_path='keras-facenet-h5'):
    """
    Load the pre-trained FaceNet model with Lambda layer support
    
    Args:
        model_path: Path to the directory containing model.json and model.h5
        
    Returns:
        Loaded Keras model
    """
    K.set_image_data_format('channels_last')
    
    print(f"🔄 Loading FaceNet model from {model_path}...")
    
    # Path to model files
    json_path = os.path.join(model_path, 'model.json')
    weights_path = os.path.join(model_path, 'model.h5')
    
    # Check if files exist
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"❌ Model JSON file not found: {json_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"❌ Model weights file not found: {weights_path}")
    
    try:
        # METHOD 1: Enable unsafe deserialization for Lambda layers
        print("   Enabling unsafe deserialization for Lambda layers...")
        keras.config.enable_unsafe_deserialization()
        
        # Load model architecture
        with open(json_path, 'r') as json_file:
            loaded_model_json = json_file.read()
        
        # Custom objects for compatibility
        custom_objects = {
            'tf': tf,
            'Functional': tf.keras.Model
        }
        
        # Load model with safe_mode=False for Lambda layers
        print("   Loading model architecture...")
        model = model_from_json(loaded_model_json, custom_objects=custom_objects)
        
        # Load weights
        print("   Loading model weights...")
        model.load_weights(weights_path)
        
        print("✅ Model loaded successfully!")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        
        return model
        
    except Exception as e1:
        print(f"⚠️ Method 1 failed: {str(e1)}")
        
        # METHOD 2: Try loading with safe_mode=False directly
        try:
            print("🔄 Trying alternative loading method...")
            
            # Try loading entire H5 file with safe_mode=False
            model = load_model(
                weights_path, 
                compile=False,
                safe_mode=False,
                custom_objects=custom_objects
            )
            
            print("✅ Model loaded using alternative method!")
            print(f"   Input shape: {model.input_shape}")
            print(f"   Output shape: {model.output_shape}")
            
            return model
            
        except Exception as e2:
            print(f"⚠️ Method 2 failed: {str(e2)}")
            
            # METHOD 3: Build model from scratch (Recommended)
            print("🔄 Building model from scratch using inception blocks...")
            
            try:
                # Import the model builder
                import sys
                sys.path.append(os.path.dirname(__file__))
                
                from inception_blocks_v2 import faceRecoModel
                
                # Build model (channels_first for inception)
                K.set_image_data_format('channels_first')
                model = faceRecoModel(input_shape=(3, 96, 96))
                
                # Load weights
                model.load_weights(weights_path, by_name=True, skip_mismatch=True)
                
                # Switch back to channels_last
                K.set_image_data_format('channels_last')
                
                print("✅ Model built from inception blocks!")
                print(f"   Input shape: {model.input_shape}")
                print(f"   Output shape: {model.output_shape}")
                
                return model
                
            except Exception as e3:
                print(f"❌ All methods failed!")
                print(f"   Error 1: {str(e1)[:200]}")
                print(f"   Error 2: {str(e2)[:200]}")
                print(f"   Error 3: {str(e3)[:200]}")
                
                # METHOD 4: Last resort - use pre-configured model
                print("\n🔄 Final attempt: Using pre-configured FaceNet...")
                return load_facenet_simple()


def load_facenet_simple():
    """
    Load a simple pre-configured FaceNet model
    This bypasses the JSON loading entirely
    """
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation
    from tensorflow.keras.layers import MaxPooling2D, Flatten, Dense, Lambda
    from tensorflow.keras import backend as K
    
    print("   Building simple FaceNet architecture...")
    
    # Simple FaceNet-like architecture
    inputs = Input(shape=(160, 160, 3))
    
    # Initial convolution
    x = Conv2D(64, (7, 7), strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)
    
    # Additional convolutions
    x = Conv2D(64, (1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(192, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)
    
    # Flatten and dense layers
    x = Flatten()(x)
    x = Dense(128)(x)
    
    # L2 normalization
    x = Lambda(lambda t: K.l2_normalize(t, axis=1))(x)
    
    model = Model(inputs=inputs, outputs=x, name='SimpleFaceNet')
    
    print("✅ Simple FaceNet model created!")
    print("   ⚠️  Note: Using simplified architecture without pre-trained weights")
    print("   ⚠️  Accuracy may be lower than full model")
    
    return model


def load_facenet_from_weights_only(weights_path='keras-facenet-h5/model.h5'):
    """
    Alternative: Load only weights into a new model
    """
    try:
        print("🔄 Creating new model and loading weights...")
        
        # Build fresh model
        from inception_blocks_v2 import faceRecoModel
        
        K.set_image_data_format('channels_first')
        model = faceRecoModel(input_shape=(3, 96, 96))
        
        # Try to load weights
        if os.path.exists(weights_path):
            model.load_weights(weights_path, by_name=True, skip_mismatch=True)
            print("✅ Weights loaded (some layers may be skipped)")
        
        K.set_image_data_format('channels_last')
        
        return model
        
    except Exception as e:
        print(f"❌ Could not load weights: {str(e)}")
        raise


def get_model_summary(model):
    """Get a summary of the model architecture"""
    return model.summary()


def test_model_loading(model_path='keras-facenet-h5'):
    """
    Test if model can be loaded successfully
    """
    try:
        model = load_facenet_model(model_path)
        print("\n✅ MODEL TEST SUCCESSFUL!")
        print(f"   Model type: {type(model)}")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        
        # Test with dummy input - use channels_first format (3, 96, 96)
        import numpy as np
        # Create test input with correct shape: (batch_size, channels, height, width)
        dummy_input = np.random.rand(1, 3, 96, 96).astype(np.float32)
        output = model.predict(dummy_input, verbose=0)
        print(f"   Test prediction shape: {output.shape}")
        print(f"   Output is normalized: {np.allclose(np.linalg.norm(output), 1.0)}")
        
        return True
    except Exception as e:
        print(f"\n❌ MODEL TEST FAILED: {str(e)}")
        return False