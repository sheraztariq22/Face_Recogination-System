"""
Test Model Loading
test_model.py
"""

from models.facenet_model import test_model_loading

print("="*50)
print("Testing FaceNet Model Loading")
print("="*50)

# Test loading
success = test_model_loading('keras-facenet-h5')

if success:
    print("\n✅ SUCCESS! Model loaded correctly.")
    print("You can now run the Streamlit app.")
else:
    print("\n❌ FAILED! Please check the error messages above.")
    print("\nTroubleshooting steps:")
    print("1. Check if keras-facenet-h5/model.json exists")
    print("2. Check if keras-facenet-h5/model.h5 exists")
    print("3. Verify file sizes (model.h5 should be ~128MB)")
    print("4. Try downloading model files again")