"""
Webcam Utilities for Streamlit Face Recognition
utils/webcam_utils.py
"""

import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
import numpy as np
from PIL import Image


# RTC Configuration for WebRTC
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


class FaceDetectionTransformer(VideoTransformerBase):
    """Video transformer for real-time face detection"""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.captured_frame = None
        self.capture_flag = False
    
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, "Face Detected", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Capture frame if flag is set
        if self.capture_flag and len(faces) > 0:
            self.captured_frame = img
            self.capture_flag = False
        
        # Add instruction text
        cv2.putText(img, f"Faces Detected: {len(faces)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    def capture(self):
        """Set flag to capture next frame"""
        self.capture_flag = True
    
    def get_captured_frame(self):
        """Get the captured frame"""
        return self.captured_frame


def capture_from_webcam():
    """
    Capture image from webcam using streamlit-webrtc
    
    Returns:
        PIL Image or None
    """
    st.markdown("### 📷 Webcam Capture")
    
    webrtc_ctx = webrtc_streamer(
        key="face-detection",
        video_transformer_factory=FaceDetectionTransformer,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
    )
    
    if webrtc_ctx.video_transformer:
        if st.button("📸 Capture Photo"):
            webrtc_ctx.video_transformer.capture()
            st.success("Photo captured! Processing...")
            
            # Wait a moment for capture
            import time
            time.sleep(1)
            
            captured = webrtc_ctx.video_transformer.get_captured_frame()
            if captured is not None:
                # Convert BGR to RGB
                captured_rgb = cv2.cvtColor(captured, cv2.COLOR_BGR2RGB)
                return Image.fromarray(captured_rgb)
    
    return None


def simple_webcam_capture():
    """
    Simple webcam capture using OpenCV (fallback method)
    
    Returns:
        PIL Image or None
    """
    st.markdown("### 📷 Simple Webcam Capture")
    
    if st.button("📸 Capture from Webcam"):
        with st.spinner("Opening webcam..."):
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("Could not open webcam. Please check your camera connection.")
                return None
            
            # Wait for camera to warm up
            import time
            time.sleep(1)
            
            # Capture frame
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                
                st.image(img, caption="Captured Image", use_column_width=True)
                st.success("✅ Image captured successfully!")
                
                return img
            else:
                st.error("Failed to capture image from webcam.")
                return None
    
    return None


def detect_faces_in_image(image):
    """
    Detect faces in an image using OpenCV
    
    Args:
        image: PIL Image or numpy array
        
    Returns:
        List of face bounding boxes [(x, y, w, h), ...]
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Load face cascade
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    
    return faces


def draw_face_boxes(image, faces):
    """
    Draw bounding boxes around detected faces
    
    Args:
        image: PIL Image
        faces: List of face bounding boxes
        
    Returns:
        PIL Image with boxes drawn
    """
    img_array = np.array(image)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img_array, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(img_array, "Face", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return Image.fromarray(img_array)


def crop_face_from_image(image, padding=20):
    """
    Detect and crop the largest face from an image
    
    Args:
        image: PIL Image
        padding: Pixels to add around face crop
        
    Returns:
        Cropped PIL Image or original if no face detected
    """
    faces = detect_faces_in_image(image)
    
    if len(faces) == 0:
        return image
    
    # Get largest face
    largest_face = max(faces, key=lambda f: f[2] * f[3])
    x, y, w, h = largest_face
    
    # Add padding
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(width, x + w + padding)
    y2 = min(height, y + h + padding)
    
    # Crop
    cropped = img_array[y1:y2, x1:x2]
    
    return Image.fromarray(cropped)