"""
Streamlit Face Recognition Application
app.py - Main application file
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
import tempfile
from datetime import datetime
import pandas as pd

# Import your face recognition system
from main import FaceRecognitionSystem
from database.db_manager import save_database, load_database, list_database_entries


# Page configuration
st.set_page_config(
    page_title="Face Recognition System",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        border-radius: 0.5rem;
        height: 3rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'system' not in st.session_state:
    st.session_state.system = None
    st.session_state.database_loaded = False
    st.session_state.access_log = []

# Initialize the system
@st.cache_resource
def initialize_system():
    """Initialize the face recognition system"""
    try:
        system = FaceRecognitionSystem(model_path='keras-facenet-h5')
        return system
    except Exception as e:
        st.error(f"Error initializing system: {str(e)}")
        return None


def load_database_ui():
    """Load the face database"""
    try:
        if st.session_state.system is None:
            st.session_state.system = initialize_system()
        
        if st.session_state.system:
            with st.spinner("Building face database..."):
                st.session_state.system.build_database('images')
                st.session_state.database_loaded = True
            return True
        return False
    except Exception as e:
        st.error(f"Error loading database: {str(e)}")
        return False


def save_uploaded_file(uploaded_file):
    """Save uploaded file temporarily"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None


def log_access(identity, verified, distance):
    """Log access attempts"""
    log_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'identity': identity,
        'verified': verified,
        'distance': f"{distance:.4f}",
        'status': 'Granted' if verified else 'Denied'
    }
    st.session_state.access_log.append(log_entry)


def main():
    # Initialize system and database on first load
    if st.session_state.system is None:
        st.session_state.system = initialize_system()
        if st.session_state.system:
            with st.spinner("Building face database..."):
                st.session_state.system.build_database('images')
                st.session_state.database_loaded = True
    
    # Header
    st.markdown('<p class="main-header">🎭 Face Recognition System</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=150)
        st.title("Navigation")
        
        page = st.radio(
            "Select Page",
            ["🏠 Home", "✅ Face Verification", "🔍 Face Recognition", 
             "➕ Add Person", "📊 Database Manager", "📈 Access Logs", "⚙️ Settings"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Database status
        if st.session_state.database_loaded:
            st.success("✅ Database Loaded")
            if st.session_state.system:
                st.info(f"👥 {len(st.session_state.system.database)} people registered")
        else:
            st.warning("⚠️ Database Not Loaded")
        
        st.markdown("---")
        
        # Load/Reload Database Button
        if st.button("🔄 Load/Reload Database"):
            with st.spinner("Loading database..."):
                if load_database_ui():
                    st.success("Database loaded successfully!")
                    st.rerun()
        
        st.markdown("---")
        st.markdown("### 📖 Quick Guide")
        st.markdown("""
        1. Load the database first
        2. Choose verification or recognition
        3. Upload an image or use webcam
        4. View results instantly
        """)
    
    # Main content based on page selection
    if page == "🏠 Home":
        show_home_page()
    elif page == "✅ Face Verification":
        show_verification_page()
    elif page == "🔍 Face Recognition":
        show_recognition_page()
    elif page == "➕ Add Person":
        show_add_person_page()
    elif page == "📊 Database Manager":
        show_database_manager_page()
    elif page == "📈 Access Logs":
        show_access_logs_page()
    elif page == "⚙️ Settings":
        show_settings_page()


def show_home_page():
    """Display home page with overview"""
    st.markdown('<p class="sub-header">Welcome to the Face Recognition System</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h2>✅</h2>
            <h3>Face Verification</h3>
            <p>Verify if a person is who they claim to be (1:1 matching)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h2>🔍</h2>
            <h3>Face Recognition</h3>
            <p>Identify who a person is from the database (1:K matching)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h2>📊</h2>
            <h3>Database Management</h3>
            <p>Add, remove, and manage authorized personnel</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # System Overview
    st.markdown("### 📊 System Overview")
    
    if st.session_state.database_loaded and st.session_state.system:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Registered Users", len(st.session_state.system.database))
        with col2:
            st.metric("Access Attempts", len(st.session_state.access_log))
        with col3:
            granted = sum(1 for log in st.session_state.access_log if log['status'] == 'Granted')
            st.metric("Access Granted", granted)
        with col4:
            denied = sum(1 for log in st.session_state.access_log if log['status'] == 'Denied')
            st.metric("Access Denied", denied)
    else:
        st.info("Load the database to see system statistics")
    
    st.markdown("---")
    
    # Features
    st.markdown("### 🌟 Key Features")
    
    features = {
        "High Accuracy": "99%+ accuracy using FaceNet deep learning model",
        "Real-time Processing": "Fast face encoding and matching (< 1 second)",
        "Secure": "128-dimensional face embeddings for secure comparison",
        "Scalable": "Handle databases with hundreds of people",
        "Easy to Use": "Intuitive interface for all operations"
    }
    
    for feature, description in features.items():
        st.markdown(f"**{feature}:** {description}")


def show_verification_page():
    """Face Verification Page (1:1 matching)"""
    st.markdown('<p class="sub-header">✅ Face Verification (1:1 Matching)</p>', unsafe_allow_html=True)
    
    if not st.session_state.database_loaded:
        st.warning("⚠️ Please load the database first from the sidebar!")
        return
    
    st.markdown("""
    <div class="info-box">
    <strong>ℹ️ How it works:</strong> Upload an image and specify who you claim to be. 
    The system will verify if you are that person.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Identity selection
        st.markdown("#### 1. Select Identity")
        people = list(st.session_state.system.database.keys())
        claimed_identity = st.selectbox("Who are you?", people)
        
        # Threshold setting
        st.markdown("#### 2. Threshold Settings")
        st.caption("⚠️ Current model uses untrained weights. Recommend threshold: 0.02-0.05")
        threshold = st.slider("Verification Threshold", 0.0, 1.0, 0.05, 0.005)
        st.caption(f"Current threshold: {threshold:.3f} (Lower = stricter, 0.05 is recommended)")
    
    with col2:
        # Image upload
        st.markdown("#### 3. Upload Image")
        uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
    
    st.markdown("---")
    
    # Verify button
    if st.button("🔍 Verify Identity", type="primary"):
        if not uploaded_file:
            st.error("❌ Please upload an image first!")
            return
        
        with st.spinner("Verifying identity..."):
            # Save uploaded file
            temp_path = save_uploaded_file(uploaded_file)
            
            try:
                # Perform verification with the user-selected threshold
                distance, verified = st.session_state.system.verify_identity(
                    temp_path, claimed_identity, threshold=threshold
                )
                
                # Log access
                log_access(claimed_identity, verified, distance)
                
                # Display results
                st.markdown("---")
                st.markdown("### 🎯 Verification Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Distance Score", f"{distance:.4f}")
                with col2:
                    st.metric("Threshold", f"{threshold:.4f}")
                with col3:
                    status = "✅ VERIFIED" if verified else "❌ DENIED"
                    st.metric("Status", status)
                
                if verified:
                    # Calculate confidence
                    if threshold > 0:
                        confidence = max(0, (1 - distance/threshold) * 100)
                    else:
                        confidence = 100 if distance == 0 else 0
                    
                    st.markdown(f"""
                    <div class="success-box">
                    <h3>✅ Access Granted</h3>
                    <p><strong>Welcome, {claimed_identity}!</strong></p>
                    <p>Distance: {distance:.4f} (Below threshold: {threshold:.4f})</p>
                    <p>Confidence: {confidence:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="error-box">
                    <h3>❌ Access Denied</h3>
                    <p><strong>You are not {claimed_identity}</strong></p>
                    <p>Distance: {distance:.4f} (Above threshold: {threshold:.4f})</p>
                    <p>The face does not match the claimed identity.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show distance comparison with all people
                st.markdown("---")
                st.markdown("### 📊 Distance Comparison with Database")
                
                from utils.image_processing import img_to_encoding
                import numpy as np
                
                test_encoding = img_to_encoding(temp_path, st.session_state.system.model)
                
                comparison_data = []
                for name, db_enc in sorted(st.session_state.system.database.items()):
                    dist = np.linalg.norm(test_encoding - db_enc)
                    match = "✓" if dist < threshold else "✗"
                    is_claimed = "← Claimed Identity" if name == claimed_identity else ""
                    comparison_data.append({
                        "Name": f"{name} {is_claimed}",
                        "Distance": f"{dist:.4f}",
                        "Below Threshold": match
                    })
                
                df_comparison = pd.DataFrame(comparison_data)
                st.dataframe(df_comparison, use_container_width=True, hide_index=True)
                
            except Exception as e:
                st.error(f"❌ Error during verification: {str(e)}")
                import traceback
                st.write(traceback.format_exc())
            finally:
                # Cleanup
                if os.path.exists(temp_path):
                    os.remove(temp_path)


def show_recognition_page():
    """Face Recognition Page (1:K matching)"""
    st.markdown('<p class="sub-header">🔍 Face Recognition (1:K Matching)</p>', unsafe_allow_html=True)
    
    if not st.session_state.database_loaded:
        st.warning("⚠️ Please load the database first from the sidebar!")
        return
    
    st.markdown("""
    <div class="info-box">
    <strong>ℹ️ How it works:</strong> Capture a live photo or upload an image and the system will search 
    the entire database to identify who the person is.
    </div>
    """, unsafe_allow_html=True)
    
    # Settings section
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### ⚙️ Settings")
        st.caption("⚠️ Current model uses untrained weights. Recommend threshold: 0.02-0.05")
        threshold = st.slider("Recognition Threshold", 0.0, 1.0, 0.05, 0.005)
        st.caption(f"Current threshold: {threshold:.3f}")
    
    with col2:
        st.markdown("#### 📊 Display Options")
        show_top_matches = st.checkbox("Show top 3 matches", value=True)
        show_distance_chart = st.checkbox("Show distance chart", value=True)
    
    st.markdown("---")
    
    # Tab selection between camera and upload
    tab1, tab2 = st.tabs(["📷 Live Camera Capture", "📁 Upload Image"])
    
    image_to_recognize = None
    source_type = None
    temp_path = None
    
    with tab1:
        st.markdown("#### Capture from Webcam")
        st.markdown("Click below to capture a photo from your camera. Make sure you have good lighting.")
        
        picture = st.camera_input("Take a picture for recognition")
        
        if picture:
            image_to_recognize = Image.open(picture)
            source_type = "camera"
            
            st.image(image_to_recognize, caption="Camera Capture", use_column_width=True)
    
    with tab2:
        st.markdown("#### Upload Image for Recognition")
        st.markdown("Choose an image from your device to identify the person.")
        
        uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'], key="recognition")
        
        if uploaded_file:
            image_to_recognize = Image.open(uploaded_file)
            source_type = "upload"
            
            st.image(image_to_recognize, caption="Uploaded Image", use_column_width=True)
    
    # Recognize button
    if st.button("🔍 Identify Person", type="primary", key="recognize_btn"):
        if image_to_recognize is None:
            st.error("❌ Please capture or upload an image first!")
            return
        
        with st.spinner("🔄 Analyzing image and searching database..."):
            try:
                # Save image temporarily
                if source_type == "camera":
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                        image_to_recognize.save(tmp_file.name)
                        temp_path = tmp_file.name
                else:
                    temp_path = save_uploaded_file(uploaded_file) if 'uploaded_file' in locals() else None
                
                if temp_path is None:
                    st.error("❌ Could not process image!")
                    return
                
                # Perform recognition
                min_distance, identity = st.session_state.system.recognize_person(temp_path)
                
                # Display results
                st.markdown("---")
                st.markdown("### 🎯 Recognition Results")
                
                # Calculate confidence
                if identity:
                    confidence = max(0, (1 - min_distance/threshold) * 100)
                else:
                    confidence = 0
                
                if identity and min_distance < threshold:
                    # Log access
                    log_access(identity, True, min_distance)
                    
                    # Success display
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("✅ Identified As", identity.upper(), delta="AUTHORIZED")
                    with col2:
                        st.metric("📏 Distance", f"{min_distance:.4f}")
                    with col3:
                        st.metric("🎯 Confidence", f"{confidence:.1f}%")
                    
                    st.markdown(f"""
                    <div class="success-box">
                    <h3>✅ Person Successfully Identified</h3>
                    <p><strong>Identity: {identity}</strong></p>
                    <p><strong>Confidence: {confidence:.1f}%</strong></p>
                    <p><strong>Distance Score: {min_distance:.4f}</strong></p>
                    <p>This person is authorized and in the database.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show the matched person's info
                    st.markdown("---")
                    st.markdown(f"### 👤 Matched Person: {identity}")
                    image_path = os.path.join('images', f"{identity}.jpg")
                    if os.path.exists(image_path):
                        matched_image = Image.open(image_path)
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.image(matched_image, caption=f"Database image of {identity}", use_column_width=True)
                        with col2:
                            st.success(f"✅ Successfully matched with {identity} in the database!")
                
                else:
                    # Not recognized
                    log_access("Unknown", False, min_distance if identity else float('inf'))
                    
                    st.markdown("""
                    <div class="error-box">
                    <h3>❌ Person Not Recognized</h3>
                    <p><strong>Unknown Person</strong></p>
                    <p>This person is not in the database or confidence is too low.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if identity:
                        st.warning(f"⚠️ Closest match: **{identity}** (Distance: {min_distance:.4f}, Confidence: {confidence:.1f}%)")
                        st.info(f"To add this person, use the 'Add Person' page.")
                
                # Show top matches
                if show_top_matches:
                    from core.recognition import get_top_k_matches
                    
                    st.markdown("---")
                    st.markdown("### 📊 Top 3 Closest Matches in Database")
                    
                    top_matches = get_top_k_matches(temp_path, st.session_state.system.database, 
                                                    st.session_state.system.model, k=3)
                    
                    match_data = []
                    for i, (name, dist) in enumerate(top_matches, 1):
                        conf = max(0, (1 - dist/threshold) * 100)
                        match_data.append({
                            "Rank": i,
                            "Name": name,
                            "Distance": f"{dist:.4f}",
                            "Confidence": f"{conf:.1f}%"
                        })
                    
                    df_matches = pd.DataFrame(match_data)
                    st.dataframe(df_matches, use_container_width=True, hide_index=True)
                    
                    # Distance chart
                    if show_distance_chart and len(match_data) > 0:
                        import matplotlib.pyplot as plt
                        
                        names = [m["Name"] for m in match_data]
                        distances = [float(m["Distance"]) for m in match_data]
                        
                        fig, ax = plt.subplots(figsize=(10, 4))
                        colors = ['green' if d < threshold else 'red' for d in distances]
                        ax.barh(names, distances, color=colors, alpha=0.7)
                        ax.axvline(x=threshold, color='orange', linestyle='--', label=f'Threshold ({threshold})')
                        ax.set_xlabel('Distance Score (lower is better)')
                        ax.set_title('Face Distance Comparison')
                        ax.legend()
                        
                        st.pyplot(fig)
                
            except Exception as e:
                st.error(f"❌ Error during recognition: {str(e)}")
                import traceback
                st.write(traceback.format_exc())
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)


def show_add_person_page():
    """Add new person to database"""
    st.markdown('<p class="sub-header">➕ Add New Person to Database</p>', unsafe_allow_html=True)
    
    if not st.session_state.database_loaded:
        st.warning("⚠️ Please load the database first from the sidebar!")
        return
    
    st.markdown("""
    <div class="info-box">
    <strong>ℹ️ Instructions:</strong> Choose to either capture a photo using your camera or upload an existing photo. 
    Make sure the face is visible, front-facing, and well-lit.
    </div>
    """, unsafe_allow_html=True)
    
    # Person name input
    st.markdown("#### Person Details")
    person_name = st.text_input("Enter person's name", placeholder="e.g., John Doe", key="person_name_input")
    
    # Tab selection between camera and upload
    tab1, tab2 = st.tabs(["📷 Capture from Camera", "📁 Upload Photo"])
    
    image_to_add = None
    source_type = None
    
    with tab1:
        st.markdown("#### Capture Photo from Camera")
        st.markdown("""
        Click the button below to capture a photo directly from your webcam.
        Position yourself in good lighting with a clear view of your face.
        """)
        
        picture = st.camera_input("Take a picture")
        
        if picture:
            image_to_add = Image.open(picture)
            source_type = "camera"
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(image_to_add, caption="Camera Capture Preview", use_column_width=True)
            with col2:
                st.markdown("#### Image Quality Checklist")
                st.markdown("""
                - ✅ Face is front-facing
                - ✅ Good lighting
                - ✅ Face clearly visible
                - ✅ No sunglasses or masks
                - ✅ Face takes up most of frame
                """)
    
    with tab2:
        st.markdown("#### Upload Photo from File")
        st.markdown("Choose a clear, front-facing photo from your device.")
        
        uploaded_file = st.file_uploader("Choose a photo", type=['jpg', 'jpeg', 'png'], key="add_person_upload")
        
        if uploaded_file:
            image_to_add = Image.open(uploaded_file)
            source_type = "upload"
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(image_to_add, caption="Upload Preview", use_column_width=True)
            with col2:
                st.markdown("#### Image Quality Checklist")
                st.markdown("""
                - ✅ Clear, front-facing photo
                - ✅ Good lighting
                - ✅ Face clearly visible
                - ✅ No sunglasses or masks
                - ✅ Minimum 160x160 pixels
                """)
    
    # Add to database button
    if st.button("➕ Add to Database", type="primary", key="add_person_btn"):
        if not person_name:
            st.error("❌ Please enter a name!")
            return
        if image_to_add is None:
            st.error("❌ Please capture or upload an image!")
            return
        
        # Check if person already exists
        if person_name.lower() in [name.lower() for name in st.session_state.system.database.keys()]:
            st.warning(f"⚠️ {person_name} already exists in the database!")
            return
        
        with st.spinner(f"Processing and adding {person_name} to database..."):
            # Save image to images folder
            image_path = os.path.join('images', f"{person_name}.jpg")
            
            # Ensure images directory exists
            os.makedirs('images', exist_ok=True)
            
            # Save the image
            image_to_add.save(image_path)
            
            try:
                # Add to database
                st.session_state.system.add_person_to_database(person_name, image_path)
                
                st.success(f"✅ {person_name} has been successfully added to the database!")
                st.balloons()
                
                # Show updated count
                st.info(f"👥 Total people in database: {len(st.session_state.system.database)}")
                
                # Display the added person
                st.markdown("---")
                st.markdown(f"### ✅ Added: {person_name}")
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(image_to_add, caption=f"{person_name}", use_column_width=True)
                with col2:
                    st.markdown(f"""
                    **Name:** {person_name}  
                    **Source:** {'Camera' if source_type == 'camera' else 'File Upload'}  
                    **Status:** ✅ Successfully Added
                    """)
                
            except Exception as e:
                st.error(f"❌ Error adding person: {str(e)}")
                # Remove image if failed
                if os.path.exists(image_path):
                    os.remove(image_path)


def show_database_manager_page():
    """Manage database entries"""
    st.markdown('<p class="sub-header">📊 Database Manager</p>', unsafe_allow_html=True)
    
    if not st.session_state.database_loaded:
        st.warning("⚠️ Please load the database first from the sidebar!")
        return
    
    # Database statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total People", len(st.session_state.system.database))
    with col2:
        st.metric("Database Size", f"{len(st.session_state.system.database) * 128 * 4 / 1024:.1f} KB")
    with col3:
        st.metric("Encoding Size", "128-D vectors")
    
    st.markdown("---")
    
    # List all people
    st.markdown("### 👥 Registered People")
    
    if st.session_state.system.database:
        # Create dataframe
        data = []
        for name in st.session_state.system.database.keys():
            image_path = os.path.join('images', f"{name}.jpg")
            exists = "✅" if os.path.exists(image_path) else "❌"
            data.append({
                "Name": name,
                "Image": exists,
                "Encoding": "✅"
            })
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
        
        # Remove person
        st.markdown("---")
        st.markdown("### 🗑️ Remove Person")
        
        person_to_remove = st.selectbox("Select person to remove", 
                                       list(st.session_state.system.database.keys()))
        
        # Confirmation checkbox
        confirm_removal = st.checkbox(f"I want to remove {person_to_remove} from the database", 
                                     key=f"confirm_{person_to_remove}")
        
        if confirm_removal:
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("🗑️ Confirm Remove", type="secondary"):
                    try:
                        # Remove from database
                        st.session_state.system.remove_person_from_database(person_to_remove)
                        
                        # Remove image file if it exists
                        image_path = os.path.join('images', f"{person_to_remove}.jpg")
                        if os.path.exists(image_path):
                            os.remove(image_path)
                        
                        st.success(f"✅ {person_to_remove} has been removed from the database!")
                        
                        # Reset the session state and rerun
                        st.session_state.pop(f"confirm_{person_to_remove}", None)
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error removing {person_to_remove}: {str(e)}")
    else:
        st.info("No people in database yet. Add some using the 'Add Person' page!")


def show_access_logs_page():
    """Display access logs"""
    st.markdown('<p class="sub-header">📈 Access Logs</p>', unsafe_allow_html=True)
    
    if not st.session_state.access_log:
        st.info("No access attempts yet. Use verification or recognition to generate logs.")
        return
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    granted = sum(1 for log in st.session_state.access_log if log['status'] == 'Granted')
    denied = sum(1 for log in st.session_state.access_log if log['status'] == 'Denied')
    total = len(st.session_state.access_log)
    
    with col1:
        st.metric("Total Attempts", total)
    with col2:
        st.metric("Access Granted", granted)
    with col3:
        st.metric("Access Denied", denied)
    with col4:
        approval_rate = (granted / total * 100) if total > 0 else 0
        st.metric("Approval Rate", f"{approval_rate:.1f}%")
    
    st.markdown("---")
    
    # Display logs
    st.markdown("### 📋 Recent Access Attempts")
    
    df = pd.DataFrame(st.session_state.access_log)
    
    # Color code status
    def highlight_status(row):
        if row['status'] == 'Granted':
            return ['background-color: #d4edda'] * len(row)
        else:
            return ['background-color: #f8d7da'] * len(row)
    
    styled_df = df.style.apply(highlight_status, axis=1)
    st.dataframe(styled_df, use_container_width=True)
    
    # Download logs
    if st.button("📥 Download Logs as CSV"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"access_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )


def show_settings_page():
    """Settings page"""
    st.markdown('<p class="sub-header">⚙️ Settings</p>', unsafe_allow_html=True)
    
    st.markdown("### 🎛️ System Configuration")
    
    # Model settings
    st.markdown("#### Model Settings")
    st.info(f"**Model Path:** keras-facenet-h5/")
    st.info(f"**Encoding Size:** 128 dimensions")
    st.info(f"**Input Size:** 160 x 160 pixels")
    
    st.markdown("---")
    
    # Database settings
    st.markdown("#### Database Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Save Database to File"):
            try:
                from database.db_manager import save_database
                save_database(st.session_state.system.database, 'database/face_database.pkl')
                st.success("✅ Database saved successfully!")
            except Exception as e:
                st.error(f"Error saving database: {str(e)}")
    
    with col2:
        if st.button("📂 Load Database from File"):
            try:
                from database.db_manager import load_database
                db = load_database('database/face_database.pkl')
                st.session_state.system.database = db
                st.success("✅ Database loaded successfully!")
            except Exception as e:
                st.error(f"Error loading database: {str(e)}")
    
    st.markdown("---")
    
    # Clear logs
    st.markdown("#### Clear Data")
    if st.button("🗑️ Clear Access Logs", type="secondary"):
        if st.checkbox("Confirm clear all logs"):
            st.session_state.access_log = []
            st.success("✅ Access logs cleared!")
            st.rerun()
    
    st.markdown("---")
    
    # About
    st.markdown("#### 📖 About")
    st.markdown("""
    **Face Recognition System v1.0**
    
    - **Technology:** FaceNet Deep Learning Model
    - **Framework:** TensorFlow/Keras
    - **Interface:** Streamlit
    - **Accuracy:** 99%+
    - **Speed:** < 1 second per face
    
    © 2024 Face Recognition System
    """)


if __name__ == "__main__":
    main()