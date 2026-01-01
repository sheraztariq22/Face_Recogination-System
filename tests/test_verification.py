"""
Unit Tests for Face Verification
tests/test_verification.py
"""

import pytest
import numpy as np
from main import FaceRecognitionSystem


@pytest.fixture
def system():
    """Create a face recognition system instance for testing"""
    sys = FaceRecognitionSystem()
    sys.build_database('images')
    return sys


class TestVerification:
    """Test cases for face verification functionality"""
    
    def test_verify_correct_identity(self, system):
        """Test verification with correct identity"""
        dist, verified = system.verify_identity(
            "images/camera_0.jpg", 
            "younes"
        )
        assert verified == True
        assert dist < 0.7
        assert isinstance(dist, (float, np.floating))
    
    def test_verify_wrong_identity(self, system):
        """Test verification with wrong identity"""
        dist, verified = system.verify_identity(
            "images/camera_2.jpg", 
            "kian"
        )
        assert verified == False
        assert dist > 0.7
    
    def test_verify_nonexistent_identity(self, system):
        """Test verification with identity not in database"""
        dist, verified = system.verify_identity(
            "images/camera_0.jpg", 
            "unknown_person"
        )
        assert verified == False
        assert dist == float('inf')
    
    def test_distance_consistency(self, system):
        """Test that same image gives same distance"""
        dist1, _ = system.verify_identity(
            "images/camera_0.jpg", 
            "younes"
        )
        dist2, _ = system.verify_identity(
            "images/camera_0.jpg", 
            "younes"
        )
        assert np.isclose(dist1, dist2)
    
    def test_threshold_boundary(self, system):
        """Test verification at threshold boundary"""
        # This should be right at or near the boundary
        dist, verified = system.verify_identity(
            "images/camera_0.jpg", 
            "younes"
        )
        if dist < 0.7:
            assert verified == True
        else:
            assert verified == False


class TestRecognition:
    """Test cases for face recognition functionality"""
    
    def test_recognize_known_person(self, system):
        """Test recognition of person in database"""
        min_dist, identity = system.recognize_person(
            "images/camera_0.jpg"
        )
        assert identity == "younes"
        assert min_dist < 0.7
    
    def test_recognize_returns_closest_match(self, system):
        """Test that recognition returns the closest match"""
        min_dist, identity = system.recognize_person(
            "images/camera_0.jpg"
        )
        assert identity is not None
        assert isinstance(min_dist, (float, np.floating))
    
    def test_recognition_consistency(self, system):
        """Test that same image gives consistent results"""
        dist1, id1 = system.recognize_person("images/camera_0.jpg")
        dist2, id2 = system.recognize_person("images/camera_0.jpg")
        
        assert id1 == id2
        assert np.isclose(dist1, dist2)


class TestDatabase:
    """Test cases for database management"""
    
    def test_database_creation(self, system):
        """Test that database is created with entries"""
        assert len(system.database) > 0
        assert isinstance(system.database, dict)
    
    def test_add_person(self, system):
        """Test adding a new person to database"""
        initial_count = len(system.database)
        
        # Note: This would need an actual test image
        # system.add_person_to_database('test_person', 'path/to/test/image.jpg')
        # assert len(system.database) == initial_count + 1
        pass
    
    def test_remove_person(self, system):
        """Test removing a person from database"""
        if len(system.database) > 0:
            person_to_remove = list(system.database.keys())[0]
            initial_count = len(system.database)
            
            system.remove_person_from_database(person_to_remove)
            assert len(system.database) == initial_count - 1
            assert person_to_remove not in system.database
    
    def test_encoding_shape(self, system):
        """Test that encodings have correct shape"""
        for name, encoding in system.database.items():
            assert encoding.shape == (1, 128)
            break  # Just test one


if __name__ == "__main__":
    pytest.main([__file__, "-v"])