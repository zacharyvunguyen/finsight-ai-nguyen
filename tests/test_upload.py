import os
import sys
import pytest
import time
from datetime import datetime
import logging

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.gcs import GCSManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestGCSUpload:
    def setup_method(self):
        """Setup test environment"""
        self.gcs_manager = GCSManager()
        self.test_dir = "data/test"
        self.ensure_test_dir_exists()
        self.clean_environment()
        logger.info("Test environment setup completed")

    def clean_environment(self):
        """Clean up test environment"""
        logger.info("Starting environment cleanup...")
        try:
            # Clean up GCS files
            blobs = list(self.gcs_manager.bucket.list_blobs(prefix='uploads/'))
            for blob in blobs:
                blob.delete()
                logger.info(f"Deleted GCS file: {blob.name}")

            # Clean up Firestore
            docs = list(self.gcs_manager.db.collection('pdf_metadata').stream())
            for doc in docs:
                doc.reference.delete()
                logger.info(f"Deleted Firestore document: {doc.id}")

            # Verify cleanup
            time.sleep(1)  # Wait for deletions to propagate
            remaining_docs = list(self.gcs_manager.db.collection('pdf_metadata').stream())
            remaining_blobs = list(self.gcs_manager.bucket.list_blobs(prefix='uploads/'))
            assert len(remaining_docs) == 0, f"Found {len(remaining_docs)} remaining documents"
            assert len(remaining_blobs) == 0, f"Found {len(remaining_blobs)} remaining files"
            
            logger.info("Environment cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            raise

    def teardown_method(self):
        """Cleanup after each test"""
        try:
            self.clean_environment()
            logger.info("Test cleanup completed")
        except Exception as e:
            logger.error(f"Teardown failed: {str(e)}")
            raise

    def ensure_test_dir_exists(self):
        """Ensure test directory exists"""
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
        logger.info(f"Test directory verified: {self.test_dir}")

    def create_test_pdf(self, content="Test content", filename="test.pdf"):
        """Create a test PDF file"""
        file_path = os.path.join(self.test_dir, filename)
        with open(file_path, 'wb') as f:
            f.write(content.encode())
        logger.info(f"Created test file: {file_path}")
        return file_path

    def run_single_test(self, test_func, name):
        """Run a single test with proper setup and cleanup"""
        logger.info(f"\nRunning test: {name}")
        self.clean_environment()
        try:
            test_func()
            print(f"✅ {name} passed")
            return True
        except Exception as e:
            logger.error(f"Test failed: {str(e)}")
            print(f"❌ {name} failed: {str(e)}")
            return False
        finally:
            self.clean_environment()

    def test_basic_upload(self):
        """Test basic file upload functionality"""
        # Create test file
        test_file = self.create_test_pdf(filename="basic_test.pdf")
        
        try:
            # Upload file
            result = self.gcs_manager.upload_file(test_file)
            logger.info(f"Upload result: {result}")
            
            # Verify upload success
            assert result['status'] == 'success', f"Upload failed with result: {result}"
            assert 'gcs_uri' in result, "GCS URI missing from result"
            assert 'file_hash' in result, "File hash missing from result"
            
            # Verify metadata storage
            metadata = self.gcs_manager.get_file_metadata(result['file_hash'])
            assert metadata is not None, "Metadata not found"
            assert metadata['status'] == 'uploaded', "Incorrect metadata status"
            
        finally:
            # Cleanup local file
            if os.path.exists(test_file):
                os.remove(test_file)
                logger.info(f"Cleaned up local file: {test_file}")

    def test_duplicate_detection(self):
        """Test duplicate file detection"""
        logger.info("Starting duplicate detection test")
        
        # Create test file
        test_file = self.create_test_pdf(filename="duplicate_test.pdf")
        logger.info(f"Created test file for duplicate detection: {test_file}")
        
        try:
            # First upload
            logger.info("Attempting first upload")
            result1 = self.gcs_manager.upload_file(test_file)
            logger.info(f"First upload result: {result1}")
            assert result1['status'] == 'success', f"First upload failed: {result1}"
            
            # Try uploading same file again
            logger.info("Attempting duplicate upload")
            result2 = self.gcs_manager.upload_file(test_file)
            logger.info(f"Second upload result: {result2}")
            assert result2['status'] == 'duplicate', f"Duplicate not detected: {result2}"
            assert result2['file_hash'] == result1['file_hash'], "Hash mismatch"
            
        except Exception as e:
            logger.error(f"Test failed: {str(e)}")
            raise
        
        finally:
            # Cleanup
            if os.path.exists(test_file):
                os.remove(test_file)
                logger.info(f"Cleaned up test file: {test_file}")

    def test_different_content(self):
        """Test uploading files with different content"""
        # Create two files with different content
        file1 = self.create_test_pdf(content="Content 1", filename="file1.pdf")
        file2 = self.create_test_pdf(content="Content 2", filename="file2.pdf")
        
        try:
            # Upload both files
            result1 = self.gcs_manager.upload_file(file1)
            result2 = self.gcs_manager.upload_file(file2)
            
            # Verify different hashes
            assert result1['file_hash'] != result2['file_hash']
            assert result1['status'] == 'success'
            assert result2['status'] == 'success'
            
        finally:
            # Cleanup
            for file in [file1, file2]:
                if os.path.exists(file):
                    os.remove(file)

    def test_list_files(self):
        """Test listing files in GCS bucket"""
        # Create and upload test file
        test_file = self.create_test_pdf(filename="list_test.pdf")
        
        try:
            # Upload file
            result = self.gcs_manager.upload_file(test_file)
            assert result['status'] == 'success'
            
            # List files
            files = self.gcs_manager.list_files()
            assert isinstance(files, list)
            assert len(files) > 0
            
        finally:
            # Cleanup
            if os.path.exists(test_file):
                os.remove(test_file)

def run_tests():
    """Run all tests with proper setup"""
    logger.info("Starting test suite")
    test = TestGCSUpload()
    
    tests = [
        (test.test_basic_upload, "Basic upload test"),
        (test.test_duplicate_detection, "Duplicate detection test"),
        (test.test_different_content, "Different content test"),
        (test.test_list_files, "List files test")
    ]
    
    failed_tests = []
    
    print("\nRunning GCS upload tests...")
    
    for test_func, test_name in tests:
        if not test.run_single_test(test_func, test_name):
            failed_tests.append(test_name)
    
    if failed_tests:
        print("\n❌ Some tests failed:")
        for failed_test in failed_tests:
            print(f"  - {failed_test}")
        raise AssertionError(f"Failed tests: {', '.join(failed_tests)}")
    else:
        print("\n🎉 All tests passed successfully!")

# Add this at the top level of the file, outside the TestGCSUpload class
def cleanup_test_environment():
    """Standalone function to clean up test environment"""
    logger.info("Starting standalone cleanup...")
    test = TestGCSUpload()
    test.setup_method()  # This will initialize gcs_manager
    test.clean_environment()
    logger.info("Standalone cleanup completed")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--cleanup':
        cleanup_test_environment()
    else:
        run_tests() 