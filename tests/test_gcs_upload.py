import os
import sys
import pytest
import time
import logging
from datetime import datetime
from typing import Dict, Any
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.gcs import GCSManager

# Configure logging with custom format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
TEST_PDF_PATH = "data/test/pdfs/06_09.05.24_0545_BD3_PrelimBook Proj_CY_AugFY25.pdf"

def get_file_link(file_path: str) -> str:
    """Convert file path to clickable link"""
    abs_path = os.path.abspath(file_path)
    return f"file://{abs_path}"

def get_gcs_link(gcs_uri: str, project_id: str) -> Dict[str, str]:
    """Convert GCS URI to clickable console and public storage links"""
    bucket_path = gcs_uri.replace('gs://', '')
    bucket_name = bucket_path.split('/')[0]
    file_path = '/'.join(bucket_path.split('/')[1:])
    
    console_url = f"https://console.cloud.google.com/storage/browser/{bucket_name}/{file_path}?project={project_id}"
    public_url = f"https://storage.cloud.google.com/{bucket_name}/{file_path}"
    
    return {
        'console_url': console_url,
        'public_url': public_url,
        'bucket_url': f"https://console.cloud.google.com/storage/browser/{bucket_name}?project={project_id}"
    }

def print_header(title: str) -> None:
    """Print a formatted header"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")

def print_result(status: bool, service: str, details: list = None) -> None:
    """Print a formatted test result"""
    status_icon = "✅" if status else "❌"
    print(f"{status_icon} {service}: {'Success' if status else 'Failed'}")
    if details:
        for line in details:
            print(f"   • {line}")

def log_upload_result(result: Dict[str, Any], project_id: str) -> None:
    """Log upload result with formatted output"""
    status = result.get('status', 'unknown')
    status_emoji = {
        'success': '✅',
        'duplicate': '⚠️',
        'error': '❌',
        'unknown': '❓'
    }.get(status, '❓')
    
    print(f"\n{status_emoji} Upload Status: {status.upper()}")
    print(f"{'─'*80}")
    
    if status == 'success':
        gcs_uri = result.get('gcs_uri', '')
        file_hash = result.get('file_hash', '')
        
        print(f"📂 File Location:")
        print(f"   • GCS URI: {gcs_uri}")
        
        links = get_gcs_link(gcs_uri, project_id)
        print(f"   • Console Link: {links['console_url']}")
        print(f"   • Public URL: {links['public_url']}")
        print(f"   • Bucket: {links['bucket_url']}")
        
        print(f"\n🔑 File Hash: {file_hash}")
        
        if 'metadata' in result:
            print("\n📋 Metadata:")
            for key, value in result['metadata'].items():
                if isinstance(value, datetime):
                    value = value.strftime('%Y-%m-%d %H:%M:%S UTC')
                print(f"   • {key}: {value}")
    
    elif status == 'duplicate':
        file_hash = result.get('file_hash', 'N/A')
        print(f"⚠️ File already exists with hash: {file_hash}")
        
        gcs = GCSManager()
        try:
            metadata = gcs.get_file_metadata(file_hash)
            if metadata and 'gcs_uri' in metadata:
                gcs_uri = metadata['gcs_uri']
                links = get_gcs_link(gcs_uri, project_id)
                print(f"\n📂 Original File Location:")
                print(f"   • GCS URI: {gcs_uri}")
                print(f"   • Console Link: {links['console_url']}")
                print(f"   • Public URL: {links['public_url']}")
                print(f"   • Bucket: {links['bucket_url']}")
            else:
                print("\n⚠️ Could not retrieve original file location")
        except Exception as e:
            logger.error(f"Error retrieving metadata for duplicate file: {str(e)}")
            print("\n⚠️ Could not retrieve original file location")
    
    elif status == 'error':
        print(f"❌ Error: {result.get('message', 'Unknown error')}")

class TestGCSUpload:
    """Test suite for GCS upload functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        print_header("Test Setup")
        
        # Initialize GCS manager
        print("\n🔧 Initializing GCS manager...")
        self.gcs_manager = GCSManager()
        self.project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
        
        if not self.project_id:
            raise ValueError("❌ GOOGLE_CLOUD_PROJECT not set in environment")
        
        print(f"📦 Project ID: {self.project_id}")
        print(f"🪣 Bucket: {self.gcs_manager.bucket_name}")
        
        # Verify test PDF exists
        if not os.path.exists(TEST_PDF_PATH):
            raise FileNotFoundError(f"❌ Test PDF not found: {TEST_PDF_PATH}")
        
        file_size = os.path.getsize(TEST_PDF_PATH) / (1024 * 1024)  # Convert to MB
        print(f"\n📄 Test PDF Information:")
        print(f"   • Path: {TEST_PDF_PATH}")
        print(f"   • Size: {file_size:.2f} MB")
        print(f"   • Last Modified: {datetime.fromtimestamp(os.path.getmtime(TEST_PDF_PATH)).strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Clean environment
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

    def test_pdf_upload(self):
        """Test uploading the actual PDF file"""
        print_header("PDF Upload Test")
        
        try:
            # Upload file
            print("\n📤 Starting upload...")
            result = self.gcs_manager.upload_file(TEST_PDF_PATH)
            log_upload_result(result, self.project_id)
            
            # Verify upload success
            assert result['status'] == 'success', f"Upload failed with result: {result}"
            assert 'gcs_uri' in result, "GCS URI missing from result"
            assert 'file_hash' in result, "File hash missing from result"
            
            # Verify metadata storage
            metadata = self.gcs_manager.get_file_metadata(result['file_hash'])
            assert metadata is not None, "Metadata not found"
            assert metadata['status'] == 'uploaded', "Incorrect metadata status"
            
            print("\n✅ Upload test passed successfully!")
            
        except Exception as e:
            logger.error(f"Upload test failed: {str(e)}")
            print(f"\n❌ Upload test failed: {str(e)}")
            raise

    def test_duplicate_upload(self):
        """Test uploading the same PDF file twice"""
        print_header("Duplicate Upload Test")
        
        try:
            # First upload
            print("\n📤 First upload attempt...")
            result1 = self.gcs_manager.upload_file(TEST_PDF_PATH)
            log_upload_result(result1, self.project_id)
            assert result1['status'] == 'success', f"First upload failed: {result1}"
            
            # Second upload (should detect duplicate)
            print("\n📤 Second upload attempt (should detect duplicate)...")
            result2 = self.gcs_manager.upload_file(TEST_PDF_PATH)
            log_upload_result(result2, self.project_id)
            assert result2['status'] == 'duplicate', f"Duplicate not detected: {result2}"
            assert result2['file_hash'] == result1['file_hash'], "Hash mismatch"
            
            print("\n✅ Duplicate detection test passed successfully!")
            
        except Exception as e:
            logger.error(f"Duplicate test failed: {str(e)}")
            print(f"\n❌ Duplicate test failed: {str(e)}")
            raise

    def test_list_files(self):
        """Test listing files in GCS bucket"""
        print_header("List Files Test")
        
        try:
            # Upload test file first
            result = self.gcs_manager.upload_file(TEST_PDF_PATH)
            log_upload_result(result, self.project_id)
            assert result['status'] == 'success', "Upload should succeed"
            
            # List files
            print("\n📁 Listing bucket contents...")
            files = self.gcs_manager.list_files()
            assert isinstance(files, list), "Should return a list"
            assert len(files) > 0, "Should have at least one file"
            
            print(f"\nFound {len(files)} files:")
            for file in files:
                gcs_uri = f"gs://{self.gcs_manager.bucket_name}/{file}"
                links = get_gcs_link(gcs_uri, self.project_id)
                print(f"   • {file}")
                print(f"     URI: {gcs_uri}")
                print(f"     Console: {links['console_url']}")
                print(f"     Public: {links['public_url']}")
            
            print("\n✅ List files test passed successfully!")
            
        except Exception as e:
            logger.error(f"List files test failed: {str(e)}")
            print(f"\n❌ List files test failed: {str(e)}")
            raise

def run_tests():
    """Run all tests with proper setup"""
    print_header("GCS Upload Test Suite")
    
    if not os.getenv('GOOGLE_CLOUD_PROJECT'):
        logger.error("❌ GOOGLE_CLOUD_PROJECT not set in environment")
        return
    
    test = TestGCSUpload()
    test.setup_method()  # Initialize the test environment
    
    tests = [
        (test.test_pdf_upload, "PDF Upload"),
        (test.test_duplicate_upload, "Duplicate Upload"),
        (test.test_list_files, "List Files")
    ]
    
    failed_tests = []
    
    for test_func, test_name in tests:
        print(f"\n{'─'*80}")
        print(f"Running: {test_name}")
        try:
            test_func()
            print(f"\n✅ {test_name} passed")
        except Exception as e:
            logger.error(f"Test failed: {str(e)}")
            print(f"\n❌ {test_name} failed: {str(e)}")
            failed_tests.append(test_name)
        finally:
            test.clean_environment()
    
    print_header("Test Summary")
    if failed_tests:
        print("\n❌ Some tests failed:")
        for failed_test in failed_tests:
            print(f"  - {failed_test}")
        sys.exit(1)
    else:
        print("\n🎉 All tests passed successfully!")
    
    # Clean up at the end
    test.teardown_method()

def cleanup_test_environment():
    """Standalone function to clean up test environment"""
    print_header("Test Environment Cleanup")
    test = TestGCSUpload()
    test.setup_method()
    test.clean_environment()
    print("\n✨ Cleanup completed")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--cleanup':
        cleanup_test_environment()
    else:
        run_tests() 