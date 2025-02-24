import os
import sys
import time
from google.cloud import storage
from google.cloud import firestore
from dotenv import load_dotenv
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def print_header(title):
    """Print a formatted header"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")

def print_result(service, status, details=None):
    """Print a formatted test result"""
    status_icon = "✅" if status else "❌"
    print(f"{status_icon} {service}: {'Success' if status else 'Failed'}")
    if details:
        for line in details:
            print(f"   • {line}")

def test_gcp_connection():
    """Test GCP connectivity and all services used in the project"""
    print_header("FinSight AI - GCP Connection Test")
    
    # Load environment variables
    load_dotenv()
    
    # Configuration
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT', 'finsight-ai-nguyen')
    credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    bucket_name = os.getenv('GCP_STORAGE_BUCKET', 'finsight-reports-bucket')
    
    # Print configuration
    print("\n📋 Configuration:")
    print(f"   • Project ID: {project_id}")
    print(f"   • Credentials: {credentials_path}")
    print(f"   • Storage Bucket: {bucket_name}")
    
    # Test results
    all_tests_passed = True
    
    # 1. Test GCP Authentication
    try:
        print("\n🔑 Testing Authentication...")
        storage_client = storage.Client()
        auth_success = True
        print_result("Authentication", True, [f"Authenticated as {storage_client.project}"])
    except Exception as e:
        auth_success = False
        all_tests_passed = False
        print_result("Authentication", False, [str(e)])
    
    if not auth_success:
        print("\n❌ Authentication failed. Cannot proceed with other tests.")
        return
    
    # 2. Test Cloud Storage
    try:
        print("\n🗄️ Testing Cloud Storage...")
        # Check bucket exists
        bucket = storage_client.bucket(bucket_name)
        bucket_exists = bucket.exists()
        
        if bucket_exists:
            # List files in bucket
            blobs = list(bucket.list_blobs())
            print_result("Cloud Storage", True, [
                f"Bucket '{bucket_name}' exists",
                f"Contains {len(blobs)} files"
            ])
            
            # Test write access by creating a test file
            test_blob = bucket.blob(f"test/connection_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            test_blob.upload_from_string("GCP Connection Test")
            print_result("Storage Write Access", True, ["Successfully created test file"])
            
            # Clean up test file
            test_blob.delete()
            print_result("Storage Delete Access", True, ["Successfully deleted test file"])
        else:
            all_tests_passed = False
            print_result("Cloud Storage", False, [f"Bucket '{bucket_name}' does not exist"])
    except Exception as e:
        all_tests_passed = False
        print_result("Cloud Storage", False, [str(e)])
    
    # 3. Test Firestore
    try:
        print("\n📊 Testing Firestore...")
        db = firestore.Client()
        
        # Test collection access
        collection_name = "pdf_metadata"
        docs = list(db.collection(collection_name).limit(5).stream())
        
        # Test write access
        test_doc_ref = db.collection(collection_name).document(f"connection_test_{int(time.time())}")
        test_doc_ref.set({
            "test": True,
            "timestamp": datetime.now(),
            "description": "GCP Connection Test"
        })
        
        # Test read access
        test_doc = test_doc_ref.get()
        
        # Clean up
        test_doc_ref.delete()
        
        print_result("Firestore", True, [
            f"Connected to Firestore database",
            f"Collection '{collection_name}' has {len(docs)} documents",
            "Successfully tested write and read operations"
        ])
    except Exception as e:
        all_tests_passed = False
        print_result("Firestore", False, [str(e)])
    
    # Summary
    print_header("Test Summary")
    if all_tests_passed:
        print("✅ All GCP services are properly configured and accessible!")
    else:
        print("❌ Some tests failed. Please check the details above.")
        print("\n🔍 Troubleshooting Tips:")
        print("   • Verify GOOGLE_APPLICATION_CREDENTIALS path is correct")
        print("   • Ensure service account has necessary permissions")
        print("   • Check if resources (buckets, collections) exist")
        print("   • Verify project ID matches your GCP project")

if __name__ == "__main__":
    test_gcp_connection() 