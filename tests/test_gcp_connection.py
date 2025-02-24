import os
import sys
from google.cloud import storage
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_gcp_connection():
    """Test GCP connectivity and credentials"""
    print("\n🔍 Testing GCP Connection...")
    
    # Load environment variables
    load_dotenv()
    
    # Print current configuration
    print(f"\nCurrent Configuration:")
    print(f"- GOOGLE_CLOUD_PROJECT: {os.getenv('GOOGLE_CLOUD_PROJECT')}")
    print(f"- GOOGLE_APPLICATION_CREDENTIALS: {os.getenv('GOOGLE_APPLICATION_CREDENTIALS')}")
    print(f"- GCP_STORAGE_BUCKET: {os.getenv('GCP_STORAGE_BUCKET')}")
    
    try:
        # Initialize storage client
        storage_client = storage.Client()
        
        # List buckets to test connection
        buckets = list(storage_client.list_buckets())
        print(f"\n✅ Successfully connected to GCP!")
        print(f"Found {len(buckets)} buckets:")
        for bucket in buckets:
            print(f"- {bucket.name}")
            
        # Test specific bucket access
        bucket_name = os.getenv('GCP_STORAGE_BUCKET')
        bucket = storage_client.bucket(bucket_name)
        
        # List files in bucket
        blobs = list(bucket.list_blobs())
        print(f"\n✅ Successfully accessed bucket: {bucket_name}")
        print(f"Found {len(blobs)} files in bucket")
        
    except Exception as e:
        print(f"\n❌ Connection failed: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Verify GOOGLE_APPLICATION_CREDENTIALS path is correct")
        print("2. Ensure GCP credentials file exists and is valid")
        print("3. Check if bucket exists and you have access")
        print("4. Verify project ID is correct")

if __name__ == "__main__":
    test_gcp_connection() 