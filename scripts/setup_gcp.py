import os
from google.cloud import storage
from dotenv import load_dotenv

def setup_gcp_resources():
    """Set up required GCP resources"""
    print("\n🔧 Setting up GCP resources...")
    
    # Load environment variables
    load_dotenv()
    
    try:
        # Initialize client
        storage_client = storage.Client()
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
        
        # Create bucket if it doesn't exist
        bucket_name = os.getenv('GCP_STORAGE_BUCKET')
        if not storage_client.lookup_bucket(bucket_name):
            bucket = storage_client.create_bucket(
                bucket_name,
                project=project_id,
                location='us-east1'  # Specify the location
            )
            print(f"\n✅ Created new bucket: {bucket_name}")
            print(f"   Location: us-east1")
            print(f"   Project: {project_id}")
        else:
            print(f"\n✅ Bucket already exists: {bucket_name}")
            
        # Create folders in bucket
        bucket = storage_client.bucket(bucket_name)
        folders = ['uploads/', 'processed/']
        for folder in folders:
            blob = bucket.blob(folder)
            if not blob.exists():
                blob.upload_from_string('')
                print(f"✅ Created folder: {folder}")
                
        # Verify setup
        print("\n🔍 Verifying setup...")
        bucket = storage_client.bucket(bucket_name)
        blobs = list(bucket.list_blobs())
        print(f"✅ Successfully verified bucket access")
        print(f"   Found {len(blobs)} files/folders")
                
    except Exception as e:
        print(f"\n❌ Setup failed: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Verify your GCP credentials")
        print("2. Ensure you have sufficient permissions")
        print("3. Check if the bucket name is available")
        print("4. Verify project ID is correct")
        print(f"Current project ID: {project_id}")

if __name__ == "__main__":
    setup_gcp_resources() 