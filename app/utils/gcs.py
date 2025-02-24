import os
import hashlib
from google.cloud import storage
from google.cloud import firestore
from datetime import datetime
from dotenv import load_dotenv
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class GCSManager:
    def __init__(self):
        """Initialize GCS manager with configuration"""
        try:
            self.bucket_name = os.getenv('GCP_STORAGE_BUCKET', 'finsight-reports-bucket')
            if not self.bucket_name:
                raise ValueError("GCP_STORAGE_BUCKET not set in environment")
            
            # Get the service account key file path
            # First try the environment variable
            key_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            
            # If not set, try the default location
            if not key_path:
                # Try different possible locations for the key file
                possible_paths = [
                    'config/keys/finsight-ai-nguyen-89af45b3c2c0.json',  # Project root
                    '../config/keys/finsight-ai-nguyen-89af45b3c2c0.json',  # From app directory
                    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                'config/keys/finsight-ai-nguyen-89af45b3c2c0.json')  # Absolute path
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        key_path = path
                        break
            
            if not key_path or not os.path.exists(key_path):
                raise FileNotFoundError(f"Service account key file not found. Tried: {possible_paths}")
            
            logger.info(f"Using service account key file: {key_path}")
            
            # Initialize clients with explicit credentials
            self.storage_client = storage.Client.from_service_account_json(key_path)
            self.db = firestore.Client.from_service_account_json(key_path)
            self.bucket = self.storage_client.bucket(self.bucket_name)
            
            # Ensure bucket exists
            if not self.bucket.exists():
                logger.warning(f"Bucket {self.bucket_name} does not exist, creating...")
                self.bucket = self.storage_client.create_bucket(
                    self.bucket_name,
                    location='us-east1'
                )
            
            logger.info(f"GCSManager initialized with bucket: {self.bucket_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize GCSManager: {str(e)}")
            raise

    def compute_file_hash(self, file_path: str) -> str:
        """Compute SHA-256 hash of a file."""
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Error computing file hash: {str(e)}")
            raise

    def check_duplicate(self, file_hash: str) -> bool:
        """Check if file hash exists in Firestore."""
        try:
            doc_ref = self.db.collection('pdf_metadata').document(file_hash)
            doc = doc_ref.get()
            # Verify document exists and has valid content
            is_duplicate = doc.exists and doc.to_dict() is not None
            if is_duplicate:
                metadata = doc.to_dict()
                # Verify the file exists in GCS
                blob = self.bucket.blob(metadata.get('gcs_uri', '').replace(f'gs://{self.bucket_name}/', ''))
                is_duplicate = blob.exists()
            
            logger.info(f"Checking duplicate for hash {file_hash}: {'exists' if is_duplicate else 'not found'}")
            return is_duplicate
        except Exception as e:
            logger.error(f"Error checking for duplicate: {str(e)}")
            return False

    def store_metadata(self, file_hash: str, original_filename: str, gcs_uri: str):
        """Store file metadata in Firestore."""
        try:
            doc_ref = self.db.collection('pdf_metadata').document(file_hash)
            metadata = {
                'original_filename': original_filename,
                'gcs_uri': gcs_uri,
                'upload_timestamp': datetime.utcnow(),
                'status': 'uploaded'
            }
            doc_ref.set(metadata)
            logger.info(f"Metadata stored for file: {original_filename}")
            return metadata
        except Exception as e:
            logger.error(f"Error storing metadata: {str(e)}")
            raise

    def upload_file(self, file_path: str, original_filename: str = None) -> dict:
        """Upload file to GCS with duplicate detection."""
        try:
            logger.info(f"Starting upload process for file: {file_path}")
            
            # Compute file hash
            file_hash = self.compute_file_hash(file_path)
            logger.info(f"Computed file hash: {file_hash}")
            
            # Check for duplicates
            if self.check_duplicate(file_hash):
                logger.info(f"Duplicate file detected with hash: {file_hash}")
                return {
                    'status': 'duplicate',
                    'message': 'File already exists',
                    'file_hash': file_hash
                }

            # Generate GCS path
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            original_filename = original_filename or os.path.basename(file_path)
            gcs_filename = f"{timestamp}_{original_filename}"
            blob = self.bucket.blob(f"uploads/{gcs_filename}")
            
            # Upload file
            logger.info(f"Uploading file to GCS: {gcs_filename}")
            blob.upload_from_filename(file_path)
            gcs_uri = f"gs://{self.bucket_name}/uploads/{gcs_filename}"
            logger.info(f"File uploaded successfully to: {gcs_uri}")

            # Store metadata
            metadata = self.store_metadata(file_hash, original_filename, gcs_uri)
            
            return {
                'status': 'success',
                'message': 'File uploaded successfully',
                'file_hash': file_hash,
                'gcs_uri': gcs_uri,
                'metadata': metadata
            }

        except Exception as e:
            error_msg = f"Upload failed: {str(e)}"
            logger.error(error_msg)
            return {
                'status': 'error',
                'message': error_msg
            }

    def list_files(self) -> list:
        """List all PDF files in the bucket."""
        try:
            files = [blob.name for blob in self.bucket.list_blobs(prefix='uploads/')]
            logger.info(f"Listed {len(files)} files from bucket")
            return files
        except Exception as e:
            logger.error(f"Error listing files: {str(e)}")
            raise

    def get_file_metadata(self, file_hash: str) -> dict:
        """Get file metadata from Firestore."""
        try:
            doc_ref = self.db.collection('pdf_metadata').document(file_hash)
            doc = doc_ref.get()
            metadata = doc.to_dict() if doc.exists else None
            logger.info(f"Retrieved metadata for hash {file_hash}: {'found' if metadata else 'not found'}")
            return metadata
        except Exception as e:
            logger.error(f"Error retrieving metadata: {str(e)}")
            raise

    def cleanup_test_data(self, file_hash: str = None):
        """Clean up test data from Firestore and GCS."""
        try:
            # Clean up GCS files first
            logger.info("Starting cleanup of GCS files...")
            blobs = list(self.bucket.list_blobs(prefix='uploads/'))
            for blob in blobs:
                try:
                    blob.delete()
                    logger.info(f"Deleted GCS file: {blob.name}")
                except Exception as e:
                    logger.warning(f"Failed to delete blob {blob.name}: {str(e)}")

            # Then clean up Firestore
            logger.info("Starting cleanup of Firestore documents...")
            if file_hash:
                # Delete specific document
                doc_ref = self.db.collection('pdf_metadata').document(file_hash)
                if doc_ref.get().exists:
                    doc_ref.delete()
                    logger.info(f"Deleted Firestore document for hash: {file_hash}")
            else:
                # Delete all documents
                batch = self.db.batch()
                docs = list(self.db.collection('pdf_metadata').stream())
                for doc in docs:
                    batch.delete(doc.reference)
                    logger.info(f"Queued deletion of document: {doc.id}")
                if docs:
                    batch.commit()
                    logger.info(f"Deleted {len(docs)} documents from Firestore")

            # Verify cleanup
            remaining_docs = list(self.db.collection('pdf_metadata').stream())
            remaining_blobs = list(self.bucket.list_blobs(prefix='uploads/'))
            logger.info(f"Cleanup verification - Remaining: {len(remaining_docs)} docs, {len(remaining_blobs)} files")

        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            raise
