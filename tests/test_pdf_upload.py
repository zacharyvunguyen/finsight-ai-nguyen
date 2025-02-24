import os
import sys
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

def get_file_link(file_path: str) -> str:
    """Convert file path to clickable link"""
    abs_path = os.path.abspath(file_path)
    return f"file://{abs_path}"

def get_gcs_link(gcs_uri: str, project_id: str) -> str:
    """Convert GCS URI to clickable console link"""
    # Remove gs:// prefix and split into bucket and path
    bucket_path = gcs_uri.replace('gs://', '')
    bucket_name = bucket_path.split('/')[0]
    file_path = '/'.join(bucket_path.split('/')[1:])
    
    # Format GCS console URL
    base_url = "https://console.cloud.google.com/storage/browser"
    return f"{base_url}/{bucket_name}/{file_path}?project={project_id}"

def log_section(title: str) -> None:
    """Print a section header with emoji"""
    print(f"\n{'='*80}")
    print(f"📌 {title}")
    print(f"{'='*80}\n")

def log_result(result: Dict[str, Any], project_id: str) -> None:
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
        
        # Generate direct GCS console link
        console_link = get_gcs_link(gcs_uri, project_id)
        print(f"   • Console: {console_link}")
        
        # Generate bucket link
        bucket_name = gcs_uri.split('/')[2]  # gs://bucket-name/path -> bucket-name
        bucket_link = f"https://console.cloud.google.com/storage/browser/{bucket_name}?project={project_id}"
        print(f"   • Bucket: {bucket_link}")
        
        print(f"🔑 File Hash: {file_hash}")
        
        if 'metadata' in result:
            print("\n📋 Metadata:")
            for key, value in result['metadata'].items():
                if isinstance(value, datetime):
                    value = value.strftime('%Y-%m-%d %H:%M:%S UTC')
                print(f"   • {key}: {value}")
    
    elif status == 'duplicate':
        print(f"⚠️ File already exists with hash: {result.get('file_hash', 'N/A')}")
        if 'gcs_uri' in result:
            console_link = get_gcs_link(result['gcs_uri'], project_id)
            print(f"   • Original file: {result['gcs_uri']}")
            print(f"   • Console link: {console_link}")
    
    elif status == 'error':
        print(f"❌ Error: {result.get('message', 'Unknown error')}")

def test_single_pdf_upload():
    """Test uploading a single PDF file"""
    
    log_section("Initializing Upload Test")
    
    # Initialize GCS manager
    logger.info("🔧 Initializing GCS manager...")
    gcs = GCSManager()
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    
    if not project_id:
        logger.error("❌ GOOGLE_CLOUD_PROJECT not set in environment")
        return
    
    logger.info(f"📦 Project ID: {project_id}")
    logger.info(f"🪣 Bucket: {gcs.bucket_name}")
    
    # PDF file path
    pdf_path = "data/test/pdfs/06_09.05.24_0545_BD3_PrelimBook Proj_CY_AugFY25.pdf"
    
    if not os.path.exists(pdf_path):
        logger.error(f"❌ File not found: {pdf_path}")
        return
        
    file_size = os.path.getsize(pdf_path) / (1024 * 1024)  # Convert to MB
    file_name = os.path.basename(pdf_path)
    
    print(f"\n📄 File Information:")
    print(f"   • Name: {file_name}")
    print(f"   • Path: {pdf_path}")
    print(f"   • Link: {get_file_link(pdf_path)}")
    print(f"   • Size: {file_size:.2f} MB")
    print(f"   • Last Modified: {datetime.fromtimestamp(os.path.getmtime(pdf_path)).strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        log_section("Uploading File")
        logger.info(f"📤 Starting upload: {file_name}")
        
        # Upload file
        result = gcs.upload_file(pdf_path)
        log_result(result, project_id)
        
        if result['status'] == 'success':
            log_section("Verifying Upload")
            
            # Verify metadata
            metadata = gcs.get_file_metadata(result['file_hash'])
            print("\n📋 Stored Metadata:")
            for key, value in metadata.items():
                if isinstance(value, datetime):
                    value = value.strftime('%Y-%m-%d %H:%M:%S UTC')
                print(f"   • {key}: {value}")
            
            # List bucket contents
            files = gcs.list_files()
            print(f"\n📁 Bucket Contents ({len(files)} files):")
            for file in files:
                gcs_uri = f"gs://{gcs.bucket_name}/{file}"
                console_link = get_gcs_link(gcs_uri, project_id)
                print(f"   • {file}")
                print(f"     URI: {gcs_uri}")
                print(f"     Link: {console_link}")
            
    except Exception as e:
        logger.error(f"❌ Upload failed: {str(e)}")
        raise
    finally:
        print(f"\n{'='*80}")

if __name__ == "__main__":
    log_section("PDF Upload Test")
    test_single_pdf_upload() 