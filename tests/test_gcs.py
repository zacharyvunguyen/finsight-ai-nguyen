import pytest
from app.utils.gcs import GCSManager
import os
import tempfile

@pytest.fixture
def gcs_manager():
    return GCSManager()

@pytest.fixture
def sample_pdf():
    # Create a temporary PDF file for testing
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
        f.write(b'Test PDF content')
        return f.name

def test_compute_file_hash(gcs_manager, sample_pdf):
    """Test hash computation."""
    hash1 = gcs_manager.compute_file_hash(sample_pdf)
    hash2 = gcs_manager.compute_file_hash(sample_pdf)
    assert hash1 == hash2
    assert len(hash1) == 64  # SHA-256 hash length

def test_upload_and_duplicate_detection(gcs_manager, sample_pdf):
    """Test file upload and duplicate detection."""
    # First upload
    result1 = gcs_manager.upload_file(sample_pdf, 'test.pdf')
    assert result1['status'] == 'success'
    assert 'gcs_uri' in result1

    # Try uploading same file
    result2 = gcs_manager.upload_file(sample_pdf, 'test.pdf')
    assert result2['status'] == 'duplicate'

def test_list_files(gcs_manager):
    """Test listing files."""
    files = gcs_manager.list_files()
    assert isinstance(files, list)

def test_get_file_metadata(gcs_manager, sample_pdf):
    """Test metadata retrieval."""
    # Upload file first
    result = gcs_manager.upload_file(sample_pdf, 'test.pdf')
    
    # Get metadata
    metadata = gcs_manager.get_file_metadata(result['file_hash'])
    assert metadata is not None
    assert 'original_filename' in metadata
    assert 'upload_timestamp' in metadata 