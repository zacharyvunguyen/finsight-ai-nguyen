# Quick Start Guide

## 1. Environment Setup

### 1.1 Create `.env` file
```bash
# API Keys
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
GOOGLE_CLOUD_PROJECT=your_gcp_project_id
LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key

# Database Configuration
BIGQUERY_DATASET=financial_data
PINECONE_INDEX_NAME=financial-reports

# Pinecone Configuration
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
PINECONE_DIMENSION=1536
PINECONE_METRIC=cosine

# Application Settings
DEBUG_MODE=True
PDF_STORAGE_PATH=./data/raw
PROCESSED_DATA_PATH=./data/processed

# Google Cloud Configuration
GCP_STORAGE_BUCKET=your_gcs_bucket_name
GOOGLE_APPLICATION_CREDENTIALS=config/keys/your-service-account-key.json
```

### 1.2 Directory Structure
```bash
mkdir -p data/test/pdfs  # For test PDFs
mkdir -p data/raw        # For raw PDFs
mkdir -p data/processed  # For processed data
mkdir -p config/keys     # For API keys and credentials
```

### 1.3 Required Files
1. Place GCP service account key in `config/keys/`
2. Place test PDF in `data/test/pdfs/`

## 2. Test Connections

### 2.1 Test GCP Connection
```bash
python tests/test_gcp_connection.py
```
Verifies:
- GCP Authentication
- Cloud Storage access
- Firestore access

### 2.2 Test Pinecone Connection
```bash
python tests/test_pinecone_connection.py
```
Verifies:
- Pinecone authentication
- Index existence
- Vector operations

### 2.3 Test GCS Upload
```bash
python tests/test_gcs_upload.py
```
Verifies:
- PDF upload functionality
- Duplicate detection
- File listing
- Metadata storage

## 3. Cleanup Commands

### 3.1 Clean GCS Test Environment
```bash
python tests/test_gcs_upload.py --cleanup
```

### 3.2 Clean Pinecone Test Environment
```bash
python tests/test_pinecone_connection.py --cleanup
```

## 4. Common Issues

### 4.1 GCP Issues
- Verify `GOOGLE_APPLICATION_CREDENTIALS` points to valid key file
- Ensure service account has required permissions
- Check if bucket exists and is accessible

### 4.2 Pinecone Issues
- Install correct package: `pip install "pinecone[grpc]"`
- Verify index exists in specified region
- Check API key has necessary permissions

### 4.3 File Issues
- Ensure test PDF exists in correct location
- Verify file permissions
- Check for sufficient disk space

## 5. Required Python Packages
```bash
pip install google-cloud-storage
pip install google-cloud-firestore
pip install "pinecone[grpc]"
pip install python-dotenv
pip install pytest
```
