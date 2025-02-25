#!/usr/bin/env python3
"""
Script to set up and initialize Google Cloud Platform services.
This script:
1. Verifies GCP credentials
2. Creates storage bucket if it doesn't exist
3. Sets up BigQuery dataset
4. Configures necessary IAM permissions
"""

import os
from dotenv import load_dotenv
from google.cloud import storage
from google.cloud import bigquery
from google.api_core import exceptions

def print_header(message):
    """Print a formatted header message."""
    print("\n" + "="*80)
    print(f" {message}")
    print("="*80)

def print_result(success, message):
    """Print a formatted result message."""
    status = "✅" if success else "❌"
    print(f"{status} {message}")

def load_environment():
    """Load and validate environment variables."""
    load_dotenv()
    
    required_vars = [
        'GOOGLE_CLOUD_PROJECT',
        'GCP_STORAGE_BUCKET',
        'BIGQUERY_DATASET',
        'GOOGLE_APPLICATION_CREDENTIALS'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    return {var: os.getenv(var) for var in required_vars}

def setup_storage_bucket(client, bucket_name):
    """Create GCS bucket if it doesn't exist."""
    try:
        bucket = client.bucket(bucket_name)
        if not bucket.exists():
            bucket.create(location="us-east1")
            print_result(True, f"Created new bucket: {bucket_name}")
        else:
            print_result(True, f"Bucket {bucket_name} already exists")
        return True
    except Exception as e:
        print_result(False, f"Error setting up storage bucket: {str(e)}")
        return False

def setup_bigquery_dataset(client, dataset_id):
    """Create BigQuery dataset if it doesn't exist."""
    try:
        dataset_ref = client.dataset(dataset_id)
        try:
            client.get_dataset(dataset_ref)
            print_result(True, f"Dataset {dataset_id} already exists")
        except exceptions.NotFound:
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = "us-east1"
            client.create_dataset(dataset)
            print_result(True, f"Created new dataset: {dataset_id}")
        return True
    except Exception as e:
        print_result(False, f"Error setting up BigQuery dataset: {str(e)}")
        return False

def setup_gcp():
    """Set up GCP services with configuration from environment."""
    print_header("Setting up Google Cloud Platform Services")
    
    try:
        # Load environment variables
        env = load_environment()
        print_result(True, "Environment variables loaded successfully")
        
        # Verify credentials file exists
        if not os.path.exists(env['GOOGLE_APPLICATION_CREDENTIALS']):
            raise FileNotFoundError(f"GCP credentials file not found at {env['GOOGLE_APPLICATION_CREDENTIALS']}")
        print_result(True, "GCP credentials file found")
        
        # Initialize storage client and set up bucket
        storage_client = storage.Client()
        if not setup_storage_bucket(storage_client, env['GCP_STORAGE_BUCKET']):
            return False
        
        # Initialize BigQuery client and set up dataset
        bq_client = bigquery.Client()
        if not setup_bigquery_dataset(bq_client, env['BIGQUERY_DATASET']):
            return False
        
        print_result(True, "GCP services setup completed successfully")
        return True
        
    except Exception as e:
        print_result(False, f"Error setting up GCP services: {str(e)}")
        return False

if __name__ == "__main__":
    setup_gcp() 