#!/usr/bin/env python3
"""
Test suite for external services (GCP and Pinecone).
This test suite:
1. Tests GCP connectivity and services
2. Tests Pinecone connectivity and operations
3. Provides detailed error messages and troubleshooting steps
"""

import os
import sys
import pytest
from dotenv import load_dotenv
from google.cloud import storage
from google.cloud import bigquery
from pinecone.grpc import PineconeGRPC

def print_header(message):
    """Print a formatted header message."""
    print("\n" + "="*80)
    print(f" {message}")
    print("="*80)

def print_result(success, message):
    """Print a formatted result message."""
    status = "✅" if success else "❌"
    print(f"{status} {message}")

class TestExternalServices:
    @classmethod
    def setup_class(cls):
        """Set up test environment."""
        load_dotenv()
        cls.required_vars = {
            'GCP': [
                'GOOGLE_CLOUD_PROJECT',
                'GCP_STORAGE_BUCKET',
                'BIGQUERY_DATASET',
                'GOOGLE_APPLICATION_CREDENTIALS'
            ],
            'Pinecone': [
                'PINECONE_API_KEY',
                'PINECONE_INDEX_NAME',
                'PINECONE_DIMENSION'
            ]
        }

    def test_environment_variables(self):
        """Test that all required environment variables are set."""
        print_header("Testing Environment Variables")
        
        for service, vars in self.required_vars.items():
            missing = [var for var in vars if not os.getenv(var)]
            if missing:
                print_result(False, f"Missing {service} variables: {', '.join(missing)}")
                pytest.fail(f"Missing required environment variables for {service}")
            print_result(True, f"All {service} environment variables are set")

    def test_gcp_storage(self):
        """Test Google Cloud Storage connectivity."""
        print_header("Testing Google Cloud Storage")
        
        try:
            client = storage.Client()
            bucket = client.bucket(os.getenv('GCP_STORAGE_BUCKET'))
            
            # Test bucket existence
            if bucket.exists():
                print_result(True, f"Successfully connected to bucket: {bucket.name}")
            else:
                print_result(False, "Bucket does not exist")
                pytest.fail("GCS bucket not found")
            
            # Test write permissions
            test_blob = bucket.blob('test/test_file.txt')
            test_blob.upload_from_string('test content')
            print_result(True, "Successfully wrote to bucket")
            
            # Clean up test file
            test_blob.delete()
            print_result(True, "Successfully cleaned up test file")
            
        except Exception as e:
            print_result(False, f"GCS test failed: {str(e)}")
            pytest.fail(f"GCS test failed: {str(e)}")

    def test_bigquery(self):
        """Test BigQuery connectivity."""
        print_header("Testing BigQuery")
        
        try:
            client = bigquery.Client()
            dataset_ref = client.dataset(os.getenv('BIGQUERY_DATASET'))
            
            # Test dataset access
            dataset = client.get_dataset(dataset_ref)
            print_result(True, f"Successfully accessed dataset: {dataset.dataset_id}")
            
            # Test query execution
            query = "SELECT 1"
            query_job = client.query(query)
            results = query_job.result()
            print_result(True, "Successfully executed test query")
            
        except Exception as e:
            print_result(False, f"BigQuery test failed: {str(e)}")
            pytest.fail(f"BigQuery test failed: {str(e)}")

    def test_pinecone(self):
        """Test Pinecone connectivity and operations."""
        print_header("Testing Pinecone")
        
        try:
            # Initialize client
            client = PineconeGRPC(api_key=os.getenv('PINECONE_API_KEY'))
            print_result(True, "Successfully initialized Pinecone client")
            
            # Get index
            index_name = os.getenv('PINECONE_INDEX_NAME')
            index = client.Index(index_name)
            
            # Test index stats
            stats = index.describe_index_stats()
            print_result(True, f"Successfully retrieved index stats: {stats}")
            
            # Test vector operations
            dimension = int(os.getenv('PINECONE_DIMENSION'))
            test_vector = [0.0] * dimension
            test_id = "test_vector"
            
            # Upsert test vector
            index.upsert(vectors=[(test_id, test_vector)])
            print_result(True, "Successfully upserted test vector")
            
            # Query test vector
            results = index.query(vector=test_vector, top_k=1)
            print_result(True, "Successfully queried index")
            
            # Delete test vector
            index.delete(ids=[test_id])
            print_result(True, "Successfully deleted test vector")
            
        except Exception as e:
            print_result(False, f"Pinecone test failed: {str(e)}")
            pytest.fail(f"Pinecone test failed: {str(e)}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 