#!/usr/bin/env python3
"""
Simple script to test processing a PDF from GCS.
This script:
1. Downloads a PDF from GCS
2. Uses LlamaParse to extract text
3. Prints basic information about the document
"""

import os
import sys
import argparse
from dotenv import load_dotenv
from pathlib import Path
from llama_cloud_services import LlamaParse
from google.cloud import storage

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
        'LLAMA_CLOUD_API_KEY',
        'GCP_STORAGE_BUCKET',
        'GOOGLE_APPLICATION_CREDENTIALS'
    ]
    
    env = {}
    missing_vars = []
    
    for var in required_vars:
        value = os.environ.get(var)
        if not value:
            missing_vars.append(var)
        else:
            env[var] = value
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    return env

def download_pdf_from_gcs(bucket_name, gcs_path, local_path):
    """Download a PDF file from Google Cloud Storage."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Initialize GCS client
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        
        # Download the file
        blob.download_to_filename(local_path)
        print_result(True, f"Downloaded {gcs_path} to {local_path}")
        
        return local_path
    except Exception as e:
        print_result(False, f"Error downloading file from GCS: {str(e)}")
        raise

def process_pdf_with_llamaparse(pdf_path):
    """Process a PDF file with LlamaParse and return the extracted text."""
    try:
        # Initialize LlamaParse client
        parser = LlamaParse(result_type="markdown", api_key=os.environ.get("LLAMA_CLOUD_API_KEY"))
        
        # Process the PDF
        print_header(f"Processing {pdf_path} with LlamaParse")
        
        # Parse the document
        documents = parser.load_data(pdf_path)
        
        print(f"Number of documents: {len(documents)}")
        
        # Print text statistics for each document
        for i, doc in enumerate(documents):
            print(f"\nDocument {i+1}:")
            print(f"Metadata: {doc.metadata}")
            text_length = len(doc.text)
            print(f"Text length: {text_length} characters")
            print(f"First 500 characters: {doc.text[:500]}...")
        
        return documents
    except Exception as e:
        print_result(False, f"Error processing PDF with LlamaParse: {str(e)}")
        raise

def main():
    """Main function to parse arguments and process the PDF."""
    parser = argparse.ArgumentParser(description="Test processing a PDF from GCS with LlamaParse")
    parser.add_argument("--gcs-file", required=True, help="Path to a PDF file in GCS to process")
    args = parser.parse_args()
    
    try:
        # Load environment variables
        env = load_environment()
        bucket_name = env['GCP_STORAGE_BUCKET']
        
        # Download PDF from GCS
        local_path = f"data/temp/{os.path.basename(args.gcs_file)}"
        pdf_path = download_pdf_from_gcs(bucket_name, args.gcs_file, local_path)
        
        # Process PDF with LlamaParse
        documents = process_pdf_with_llamaparse(pdf_path)
        
        print_header("Processing Complete")
        print_result(True, "PDF successfully processed with LlamaParse")
        
    except Exception as e:
        print_result(False, f"Error in main function: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 