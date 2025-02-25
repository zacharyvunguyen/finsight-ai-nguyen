#!/usr/bin/env python3
"""
Script to set up and initialize Pinecone vector database.
This script:
1. Checks for required environment variables
2. Initializes Pinecone client
3. Creates index if it doesn't exist
4. Verifies index is ready for use
"""

import os
import time
from dotenv import load_dotenv
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

def load_environment():
    """Load and validate environment variables."""
    load_dotenv()
    
    required_vars = [
        'PINECONE_API_KEY',
        'PINECONE_INDEX_NAME',
        'PINECONE_CLOUD',
        'PINECONE_REGION',
        'PINECONE_DIMENSION',
        'PINECONE_METRIC'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    return {var: os.getenv(var) for var in required_vars}

def setup_pinecone():
    """Set up Pinecone index with configuration from environment."""
    print_header("Setting up Pinecone")
    
    try:
        # Load environment variables
        env = load_environment()
        print_result(True, "Environment variables loaded successfully")
        
        # Initialize Pinecone client
        client = PineconeGRPC(api_key=env['PINECONE_API_KEY'])
        print_result(True, "Pinecone client initialized")
        
        # List existing indexes
        indexes = client.list_indexes()
        index_name = env['PINECONE_INDEX_NAME']
        
        if index_name not in [index.name for index in indexes]:
            # Create index if it doesn't exist
            client.create_index(
                name=index_name,
                dimension=int(env['PINECONE_DIMENSION']),
                metric=env['PINECONE_METRIC']
            )
            print_result(True, f"Created new index: {index_name}")
            
            # Wait for index to be ready
            print("Waiting for index to be ready...")
            time.sleep(10)  # Give time for index to initialize
        else:
            print_result(True, f"Index {index_name} already exists")
        
        # Connect to index and verify
        index = client.Index(index_name)
        stats = index.describe_index_stats()
        print_result(True, f"Successfully connected to index. Current stats: {stats}")
        
        return True
        
    except Exception as e:
        print_result(False, f"Error setting up Pinecone: {str(e)}")
        return False

if __name__ == "__main__":
    setup_pinecone() 