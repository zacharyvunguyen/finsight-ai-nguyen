#!/usr/bin/env python3
"""
Test script for directly querying the Pinecone index without using LlamaIndex.
This script:
1. Connects to Pinecone
2. Generates embeddings for a query using OpenAI
3. Directly queries Pinecone with the embeddings
4. Formats and displays the results
"""

import os
import sys
import json
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

# Add the parent directory to the path so we can import from scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import common utility functions
from scripts.utils.common import print_header, print_result

# Import functions from the setup_rag_pipeline script with updated paths
from scripts.setup.setup_rag_pipeline import (
    verify_pinecone_setup,
    verify_openai_api
)

def main():
    """Main function to test directly querying Pinecone."""
    print_header("Testing Direct Pinecone Query")
    
    # Load environment variables
    load_dotenv()
    
    # Verify OpenAI API
    if not verify_openai_api():
        print_result(False, "OpenAI API verification failed")
        return
    
    # Verify Pinecone setup
    index_name = os.getenv("PINECONE_INDEX_NAME")
    if not verify_pinecone_setup(index_name):
        print_result(False, "Pinecone setup verification failed")
        return
    
    try:
        # Initialize OpenAI client
        print_header("Initializing OpenAI Client")
        openai_client = OpenAI()
        print_result(True, "OpenAI client initialized")
        
        # Initialize Pinecone client
        print_header("Initializing Pinecone Client")
        pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        pinecone_index = pinecone_client.Index(index_name)
        print_result(True, "Pinecone client initialized")
        
        # Get index stats
        stats = pinecone_index.describe_index_stats()
        print(f"Index stats: {stats}")
        
        # Generate embeddings for queries
        print_header("Generating Embeddings for Queries")
        
        # Define queries
        queries = [
            "What is GROSS PATIENT SERVICE REVENUE for inpatient and outpatient?",
            "GROSS PATIENT SERVICE REVENUE",
            "inpatient and outpatient revenue"
        ]
        
        for query in queries:
            print(f"\nProcessing query: '{query}'")
            
            # Generate embeddings
            response = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=query,
                encoding_format="float"
            )
            query_embedding = response.data[0].embedding
            
            # Query Pinecone
            print(f"Querying Pinecone with embedding for: '{query}'")
            results = pinecone_index.query(
                vector=query_embedding,
                top_k=5,
                namespace="doc_b7712a3a4e",
                include_metadata=True
            )
            
            # Process and display results
            print(f"Found {len(results['matches'])} matches:")
            
            for i, match in enumerate(results['matches']):
                print(f"\nMatch {i+1} (Score: {match['score']}):")
                
                # Extract text from metadata
                metadata = match['metadata']
                
                # Try to extract the text content
                if '_node_content' in metadata:
                    try:
                        node_content = json.loads(metadata['_node_content'])
                        if 'text' in node_content:
                            print(f"Text: {node_content['text'][:200]}...")
                        elif 'obj' in node_content and '__data__' in node_content['obj'] and 'text' in node_content['obj']['__data__']:
                            print(f"Text: {node_content['obj']['__data__']['text'][:200]}...")
                    except json.JSONDecodeError:
                        print("Could not parse node content")
                
                # Print content preview if available
                if 'content_preview' in metadata:
                    print(f"Content Preview: {metadata['content_preview']}")
                
                # Print file name if available
                if 'file_name' in metadata:
                    print(f"File Name: {metadata['file_name']}")
                elif 'fn' in metadata:
                    print(f"File Name: {metadata['fn']}")
        
        print_result(True, "Direct Pinecone queries executed successfully")
        
    except Exception as e:
        print_result(False, f"Error during direct Pinecone query: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 