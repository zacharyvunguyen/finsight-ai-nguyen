#!/usr/bin/env python3
"""
Test script for querying the Pinecone index with a specific question.
This script:
1. Connects to Pinecone
2. Creates a query engine
3. Executes a specific query about gross patient service revenue
"""

import os
import sys
from dotenv import load_dotenv

# Add the parent directory to the path so we can import from scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import common utility functions
from scripts.utils.common import print_header, print_result

# Import functions from the setup_rag_pipeline script with updated paths
from scripts.setup.setup_rag_pipeline import (
    load_environment,
    setup_llama_index,
    setup_pinecone_vector_store,
    verify_pinecone_setup,
    verify_openai_api
)

def main():
    """Main function to test querying the Pinecone index."""
    print_header("Testing Pinecone Query")
    
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
        # Set up LlamaIndex
        print_header("Setting up LlamaIndex")
        settings = setup_llama_index()
        print_result(True, "LlamaIndex setup successful")
        
        # Set up Pinecone vector store
        print_header("Setting up Pinecone Vector Store")
        vector_store = setup_pinecone_vector_store(index_name)
        print_result(True, f"Connected to Pinecone index '{index_name}'")
        
        # Create vector index from existing Pinecone index
        print_header("Creating Vector Index from Pinecone")
        from llama_index.core import VectorStoreIndex
        index = VectorStoreIndex.from_vector_store(vector_store)
        print_result(True, "Vector index created from existing Pinecone index")
        
        # Debug: Retrieve nodes directly from vector store
        print_header("Debugging: Retrieving Nodes from Vector Store")
        
        # Retrieve nodes from vector store
        retriever = index.as_retriever(similarity_top_k=5)
        nodes = retriever.retrieve("financial information")
        
        print(f"Retrieved {len(nodes)} nodes for query: 'financial information'")
        
        if len(nodes) > 0:
            print("\nNode details:")
            for i, node in enumerate(nodes):
                print(f"\nNode {i+1}:")
                print(f"Score: {node.score}")
                print(f"Text: {node.node.text[:100]}...")
                print(f"Metadata: {node.node.metadata}")
        else:
            print("No nodes retrieved. This suggests an issue with the vector store or the embeddings.")
            
            # Try to get stats from Pinecone
            print("\nPinecone index stats:")
            pinecone_index = vector_store._pinecone_index
            stats = pinecone_index.describe_index_stats()
            print(f"Total vectors: {stats.get('total_vector_count', 0)}")
            print(f"Namespaces: {stats.get('namespaces', {})}")
            
            # Try a direct query to Pinecone
            print("\nTrying direct query to Pinecone:")
            try:
                # Import OpenAI for embeddings
                from openai import OpenAI
                
                # Create OpenAI client
                client = OpenAI()
                
                # Generate embeddings for a query
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input="financial information",
                    encoding_format="float"
                )
                query_embedding = response.data[0].embedding
                
                direct_results = pinecone_index.query(
                    vector=query_embedding,
                    top_k=5,
                    namespace="doc_b7712a3a4e",
                    include_metadata=True
                )
                print(f"Direct query results: {direct_results}")
            except Exception as e:
                print(f"Error with direct Pinecone query: {e}")
        
        # Create a simple query engine
        print_header("Creating Query Engine")
        query_engine = index.as_query_engine(similarity_top_k=10)
        print_result(True, "Query engine created")
        
        # Try a more general query first
        general_query = "What information is available in this document?"
        print_header(f"Executing General Query: {general_query}")
        
        # Execute general query
        general_response = query_engine.query(general_query)
        
        print("\nGeneral Query Response:")
        print(f"{general_response}")
        
        # Execute specific query
        specific_query = "What is GROSS PATIENT SERVICE REVENUE for inpatient and outpatient?"
        print_header(f"Executing Specific Query: {specific_query}")
        
        # Execute specific query
        specific_response = query_engine.query(specific_query)
        
        print("\nSpecific Query Response:")
        print(f"{specific_response}")
        
        print_result(True, "Queries executed successfully")
        
    except Exception as e:
        print_result(False, f"Error during query execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 