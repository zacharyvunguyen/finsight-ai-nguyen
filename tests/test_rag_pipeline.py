#!/usr/bin/env python3
"""
Test script for the RAG pipeline.
This script:
1. Tests the connection to required services
2. Verifies document parsing with LlamaParse
3. Tests vector embedding and storage in Pinecone
4. Validates query functionality
"""

import os
import sys
import pytest
from dotenv import load_dotenv
import tempfile
import shutil
from pathlib import Path

# Add the parent directory to the path so we can import from scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import functions from the setup_rag_pipeline script
from scripts.setup_rag_pipeline import (
    print_header,
    print_result,
    load_environment,
    setup_llama_index,
    parse_document_with_llamaparse,
    create_nodes_from_documents,
    setup_pinecone_vector_store,
    create_vector_index,
    create_query_engine
)

class TestRAGPipeline:
    @classmethod
    def setup_class(cls):
        """Set up test environment."""
        print_header("Setting up test environment")
        
        # Load environment variables
        load_dotenv()
        
        # Create temporary directory for test files
        cls.temp_dir = tempfile.mkdtemp()
        print_result(True, f"Created temporary directory: {cls.temp_dir}")
        
        # Define test PDF path
        cls.test_pdf_path = os.path.join("data", "test", "pdfs", "test.pdf")
        
        # Check if test PDF exists
        if not os.path.exists(cls.test_pdf_path):
            print_result(False, f"Test PDF not found at {cls.test_pdf_path}")
            print("Please place a test PDF in the data/test/pdfs directory")
            pytest.skip("Test PDF not found")
        
        print_result(True, f"Found test PDF at {cls.test_pdf_path}")
    
    @classmethod
    def teardown_class(cls):
        """Clean up test environment."""
        print_header("Cleaning up test environment")
        
        # Remove temporary directory
        shutil.rmtree(cls.temp_dir)
        print_result(True, f"Removed temporary directory: {cls.temp_dir}")
    
    def test_environment_variables(self):
        """Test that all required environment variables are set."""
        print_header("Testing Environment Variables")
        
        required_vars = [
            'OPENAI_API_KEY',
            'LLAMA_CLOUD_API_KEY',
            'PINECONE_API_KEY',
            'PINECONE_INDEX_NAME',
            'GCP_STORAGE_BUCKET',
            'GOOGLE_APPLICATION_CREDENTIALS'
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            print_result(False, f"Missing environment variables: {', '.join(missing_vars)}")
            pytest.fail(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        print_result(True, "All required environment variables are set")
    
    def test_llama_index_setup(self):
        """Test LlamaIndex setup with GPT-4 and OpenAI embeddings."""
        print_header("Testing LlamaIndex Setup")
        
        try:
            llm, embed_model = setup_llama_index()
            assert llm is not None, "LLM should not be None"
            assert embed_model is not None, "Embedding model should not be None"
            print_result(True, "LlamaIndex setup successful")
        except Exception as e:
            print_result(False, f"LlamaIndex setup failed: {str(e)}")
            pytest.fail(f"LlamaIndex setup failed: {str(e)}")
    
    def test_llamaparse_document_parsing(self):
        """Test document parsing with LlamaParse."""
        print_header("Testing LlamaParse Document Parsing")
        
        try:
            documents = parse_document_with_llamaparse(self.test_pdf_path)
            assert documents is not None, "Documents should not be None"
            assert len(documents) > 0, "Should parse at least one document"
            print_result(True, f"LlamaParse parsed {len(documents)} documents")
        except Exception as e:
            print_result(False, f"LlamaParse parsing failed: {str(e)}")
            pytest.fail(f"LlamaParse parsing failed: {str(e)}")
    
    def test_node_creation(self):
        """Test node creation from parsed documents."""
        print_header("Testing Node Creation")
        
        try:
            # Parse document
            documents = parse_document_with_llamaparse(self.test_pdf_path)
            
            # Set up LlamaIndex
            llm, _ = setup_llama_index()
            
            # Create nodes
            all_nodes, base_nodes, objects, page_nodes = create_nodes_from_documents(documents, llm)
            
            assert all_nodes is not None, "All nodes should not be None"
            assert len(all_nodes) > 0, "Should create at least one node"
            print_result(True, f"Created {len(all_nodes)} nodes in total")
        except Exception as e:
            print_result(False, f"Node creation failed: {str(e)}")
            pytest.fail(f"Node creation failed: {str(e)}")
    
    def test_pinecone_connection(self):
        """Test connection to Pinecone."""
        print_header("Testing Pinecone Connection")
        
        try:
            index_name = os.getenv("PINECONE_INDEX_NAME")
            vector_store = setup_pinecone_vector_store(index_name)
            assert vector_store is not None, "Vector store should not be None"
            print_result(True, f"Connected to Pinecone index '{index_name}'")
        except Exception as e:
            print_result(False, f"Pinecone connection failed: {str(e)}")
            pytest.fail(f"Pinecone connection failed: {str(e)}")
    
    def test_full_pipeline(self):
        """Test the full RAG pipeline."""
        print_header("Testing Full RAG Pipeline")
        
        try:
            # Parse document
            documents = parse_document_with_llamaparse(self.test_pdf_path)
            
            # Set up LlamaIndex
            llm, _ = setup_llama_index()
            
            # Create nodes
            all_nodes, _, _, _ = create_nodes_from_documents(documents, llm)
            
            # Set up Pinecone
            index_name = os.getenv("PINECONE_INDEX_NAME")
            vector_store = setup_pinecone_vector_store(index_name)
            
            # Create vector index
            index = create_vector_index(all_nodes, vector_store)
            
            # Create query engine
            query_engine = create_query_engine(index)
            
            # Test a simple query
            test_query = "What is this document about?"
            response = query_engine.query(test_query)
            
            assert response is not None, "Query response should not be None"
            assert response.response is not None, "Response text should not be None"
            assert len(response.response) > 0, "Response should not be empty"
            
            print_result(True, "Full RAG pipeline test successful")
            print(f"Query: {test_query}")
            print(f"Response: {response.response[:100]}...")
        except Exception as e:
            print_result(False, f"Full pipeline test failed: {str(e)}")
            pytest.fail(f"Full pipeline test failed: {str(e)}")

if __name__ == "__main__":
    pytest.main(["-v", __file__]) 