#!/usr/bin/env python3
"""
Script to set up a RAG (Retrieval Augmented Generation) pipeline.
This script:
1. Connects to GCP to retrieve PDF documents
2. Uses LlamaParse for document parsing
3. Utilizes GPT-4 as the language model
4. Integrates with Pinecone for vector storage
5. Creates an optimized query engine for document retrieval
"""

import os
import sys
import argparse
import nest_asyncio
from dotenv import load_dotenv
from pathlib import Path
from copy import deepcopy
from typing import List, Dict, Any, Optional

# Apply nest_asyncio to allow nested event loops (needed for some async operations)
nest_asyncio.apply()

# LlamaIndex imports
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.core.schema import TextNode, Document
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_cloud_services import LlamaParse

# Google Cloud imports
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
        'OPENAI_API_KEY',
        'LLAMA_CLOUD_API_KEY',
        'PINECONE_API_KEY',
        'PINECONE_INDEX_NAME',
        'GCP_STORAGE_BUCKET',
        'GOOGLE_APPLICATION_CREDENTIALS'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    # Set environment variables
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    os.environ["LLAMA_CLOUD_API_KEY"] = os.getenv("LLAMA_CLOUD_API_KEY")
    
    return {var: os.getenv(var) for var in required_vars}

def download_pdf_from_gcs(bucket_name: str, source_blob_name: str, destination_file_name: str):
    """Download a PDF file from Google Cloud Storage."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    
    os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
    blob.download_to_filename(destination_file_name)
    
    print_result(True, f"Downloaded {source_blob_name} to {destination_file_name}")
    return destination_file_name

def list_pdfs_in_gcs(bucket_name: str, prefix: str = ""):
    """List all PDF files in a GCS bucket with the given prefix."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    
    pdf_files = [blob.name for blob in blobs if blob.name.lower().endswith('.pdf')]
    return pdf_files

def get_page_nodes(docs: List[Document], separator: str = "\n---\n") -> List[TextNode]:
    """Convert documents into page nodes."""
    nodes = []
    for doc in docs:
        doc_chunks = doc.text.split(separator)
        for doc_chunk in doc_chunks:
            node = TextNode(text=doc_chunk, metadata=deepcopy(doc.metadata))
            nodes.append(node)
    return nodes

def setup_llama_index():
    """Set up LlamaIndex with GPT-4 and OpenAI embeddings."""
    # Initialize language model and embedding model
    llm = OpenAI(model="gpt-4-turbo-preview")
    embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    
    # Configure global settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    print_result(True, "Initialized LlamaIndex with GPT-4 and OpenAI embeddings")
    return llm, embed_model

def parse_document_with_llamaparse(file_path: str):
    """Parse a document using LlamaParse."""
    parser = LlamaParse(result_type="markdown")
    documents = parser.load_data(file_path)
    print_result(True, f"Parsed {file_path} with LlamaParse")
    return documents

def create_nodes_from_documents(documents: List[Document], llm):
    """Create different types of nodes from parsed documents."""
    # Create page nodes
    page_nodes = get_page_nodes(documents)
    print_result(True, f"Created {len(page_nodes)} page nodes")
    
    # Create markdown element nodes
    node_parser = MarkdownElementNodeParser(llm=llm, num_workers=8)
    nodes = node_parser.get_nodes_from_documents(documents)
    base_nodes, objects = node_parser.get_nodes_and_objects(nodes)
    print_result(True, f"Created {len(base_nodes)} base nodes and {len(objects)} object nodes")
    
    # Combine all nodes
    all_nodes = base_nodes + objects + page_nodes
    print_result(True, f"Combined into {len(all_nodes)} total nodes")
    
    return all_nodes, base_nodes, objects, page_nodes

def setup_pinecone_vector_store(index_name: str):
    """Set up Pinecone vector store."""
    import pinecone
    from pinecone.grpc import PineconeGRPC
    
    api_key = os.getenv("PINECONE_API_KEY")
    
    # Initialize Pinecone
    pc = PineconeGRPC(api_key=api_key)
    
    # Check if index exists
    indexes = pc.list_indexes()
    index_names = [index.name for index in indexes]
    
    if index_name not in index_names:
        raise ValueError(f"Pinecone index '{index_name}' does not exist. Please run setup_pinecone.py first.")
    
    # Get the index
    index = pc.Index(index_name)
    
    # Create vector store
    vector_store = PineconeVectorStore(pinecone_index=index)
    print_result(True, f"Connected to Pinecone index '{index_name}'")
    
    return vector_store

def create_vector_index(nodes: List[TextNode], vector_store):
    """Create a vector index from nodes using the specified vector store."""
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)
    print_result(True, "Created vector index with Pinecone storage")
    return index

def create_query_engine(index, top_k: int = 5):
    """Create an optimized query engine with reranking."""
    # Initialize reranker
    reranker = FlagEmbeddingReranker(top_n=top_k, model="BAAI/bge-reranker-large")
    
    # Create query engine
    query_engine = index.as_query_engine(
        similarity_top_k=top_k,
        node_postprocessors=[reranker],
        verbose=True
    )
    
    print_result(True, f"Created optimized query engine with top_k={top_k}")
    return query_engine

def process_pdf_for_rag(pdf_path: str, index_name: str):
    """Process a PDF file and set up the RAG pipeline."""
    print_header(f"Processing {pdf_path} for RAG Pipeline")
    
    try:
        # Set up LlamaIndex
        llm, embed_model = setup_llama_index()
        
        # Parse document with LlamaParse
        documents = parse_document_with_llamaparse(pdf_path)
        
        # Create nodes from documents
        all_nodes, base_nodes, objects, page_nodes = create_nodes_from_documents(documents, llm)
        
        # Set up Pinecone vector store
        vector_store = setup_pinecone_vector_store(index_name)
        
        # Create vector index
        index = create_vector_index(all_nodes, vector_store)
        
        # Create query engine
        query_engine = create_query_engine(index)
        
        print_header("RAG Pipeline Setup Complete")
        print(f"Document: {pdf_path}")
        print(f"Total nodes: {len(all_nodes)}")
        print(f"Vector store: Pinecone ({index_name})")
        print(f"Language model: GPT-4")
        print(f"Embedding model: text-embedding-3-small")
        
        # Return the query engine for further use
        return query_engine
        
    except Exception as e:
        print_result(False, f"Error setting up RAG pipeline: {str(e)}")
        return None

def main():
    """Main function to parse arguments and set up the RAG pipeline."""
    parser = argparse.ArgumentParser(description="Set up a RAG pipeline with LlamaParse, GPT-4, and Pinecone")
    parser.add_argument("--file", help="Path to a local PDF file to process")
    parser.add_argument("--gcs-file", help="Path to a PDF file in GCS to process")
    parser.add_argument("--list-pdfs", action="store_true", help="List all PDFs in the GCS bucket")
    parser.add_argument("--query", help="Query to run against the processed document")
    args = parser.parse_args()
    
    try:
        # Load environment variables
        env = load_environment()
        bucket_name = env['GCP_STORAGE_BUCKET']
        index_name = env['PINECONE_INDEX_NAME']
        
        if args.list_pdfs:
            # List PDFs in GCS bucket
            pdf_files = list_pdfs_in_gcs(bucket_name)
            print_header(f"PDFs in GCS Bucket: {bucket_name}")
            for pdf_file in pdf_files:
                print(f"- {pdf_file}")
            return
        
        # Determine PDF file to process
        pdf_path = None
        if args.file:
            pdf_path = args.file
        elif args.gcs_file:
            # Download from GCS
            local_path = f"data/temp/{os.path.basename(args.gcs_file)}"
            pdf_path = download_pdf_from_gcs(bucket_name, args.gcs_file, local_path)
        else:
            print("Please specify either --file or --gcs-file")
            return
        
        # Process PDF and set up RAG pipeline
        query_engine = process_pdf_for_rag(pdf_path, index_name)
        
        # Run query if provided
        if args.query and query_engine:
            print_header(f"Running Query: {args.query}")
            response = query_engine.query(args.query)
            print("\nResponse:")
            print(response.response)
            print("\nSources:")
            for i, node in enumerate(response.source_nodes[:3]):
                print(f"\nSource {i+1}:")
                print(node.get_content()[:300] + "..." if len(node.get_content()) > 300 else node.get_content())
        
    except Exception as e:
        print_result(False, f"Error in main function: {str(e)}")

if __name__ == "__main__":
    main() 