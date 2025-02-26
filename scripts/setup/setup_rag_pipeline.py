#!/usr/bin/env python3
"""
Script to set up a RAG (Retrieval Augmented Generation) pipeline.
This script:
1. Connects to GCP to retrieve PDF documents
2. Uses LlamaParse for document parsing
3. Utilizes GPT-4o-mini as the language model
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
from typing import List, Dict, Any, Optional, Tuple
import hashlib
import time
import tempfile
import json
import pickle

# Apply nest_asyncio to allow nested event loops (needed for some async operations)
nest_asyncio.apply()

# LlamaIndex imports
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex, Settings, StorageContext, ServiceContext
from llama_index.core.schema import TextNode, Document
from llama_index.core.node_parser import MarkdownElementNodeParser, SentenceSplitter
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

def get_gcs_document_fingerprint(storage_client, bucket_name, gcs_path):
    """
    Generate a unique fingerprint for a document in GCS based on its metadata.
    """
    try:
        # Get the bucket and blob
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        
        # Get blob metadata
        blob.reload()  # Ensure we have the latest metadata
        file_name = os.path.basename(gcs_path)
        file_size = blob.size
        file_updated = blob.updated.timestamp() if blob.updated else 0
        generation = blob.generation
        
        # Create a fingerprint using available metadata
        fingerprint = f"{file_name}_{file_size}_{int(file_updated)}_{generation}"
        return fingerprint
        
    except Exception as e:
        print_result(False, f"Error generating GCS document fingerprint: {e}")
        # Return a timestamp-based fingerprint as fallback
        return f"{os.path.basename(gcs_path)}_{int(time.time())}"

def download_from_gcs(gcs_path):
    """
    Download a file from Google Cloud Storage.
    
    Args:
        gcs_path: Path to the file in GCS (e.g., 'uploads/file.pdf')
        
    Returns:
        Local path to the downloaded file
    """
    try:
        # Load environment variables
        load_dotenv()
        bucket_name = os.environ.get("GCP_STORAGE_BUCKET")
        
        if not bucket_name:
            print_result(False, "GCP_STORAGE_BUCKET environment variable is not set")
            return None
            
        # Create local directory if it doesn't exist
        local_dir = "data/temp"
        os.makedirs(local_dir, exist_ok=True)
        
        # Set local path
        local_path = os.path.join(local_dir, os.path.basename(gcs_path))
        
        print(f"Downloading {gcs_path} from GCS bucket {bucket_name}...")
        
        # Initialize GCS client
        storage_client = storage.Client()
        
        # Get document fingerprint from GCS metadata
        gcs_fingerprint = get_gcs_document_fingerprint(storage_client, bucket_name, gcs_path)
        print(f"GCS document fingerprint: {gcs_fingerprint}")
        
        # Initialize Pinecone
        index_name = "financial-reports"
        pinecone_client = init_pinecone()
        
        if pinecone_client:
            # Check if document has already been processed
            is_processed, document_namespace = check_document_processed(
                pinecone_client, 
                index_name, 
                gcs_fingerprint
            )
            
            if is_processed and document_namespace:
                print_result(True, f"Document already processed in Pinecone with namespace: {document_namespace}")
                # Store the namespace in a file for later use
                namespace_file = os.path.join(local_dir, f"{os.path.basename(gcs_path)}.namespace")
                with open(namespace_file, 'w') as f:
                    f.write(document_namespace)
        
        # Download the file regardless of whether it's been processed
        # This ensures we have the file for querying
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        blob.download_to_filename(local_path)
        
        # Store the fingerprint in a file for later use
        fingerprint_file = os.path.join(local_dir, f"{os.path.basename(gcs_path)}.fingerprint")
        with open(fingerprint_file, 'w') as f:
            f.write(gcs_fingerprint)
        
        print_result(True, f"Downloaded {gcs_path} to {local_path}")
        return local_path
        
    except Exception as e:
        print_result(False, f"Error downloading file from GCS: {e}")
        return None

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
    """
    Initialize LlamaIndex with GPT-3.5-Turbo-0125 and OpenAI embeddings.
    Returns the initialized settings.
    """
    print("\nInitializing LlamaIndex with GPT-3.5-Turbo-0125 and OpenAI embeddings...")
    
    # Set up the language model
    llm = OpenAI(model="gpt-3.5-turbo-0125", temperature=0.1)
    
    # Set up the embeddings model
    embed_model = OpenAIEmbedding()
    
    # Configure settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    print_result(True, "LlamaIndex initialized with GPT-3.5-Turbo-0125 and OpenAI embeddings")
    
    return Settings

def get_llamaparse_cache_path(document_fingerprint):
    """Get the path to the cached LlamaParse result for a document."""
    cache_dir = os.path.join(get_cache_dir(), "llamaparse")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{document_fingerprint}.json")

def save_llamaparse_result(document_fingerprint, documents):
    """Save LlamaParse result to cache."""
    try:
        cache_path = get_llamaparse_cache_path(document_fingerprint)
        
        # Convert Document objects to serializable format
        serializable_docs = []
        for doc in documents:
            serializable_doc = {
                "text": doc.text,
                "metadata": doc.metadata
            }
            serializable_docs.append(serializable_doc)
        
        with open(cache_path, 'w') as f:
            json.dump(serializable_docs, f)
            
        print_result(True, f"Saved LlamaParse result to cache: {cache_path}")
        return True
    except Exception as e:
        print_result(False, f"Error saving LlamaParse result to cache: {e}")
        return False

def load_llamaparse_result(document_fingerprint):
    """Load LlamaParse result from cache."""
    try:
        cache_path = get_llamaparse_cache_path(document_fingerprint)
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                serialized_docs = json.load(f)
            
            # Convert serialized format back to Document objects
            from llama_index.core.schema import Document
            documents = []
            for doc_data in serialized_docs:
                doc = Document(
                    text=doc_data["text"],
                    metadata=doc_data["metadata"]
                )
                documents.append(doc)
                
            print_result(True, f"Loaded LlamaParse result from cache: {cache_path}")
            return documents
        return None
    except Exception as e:
        print_result(False, f"Error loading LlamaParse result from cache: {e}")
        return None

def list_llamaparse_cache():
    """List all documents in the LlamaParse cache."""
    print_header("Documents in LlamaParse Cache")
    
    try:
        cache_dir = os.path.join(get_cache_dir(), "llamaparse")
        os.makedirs(cache_dir, exist_ok=True)
        
        cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.json')]
        
        if not cache_files:
            print("No documents found in LlamaParse cache.")
            return
        
        print(f"Found {len(cache_files)} documents in LlamaParse cache:")
        print("\n{:<40} {:<20}".format(
            "Document Fingerprint", "Cache File Size"
        ))
        print("-" * 60)
        
        for cache_file in cache_files:
            file_path = os.path.join(cache_dir, cache_file)
            file_size = os.path.getsize(file_path)
            print("{:<40} {:<20}".format(
                cache_file.replace('.json', ''), f"{file_size/1024:.2f} KB"
            ))
        
    except Exception as e:
        print_result(False, f"Error listing LlamaParse cache: {e}")

def clear_llamaparse_cache():
    """Clear all LlamaParse cache files."""
    try:
        cache_dir = os.path.join(get_cache_dir(), "llamaparse")
        if os.path.exists(cache_dir):
            count = 0
            for file in os.listdir(cache_dir):
                if file.endswith('.json'):
                    os.remove(os.path.join(cache_dir, file))
                    count += 1
            print_result(True, f"Cleared {count} files from LlamaParse cache")
        else:
            print_result(True, "LlamaParse cache directory does not exist")
        return True
    except Exception as e:
        print_result(False, f"Error clearing LlamaParse cache: {e}")
        return False

def parse_document_with_llamaparse(file_path: str):
    """Parse a document using LlamaParse with caching."""
    # Generate document fingerprint
    document_fingerprint = get_document_fingerprint(file_path)
    
    # Check if we have a cached result
    cached_docs = load_llamaparse_result(document_fingerprint)
    if cached_docs:
        print_result(True, f"Using cached LlamaParse result for {file_path}")
        return cached_docs
    
    # If not cached, parse with LlamaParse
    print(f"Parsing {file_path} with LlamaParse (not found in cache)...")
    parser = LlamaParse(result_type="markdown")
    documents = parser.load_data(file_path)
    print_result(True, f"Parsed {file_path} with LlamaParse")
    
    # Save result to cache
    save_llamaparse_result(document_fingerprint, documents)
    
    return documents

def create_nodes_from_documents(documents: List[Document], settings=None):
    """Create different types of nodes from parsed documents."""
    from llama_index.core.node_parser import MarkdownElementNodeParser, SentenceSplitter
    from llama_index.core.schema import TextNode
    
    # Create page nodes
    page_nodes = get_page_nodes(documents)
    print_result(True, f"Created {len(page_nodes)} page nodes")
    
    # Create markdown element nodes with more workers for better performance
    node_parser = MarkdownElementNodeParser(num_workers=8)
    nodes = node_parser.get_nodes_from_documents(documents)
    base_nodes, objects = node_parser.get_nodes_and_objects(nodes)
    print_result(True, f"Created {len(base_nodes)} base nodes and {len(objects)} object nodes")
    
    # Combine all nodes
    all_nodes = []
    
    # Process and enhance each node
    for i, node in enumerate(base_nodes + objects + page_nodes):
        # Skip nodes with empty or very short content
        if not node.text or len(node.text.strip()) < 20:
            continue
            
        # Ensure each node has proper metadata
        if not hasattr(node, 'metadata') or not node.metadata:
            node.metadata = {}
            
        # Create an extremely minimal metadata set for Pinecone
        # Only include absolutely essential fields to stay under the 40KB limit
        pinecone_metadata = {}
        
        # Add only the most essential metadata fields
        # Use short key names to reduce metadata size
        pinecone_metadata['fn'] = os.path.basename(documents[0].metadata.get('file_name', f'doc_{i}')) if documents else f'doc_{i}'
        
        # Add page number if available - use short key name
        if 'page_label' in node.metadata:
            try:
                pinecone_metadata['pg'] = int(node.metadata['page_label'])
            except:
                pinecone_metadata['pg'] = 0
                
        # Add very short content preview (just 10 chars)
        content_preview = node.text[:10] + "..." if len(node.text) > 10 else node.text
        pinecone_metadata['prev'] = content_preview
        
        # Add unique ID based on content hash - use short key name
        content_hash = hashlib.md5(node.text.encode()).hexdigest()[:8]
        pinecone_metadata['hash'] = content_hash
        
        # Replace node metadata with the minimal set for Pinecone
        node.metadata = pinecone_metadata
        
        # Add to final nodes list
        all_nodes.append(node)
    
    # Filter out duplicate nodes based on content hash
    unique_nodes = {}
    for node in all_nodes:
        content_hash = node.metadata.get('hash')
        if content_hash and content_hash not in unique_nodes:
            unique_nodes[content_hash] = node
    
    final_nodes = list(unique_nodes.values())
    print_result(True, f"Final node count after filtering: {len(final_nodes)}")
    
    # Further chunk nodes if they're too large
    print("Chunking nodes to ensure they fit within Pinecone's metadata limits...")
    chunked_nodes = []
    
    # Initialize sentence splitter for chunking
    splitter = SentenceSplitter(chunk_size=256, chunk_overlap=20)
    
    # Count statistics for reporting
    large_nodes_count = 0
    total_chunks_from_large_nodes = 0
    
    for node in final_nodes:
        # Check if node text is very large (which could lead to large metadata)
        if len(node.text) > 1000:
            large_nodes_count += 1
            # Split the node into smaller chunks
            try:
                node_chunks = splitter.split_text(node.text)
                total_chunks_from_large_nodes += len(node_chunks)
                
                # Create new nodes from chunks
                for j, chunk in enumerate(node_chunks):
                    chunk_node = TextNode(
                        text=chunk,
                        metadata=node.metadata.copy()  # Copy the minimal metadata
                    )
                    # Add chunk number to metadata
                    chunk_node.metadata['c'] = j
                    chunked_nodes.append(chunk_node)
            except Exception as e:
                print(f"Error chunking node: {e}")
                # If chunking fails, keep the original node
                chunked_nodes.append(node)
        else:
            # Node is small enough, keep as is
            chunked_nodes.append(node)
    
    # Print detailed chunking statistics
    small_nodes_count = len(final_nodes) - large_nodes_count
    print(f"\nChunking Statistics:")
    print(f"- Original nodes: {len(final_nodes)}")
    print(f"- Small nodes (kept as is): {small_nodes_count}")
    print(f"- Large nodes (chunked): {large_nodes_count}")
    print(f"- New chunks created from large nodes: {total_chunks_from_large_nodes}")
    print(f"- Final node count after chunking: {len(chunked_nodes)}")
    
    # Calculate batch information
    batch_size = 5  # Same as used in process_pdf_for_rag
    batch_count = (len(chunked_nodes) + batch_size - 1) // batch_size
    print(f"- Will be processed in {batch_count} batches of {batch_size} nodes each")
    
    print_result(True, f"Final node count after chunking: {len(chunked_nodes)}")
    return chunked_nodes, base_nodes, objects, page_nodes

def setup_pinecone_vector_store(index_name: str, namespace: str = None):
    """Set up Pinecone vector store."""
    from pinecone import Pinecone
    
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable is not set")
    
    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=api_key)
        
        # Check if index exists
        indexes = pc.list_indexes()
        index_names = [index.name for index in indexes]
        print(f"Available Pinecone indexes: {index_names}")
        
        if index_name not in index_names:
            # Create index if it doesn't exist
            print(f"Creating new Pinecone index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=1536,  # OpenAI embedding dimension
                metric="cosine"
            )
            print_result(True, f"Created Pinecone index '{index_name}'")
        
        # Connect to index
        pinecone_index = pc.Index(index_name)
        
        # Create vector store with namespace if provided
        if namespace:
            # Clean namespace to ensure it's valid
            clean_namespace = str(namespace).replace(" ", "_").replace("/", "_")[:36]
            print(f"Using namespace: {clean_namespace}")
            vector_store = PineconeVectorStore(pinecone_index=pinecone_index, namespace=clean_namespace)
            print_result(True, f"Connected to Pinecone index '{index_name}' with namespace '{clean_namespace}'")
        else:
            vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
            print_result(True, f"Connected to Pinecone index '{index_name}'")
        
        return vector_store
    
    except Exception as e:
        print_result(False, f"Error setting up Pinecone: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def create_vector_index(nodes: List[TextNode], vector_store):
    """Create a vector index from nodes using the specified vector store."""
    from llama_index.core import StorageContext, VectorStoreIndex
    from llama_index.core.node_parser import SentenceSplitter
    
    # Log the number of nodes being indexed
    print(f"Indexing {len(nodes)} nodes into Pinecone")
    
    # Create a storage context with the vector store
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Create the vector index
    try:
        # First try to create index with all nodes
        index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)
        print_result(True, f"Created vector index with {len(nodes)} nodes")
    except Exception as e:
        print(f"Error creating index with all nodes: {str(e)}")
        print("Attempting to create index with chunked nodes...")
        
        # If that fails, try chunking the nodes
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        chunked_nodes = []
        
        for node in nodes:
            try:
                # Split the node into smaller chunks
                node_chunks = splitter.split_node(node)
                chunked_nodes.extend(node_chunks)
            except Exception as chunk_error:
                print(f"Error chunking node: {str(chunk_error)}")
                # Skip problematic nodes
                continue
        
        print(f"Created {len(chunked_nodes)} chunked nodes")
        
        # Create index with chunked nodes
        index = VectorStoreIndex(nodes=chunked_nodes, storage_context=storage_context)
        print_result(True, f"Created vector index with {len(chunked_nodes)} chunked nodes")
    
    return index

def create_query_engine(index, top_k: int = 8):
    """Create an optimized query engine with reranking."""
    # Initialize reranker
    reranker = FlagEmbeddingReranker(top_n=top_k, model="BAAI/bge-reranker-large")
    
    # Create response synthesizer with better prompting
    from llama_index.core.response_synthesizers import get_response_synthesizer
    
    response_synthesizer = get_response_synthesizer(
        response_mode="refine",
        verbose=True,
        streaming=False,
        structured_answer_filtering=True
    )
    
    # Create query engine with improved settings
    query_engine = index.as_query_engine(
        similarity_top_k=top_k * 2,  # Retrieve more nodes initially for better reranking
        node_postprocessors=[reranker],
        response_synthesizer=response_synthesizer,
        verbose=True
    )
    
    print_result(True, f"Created optimized query engine with top_k={top_k}")
    return query_engine

def get_document_fingerprint(file_path):
    """
    Generate a unique fingerprint for a document based on its content and metadata.
    This helps identify if a document has already been processed.
    """
    try:
        # Get file size and modification time
        file_stats = os.stat(file_path)
        file_size = file_stats.st_size
        file_mtime = file_stats.st_mtime
        
        # Get file name
        file_name = os.path.basename(file_path)
        
        # Calculate MD5 hash of the first 10KB of the file
        # (Using the entire file could be slow for large PDFs)
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5(f.read(10240)).hexdigest()
        
        # Combine all information into a unique fingerprint
        fingerprint = f"{file_name}_{file_size}_{int(file_mtime)}_{file_hash}"
        return fingerprint
        
    except Exception as e:
        print_result(False, f"Error generating document fingerprint: {e}")
        # Return a timestamp-based fingerprint as fallback
        return f"{os.path.basename(file_path)}_{int(time.time())}"

def check_document_processed(pinecone_client, index_name, document_fingerprint):
    """
    Check if a document with the given fingerprint has already been processed and stored in Pinecone.
    
    Returns:
        tuple: (is_processed, namespace) - Whether the document is processed and its namespace if it exists
    """
    try:
        # Connect to the index
        index = pinecone_client.Index(index_name)
        
        # Check the document registry
        # We use a special namespace to store document metadata
        registry_namespace = "document_registry"
        
        # Query the registry for the document fingerprint
        query_response = index.query(
            namespace=registry_namespace,
            vector=[0.0] * 1536,  # Dummy vector for metadata-only query
            filter={"fingerprint": document_fingerprint},
            top_k=1,
            include_metadata=True
        )
        
        # Check if we found a match
        if query_response.matches and len(query_response.matches) > 0:
            # Document exists in registry
            match = query_response.matches[0]
            document_namespace = match.metadata.get("namespace")
            print_result(True, f"Document already processed. Using existing namespace: {document_namespace}")
            return True, document_namespace
        
        # Document not found in registry
        return False, None
        
    except Exception as e:
        print_result(False, f"Error checking document registry: {e}")
        # If there's an error, assume document hasn't been processed
        return False, None

def register_document(pinecone_client, index_name, document_fingerprint, document_namespace, metadata=None):
    """Register a document in the document registry."""
    try:
        # Create a registry namespace if it doesn't exist
        registry_namespace = "document_registry"
        
        # Get the Pinecone index
        index = pinecone_client.Index(index_name)
        
        # Create a unique ID for the document
        document_id = f"doc_{hashlib.md5(document_fingerprint.encode()).hexdigest()[:24]}"
        
        # Create metadata for the document
        if metadata is None:
            metadata = {}
            
        # Add timestamp
        metadata['registered_at'] = time.time()
        metadata['namespace'] = document_namespace
        
        # Create a non-zero vector (required by Pinecone)
        # Use a simple vector with small non-zero values
        vector_dimension = 1536  # OpenAI embedding dimension
        non_zero_vector = [0.001] * vector_dimension
        
        # Upsert the document registry entry
        index.upsert(
            vectors=[
                {
                    "id": document_id,
                    "values": non_zero_vector,
                    "metadata": metadata
                }
            ],
            namespace=registry_namespace
        )
        
        print_result(True, f"Registered document in registry: {document_id}")
        return True
        
    except Exception as e:
        print_result(False, f"Error registering document: {e}")
        return False

def estimate_token_usage(text_length, model="gpt-3.5-turbo-0125"):
    """
    Estimate token usage for a given text length and model.
    This is a rough estimation based on average token-to-character ratios.
    
    Args:
        text_length: Length of text in characters
        model: Model name to estimate for
        
    Returns:
        int: Estimated token count
    """
    # Average characters per token (rough estimates)
    chars_per_token = {
        "gpt-3.5-turbo-0125": 4.0,
        "gpt-4-0125-preview": 3.8,
        "text-embedding-3-small": 4.5,
        "default": 4.0
    }
    
    # Get the appropriate ratio
    ratio = chars_per_token.get(model, chars_per_token["default"])
    
    # Estimate tokens
    estimated_tokens = int(text_length / ratio)
    
    return estimated_tokens

def get_cache_dir():
    """Get the directory for local cache files."""
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def get_local_cache_path(document_fingerprint):
    """Get the path to the local cache file for a document."""
    cache_dir = get_cache_dir()
    return os.path.join(cache_dir, f"{document_fingerprint}.pickle")

def save_to_local_cache(document_fingerprint, data):
    """Save processed document data to local cache."""
    try:
        cache_path = get_local_cache_path(document_fingerprint)
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        print_result(True, f"Saved document to local cache: {cache_path}")
        return True
    except Exception as e:
        print_result(False, f"Error saving to local cache: {e}")
        return False

def load_from_local_cache(document_fingerprint):
    """Load processed document data from local cache."""
    try:
        cache_path = get_local_cache_path(document_fingerprint)
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            print_result(True, f"Loaded document from local cache: {cache_path}")
            return data
        return None
    except Exception as e:
        print_result(False, f"Error loading from local cache: {e}")
        return None

def clear_local_cache():
    """Clear all local cache files."""
    try:
        cache_dir = get_cache_dir()
        count = 0
        for file in os.listdir(cache_dir):
            if file.endswith('.pickle'):
                os.remove(os.path.join(cache_dir, file))
                count += 1
        print_result(True, f"Cleared {count} files from local cache")
        return True
    except Exception as e:
        print_result(False, f"Error clearing local cache: {e}")
        return False

def process_pdf_for_rag(pdf_path, query=None, use_local_cache=True, use_llamaparse_cache=True):
    """
    Process a PDF file for RAG using LlamaIndex with GPT-3.5-Turbo-0125.
    
    Steps:
    1. Check if document has already been processed
    2. Connect to GCP for document storage
    3. Use LlamaParse for document parsing
    4. Integrate with Pinecone for vector storage
    5. Create an optimized query engine for document retrieval
    """
    print(f"\n{'='*80}\n Processing PDF for RAG: {pdf_path}\n{'='*80}")
    
    try:
        # Generate document fingerprint
        document_fingerprint = get_document_fingerprint(pdf_path)
        print(f"Document fingerprint: {document_fingerprint}")
        
        # Check local cache first if enabled
        if use_local_cache:
            cached_data = load_from_local_cache(document_fingerprint)
            if cached_data:
                print("Using document from local cache")
                
                # Set up LlamaIndex
                settings = setup_llama_index()
                
                # Extract data from cache
                nodes = cached_data.get('nodes')
                document_namespace = cached_data.get('namespace')
                
                if nodes:
                    # Create index directly from nodes
                    index = VectorStoreIndex(nodes)
                    
                    # Create query engine
                    query_engine = index.as_query_engine(similarity_top_k=5)
                    
                    # Print summary
                    print("\nRAG Pipeline Setup Complete (using local cache):")
                    print(f"Document: {os.path.basename(pdf_path)}")
                    print(f"Total nodes: {len(nodes)}")
                    print(f"Language model: GPT-3.5-Turbo-0125")
                    print(f"Embedding model: text-embedding-3-small")
                    
                    # Run query if provided
                    if query:
                        print(f"\n{'='*80}\n Running Query: {query}\n{'='*80}")
                        
                        # Estimate token usage for query
                        query_tokens = estimate_token_usage(len(query))
                        response_tokens = query_tokens * 3  # Rough estimate: response is ~3x query length
                        print(f"Estimated token usage for query: ~{query_tokens} tokens")
                        print(f"Estimated token usage for response: ~{response_tokens} tokens")
                        print(f"Total estimated token usage: ~{query_tokens + response_tokens} tokens")
                        
                        # Execute query
                        response = query_engine.query(query)
                        print(f"\nResponse:\n{response}")
                    
                    print_result(True, "RAG pipeline setup complete (using local cache)")
                    return query_engine
        
        # Initialize Pinecone
        index_name = "financial-reports"
        pinecone_client = init_pinecone()
        
        if pinecone_client is None:
            print_result(False, "Failed to initialize Pinecone client")
            return None
            
        # Check if document has already been processed in Pinecone
        is_processed, document_namespace = check_document_processed(
            pinecone_client, 
            index_name, 
            document_fingerprint
        )
        
        # Create a query engine using the existing namespace if document was already processed
        if is_processed and document_namespace:
            print("Using existing document embeddings from Pinecone")
            
            # Set up LlamaIndex
            settings = setup_llama_index()
            
            # Create vector store with the existing namespace
            vector_store = PineconeVectorStore(
                pinecone_index=pinecone_client.Index(index_name),
                namespace=document_namespace
            )
            
            # Create storage context
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Load the existing index
            index = VectorStoreIndex.from_vector_store(
                vector_store
            )
            
            # Create query engine
            query_engine = index.as_query_engine(similarity_top_k=5)
            
            # Print summary
            print("\nRAG Pipeline Setup Complete (using cached embeddings):")
            print(f"Document: {os.path.basename(pdf_path)}")
            print(f"Vector store: Pinecone ({index_name})")
            print(f"Namespace: {document_namespace}")
            print(f"Language model: GPT-3.5-Turbo-0125")
            print(f"Embedding model: text-embedding-3-small")
            
        else:
            # Document hasn't been processed yet, proceed with full processing
            print("Document not found in cache. Processing new document...")
            
            # Estimate token usage before processing
            file_size = os.path.getsize(pdf_path)
            estimated_tokens = estimate_token_usage(file_size * 0.5)  # Rough estimate: 50% of file size in chars
            print(f"Estimated token usage for processing: ~{estimated_tokens} tokens")
            print("Note: This is a rough estimate and actual usage may vary.")
            
            # Ask for confirmation if estimated token usage is high
            if estimated_tokens > 100000:
                print("\nWARNING: Processing this document may use a significant number of tokens.")
                print("If you're experiencing rate limits, consider processing a smaller document first.")
                print("You can also try again later when your rate limits have reset.")
                
                # Continue with processing after warning
                print("\nContinuing with document processing...")
            
            # Create a unique namespace for this document
            document_namespace = f"doc_{hashlib.md5(document_fingerprint.encode()).hexdigest()[:10]}"
            print(f"Using namespace: {document_namespace}")
            
            # Set up LlamaIndex
            settings = setup_llama_index()
            
            # Parse document with LlamaParse (with caching if enabled)
            if use_llamaparse_cache:
                documents = parse_document_with_llamaparse(pdf_path)
            else:
                # Bypass the caching mechanism if disabled
                print(f"Parsing {pdf_path} with LlamaParse (caching disabled)...")
                parser = LlamaParse(result_type="markdown")
                documents = parser.load_data(pdf_path)
                print_result(True, f"Parsed {pdf_path} with LlamaParse")
            
            # Create nodes from documents with improved chunking
            all_nodes, base_nodes, objects, page_nodes = create_nodes_from_documents(documents)
            
            # Save to local cache with original metadata
            if use_local_cache:
                cache_data = {
                    'nodes': all_nodes,
                    'namespace': document_namespace,
                    'metadata': {
                        'file_name': os.path.basename(pdf_path),
                        'node_count': len(all_nodes),
                        'page_count': len(page_nodes) if page_nodes else 0,
                        'estimated_tokens': estimated_tokens,
                        'processed_at': time.time()
                    }
                }
                save_to_local_cache(document_fingerprint, cache_data)
            
            # Create vector store with the new namespace
            vector_store = PineconeVectorStore(
                pinecone_index=pinecone_client.Index(index_name),
                namespace=document_namespace
            )
            
            # Create storage context
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            try:
                # Create vector index with batch processing to avoid errors
                print("Creating vector index with reduced metadata size...")
                
                # Process in smaller batches to avoid overwhelming Pinecone
                batch_size = 5  # Reduce batch size to 5 for even smaller batches
                successful_nodes = 0
                
                for i in range(0, len(all_nodes), batch_size):
                    batch = all_nodes[i:i+batch_size]
                    try:
                        # Create a temporary storage context for this batch
                        batch_storage_context = StorageContext.from_defaults(
                            vector_store=vector_store
                        )
                        
                        # Add this batch to the vector store
                        print(f"Processing batch {i//batch_size + 1}/{(len(all_nodes) + batch_size - 1)//batch_size}...")
                        
                        # Create index for this batch
                        VectorStoreIndex(batch, storage_context=batch_storage_context)
                        successful_nodes += len(batch)
                        
                    except Exception as batch_error:
                        print(f"Error processing batch {i//batch_size + 1}: {str(batch_error)}")
                        
                        # If batch fails, try processing nodes one by one
                        print("Attempting to process nodes individually...")
                        for node in batch:
                            try:
                                # Create a single-node batch
                                single_node_context = StorageContext.from_defaults(
                                    vector_store=vector_store
                                )
                                VectorStoreIndex([node], storage_context=single_node_context)
                                successful_nodes += 1
                            except Exception as node_error:
                                print(f"Failed to process individual node: {str(node_error)}")
                
                # Create the final index from the vector store
                index = VectorStoreIndex.from_vector_store(vector_store)
                print_result(True, f"Successfully created vector index with {successful_nodes}/{len(all_nodes)} nodes")
                
            except Exception as e:
                print(f"Error creating index: {str(e)}")
                print("Falling back to local index...")
                
                # If Pinecone fails, create a local index
                index = VectorStoreIndex(all_nodes)
                print_result(True, "Created local vector index as fallback")
            
            # Register the document in the registry
            document_metadata = {
                "file_name": os.path.basename(pdf_path),
                "node_count": len(all_nodes),
                "page_count": len(page_nodes) if page_nodes else 0,
                "estimated_tokens": estimated_tokens
            }
            
            try:
                register_document(
                    pinecone_client, 
                    index_name, 
                    document_fingerprint, 
                    document_namespace,
                    metadata=document_metadata
                )
            except Exception as e:
                print(f"Warning: Could not register document in Pinecone: {e}")
                print("Continuing with local index only.")
            
            # Create query engine
            query_engine = index.as_query_engine(similarity_top_k=5)
            
            # Print summary
            print("\nRAG Pipeline Setup Complete (new document processed):")
            print(f"Document: {os.path.basename(pdf_path)}")
            print(f"Total nodes: {len(all_nodes)}")
            print(f"Vector store: {'Local (Pinecone failed)' if 'e' in locals() and isinstance(e, Exception) else f'Pinecone ({index_name})'}")
            print(f"Namespace: {document_namespace}")
            print(f"Language model: GPT-3.5-Turbo-0125")
            print(f"Embedding model: text-embedding-3-small")
        
        # Run query if provided
        if query:
            print(f"\n{'='*80}\n Running Query: {query}\n{'='*80}")
            
            # Estimate token usage for query
            query_tokens = estimate_token_usage(len(query))
            response_tokens = query_tokens * 3  # Rough estimate: response is ~3x query length
            print(f"Estimated token usage for query: ~{query_tokens} tokens")
            print(f"Estimated token usage for response: ~{response_tokens} tokens")
            print(f"Total estimated token usage: ~{query_tokens + response_tokens} tokens")
            
            # Execute query
            response = query_engine.query(query)
            print(f"\nResponse:\n{response}")
        
        print_result(True, "RAG pipeline setup complete")
        return query_engine
        
    except Exception as e:
        print_result(False, f"Error setting up RAG pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None

def init_pinecone():
    """Initialize Pinecone client and return it."""
    try:
        # Explicitly load environment variables
        load_dotenv()
        
        from pinecone import Pinecone
        
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            print_result(False, "PINECONE_API_KEY environment variable is not set")
            return None
            
        # Initialize Pinecone
        pc = Pinecone(api_key=api_key)
        print_result(True, "Pinecone client initialized")
        return pc
        
    except Exception as e:
        print_result(False, f"Error initializing Pinecone client: {e}")
        return None

def verify_pinecone_setup(index_name=None):
    """Verify that Pinecone is properly set up and the index exists."""
    print_header("Verifying Pinecone Setup")
    try:
        # Explicitly load environment variables
        load_dotenv()
        
        from pinecone import Pinecone
        
        # Get index name from environment if not provided
        if index_name is None:
            index_name = os.getenv("PINECONE_INDEX_NAME")
            if not index_name:
                print_result(False, "PINECONE_INDEX_NAME environment variable is not set")
                return False
        
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            print_result(False, "PINECONE_API_KEY environment variable is not set")
            return False
        
        # Initialize Pinecone
        pc = Pinecone(api_key=api_key)
        print_result(True, "Pinecone client initialized")
        
        # Check if index exists
        indexes = pc.list_indexes()
        index_names = [index.name for index in indexes]
        
        if index_name not in index_names:
            print_result(False, f"Pinecone index '{index_name}' does not exist")
            return False
        
        # Connect to index to verify it's accessible
        pinecone_index = pc.Index(index_name)
        stats = pinecone_index.describe_index_stats()
        print_result(True, f"Successfully connected to index '{index_name}'")
        print(f"Index stats: {stats}")
        
        return True
        
    except Exception as e:
        print_result(False, f"Error verifying Pinecone setup: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def verify_openai_api(verbose=True):
    """Verify that the OpenAI API key is valid and check quota status."""
    print_header("Verifying OpenAI API")
    try:
        import openai
        from openai import OpenAI
        
        # Explicitly load environment variables
        load_dotenv()
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print_result(False, "OPENAI_API_KEY environment variable is not set")
            return False
        
        print(f"Using API key: {api_key[:10]}...{api_key[-5:]}")
        
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Make a simple API call to check if the key is valid
        try:
            # Use a minimal model call to check API access
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input="Hello, world!",
                encoding_format="float"
            )
            print_result(True, "OpenAI API key is valid")
            
            # If we get here, the API key is valid
            if verbose:
                print("Successfully connected to OpenAI API")
                print("Used model: text-embedding-3-small")
                print(f"Embedding dimensions: {len(response.data[0].embedding)}")
                print("Your OpenAI API key is working correctly.")
            
            return True
            
        except openai.RateLimitError as e:
            print_result(False, f"OpenAI API quota exceeded: {str(e)}")
            print("\nYour API key is valid, but you've exceeded your current quota.")
            
            # Try to get usage information
            try:
                from datetime import datetime
                import requests
                
                # Get current date in YYYY-MM-DD format
                today = datetime.now().strftime("%Y-%m-%d")
                
                # Make a request to the OpenAI API to get usage data
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
                response = requests.get(
                    f"https://api.openai.com/v1/usage?date={today}",
                    headers=headers
                )
                
                if response.status_code == 200:
                    usage_data = response.json()
                    
                    if 'data' in usage_data and usage_data['data']:
                        total_tokens = sum(item.get('n_context_tokens_total', 0) + item.get('n_generated_tokens_total', 0) 
                                          for item in usage_data['data'])
                        
                        # Get usage by model
                        models_usage = {}
                        for item in usage_data['data']:
                            model = item.get('snapshot_id', 'unknown')
                            tokens = item.get('n_context_tokens_total', 0) + item.get('n_generated_tokens_total', 0)
                            if model in models_usage:
                                models_usage[model] += tokens
                            else:
                                models_usage[model] = tokens
                        
                        print(f"\nYou've used {total_tokens} tokens today:")
                        for model, tokens in models_usage.items():
                            print(f"  - {model}: {tokens} tokens")
                        
                        # Update message for Tier 1 users
                        print("\nYou're on OpenAI's Usage Tier 1, which has higher limits than the free tier.")
                        print("However, you've still exceeded your current rate limit.")
            except:
                pass
            
            print("\nOptions to resolve this:")
            print("1. Wait for your rate limit to reset (typically within a few hours)")
            print("2. Upgrade to Tier 2 by spending at least $50 on the API and waiting 7 days after your first payment")
            print("3. Use a different API key by updating your .env file")
            print("\nTo check your current usage and limits, visit: https://platform.openai.com/account/usage")
            print("To check your rate limits, visit: https://platform.openai.com/account/rate-limits")
            
            # Add specific guidance for this project
            print("\nFor this project:")
            print("1. You can run the script with --verify-pinecone to check if Pinecone is set up correctly")
            print("2. You can run the script with --check-usage to monitor your OpenAI API usage")
            print("3. Consider implementing token usage estimation to better manage your quota")
            
            return False
            
        except Exception as e:
            print_result(False, f"Error validating OpenAI API key: {str(e)}")
            return False
    
    except Exception as e:
        print_result(False, f"Error importing OpenAI package: {str(e)}")
        return False

def check_openai_usage():
    """Check the current OpenAI API usage and limits."""
    print_header("Checking OpenAI API Usage")
    try:
        import openai
        from openai import OpenAI
        import requests
        from datetime import datetime
        
        # Explicitly load environment variables
        load_dotenv()
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print_result(False, "OPENAI_API_KEY environment variable is not set")
            return
            
        print(f"Using API key: {api_key[:10]}...{api_key[-5:]}")
        
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Get current date in YYYY-MM-DD format
        today = datetime.now().strftime("%Y-%m-%d")
        
        try:
            # Make a request to the OpenAI API to get usage data
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # Get usage for the current day
            response = requests.get(
                f"https://api.openai.com/v1/usage?date={today}",
                headers=headers
            )
            
            if response.status_code == 200:
                usage_data = response.json()
                print_result(True, f"Successfully retrieved usage data for {today}")
                
                # Display usage information
                print("\nUsage Summary:")
                print(f"Date: {today}")
                
                if 'data' in usage_data and usage_data['data']:
                    total_tokens = sum(item.get('n_context_tokens_total', 0) + item.get('n_generated_tokens_total', 0) 
                                      for item in usage_data['data'])
                    print(f"Total tokens used today: {total_tokens}")
                    
                    # Display usage by model
                    print("\nUsage by Model:")
                    models_usage = {}
                    for item in usage_data['data']:
                        model = item.get('snapshot_id', 'unknown')
                        tokens = item.get('n_context_tokens_total', 0) + item.get('n_generated_tokens_total', 0)
                        if model in models_usage:
                            models_usage[model] += tokens
                        else:
                            models_usage[model] = tokens
                    
                    for model, tokens in models_usage.items():
                        print(f"  - {model}: {tokens} tokens")
                else:
                    print("No usage data available for today.")
            else:
                print_result(False, f"Failed to retrieve usage data: {response.status_code} - {response.text}")
                
        except Exception as e:
            print_result(False, f"Error checking OpenAI API usage: {str(e)}")
    
    except Exception as e:
        print_result(False, f"Error importing required packages: {str(e)}")

def check_model_availability():
    """Check if the GPT-3.5-Turbo-0125 model is available with your API key."""
    print("\nChecking if GPT-3.5-Turbo-0125 model is available...")
    
    try:
        import openai
        
        # Explicitly load environment variables
        load_dotenv()
        
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print_result(False, "OPENAI_API_KEY environment variable is not set")
            return False
            
        print(f"Using API key: {api_key[:10]}...{api_key[-5:]}")
        
        client = openai.OpenAI(api_key=api_key)
        
        # List available models
        models = client.models.list()
        model_ids = [model.id for model in models.data]
        
        # Check if our target model is available
        target_model = "gpt-3.5-turbo-0125"
        is_available = target_model in model_ids
        
        if is_available:
            print_result(True, f"The {target_model} model is available with your API key")
        else:
            print_result(False, f"The {target_model} model is NOT available with your API key")
            print("Available models include:")
            for model_id in sorted(model_ids):
                if "gpt" in model_id:
                    print(f"  - {model_id}")
        
        return is_available
    
    except openai.RateLimitError as e:
        print_result(False, f"Rate limit exceeded when checking model availability: {e}")
        return False
    except ImportError:
        print_result(False, "OpenAI package not installed. Run: pip install openai")
        return False
    except Exception as e:
        print_result(False, f"Error checking model availability: {e}")
        return False

def main():
    """Main function to run the script."""
    # Load environment variables at the start
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Set up RAG pipeline for financial documents")
    
    # File source options
    file_group = parser.add_mutually_exclusive_group()
    file_group.add_argument("--file", help="Path to local PDF file")
    file_group.add_argument("--gcs-file", help="Path to PDF file in GCS bucket")
    
    # Query option
    parser.add_argument("--query", help="Query to run against the document")
    
    # Verification options
    parser.add_argument("--verify-openai", action="store_true", help="Verify OpenAI API key")
    parser.add_argument("--verify-pinecone", action="store_true", help="Verify Pinecone setup")
    parser.add_argument("--check-usage", action="store_true", help="Check OpenAI API usage")
    parser.add_argument("--check-model", action="store_true", help="Check if GPT-3.5-Turbo-0125 is available")
    
    # Cache options
    parser.add_argument("--list-cached", action="store_true", help="List all cached documents in Pinecone")
    parser.add_argument("--list-local-cache", action="store_true", help="List all documents in local cache")
    parser.add_argument("--clear-local-cache", action="store_true", help="Clear the local document cache")
    parser.add_argument("--no-local-cache", action="store_true", help="Disable local caching")
    
    # LlamaParse cache options
    parser.add_argument("--list-llamaparse-cache", action="store_true", help="List all documents in LlamaParse cache")
    parser.add_argument("--clear-llamaparse-cache", action="store_true", help="Clear the LlamaParse cache")
    parser.add_argument("--no-llamaparse-cache", action="store_true", help="Disable LlamaParse caching")
    
    args = parser.parse_args()
    
    # Check OpenAI API usage if requested
    if args.check_usage:
        check_openai_usage()
        return
    
    # Check if GPT-3.5-Turbo-0125 is available
    if args.check_model:
        check_model_availability()
        return
    
    # Verify OpenAI API key if requested
    if args.verify_openai:
        verify_openai_api()
        return
    
    # Verify Pinecone setup if requested
    if args.verify_pinecone:
        verify_pinecone_setup()
        return
        
    # List cached documents if requested
    if args.list_cached:
        list_cached_documents()
        return
    
    # List local cache if requested
    if args.list_local_cache:
        list_local_cache()
        return
    
    # Clear local cache if requested
    if args.clear_local_cache:
        clear_local_cache()
        return
    
    # List LlamaParse cache if requested
    if args.list_llamaparse_cache:
        list_llamaparse_cache()
        return
    
    # Clear LlamaParse cache if requested
    if args.clear_llamaparse_cache:
        clear_llamaparse_cache()
        return
    
    # Process PDF file
    if args.file or args.gcs_file:
        # Verify OpenAI API key
        api_ok = verify_openai_api()
        if not api_ok:
            print("OpenAI API verification failed. Please fix the issues before proceeding.")
            return
        
        # Verify Pinecone setup
        pinecone_ok = verify_pinecone_setup()
        if not pinecone_ok:
            print("Pinecone verification failed. Please fix the issues before proceeding.")
            return
        
        # Check if GPT-3.5-Turbo-0125 is available
        model_ok = check_model_availability()
        if not model_ok:
            print("GPT-3.5-Turbo-0125 is not available with your API key. Please check your OpenAI plan.")
            return
        
        # Get PDF path
        pdf_path = None
        if args.file:
            pdf_path = args.file
        elif args.gcs_file:
            pdf_path = download_from_gcs(args.gcs_file)
        
        if not pdf_path or not os.path.exists(pdf_path):
            print_result(False, f"PDF file not found: {pdf_path}")
            return
        
        # Process PDF and set up RAG pipeline
        use_local_cache = not args.no_local_cache
        use_llamaparse_cache = not args.no_llamaparse_cache
        query_engine = process_pdf_for_rag(pdf_path, args.query, use_local_cache=use_local_cache, use_llamaparse_cache=use_llamaparse_cache)
        
        # Clean up temporary file if downloaded from GCS
        if args.gcs_file and pdf_path and os.path.exists(pdf_path):
            os.remove(pdf_path)
            print(f"Removed temporary file: {pdf_path}")
    else:
        parser.print_help()

def list_cached_documents():
    """List all cached documents in Pinecone."""
    print_header("Cached Documents in Pinecone")
    
    try:
        # Initialize Pinecone
        index_name = "financial-reports"
        pinecone_client = init_pinecone()
        
        if pinecone_client is None:
            print_result(False, "Failed to initialize Pinecone client")
            return
            
        # Connect to the index
        index = pinecone_client.Index(index_name)
        
        # Get stats to check if the registry namespace exists
        stats = index.describe_index_stats()
        namespaces = stats.get("namespaces", {})
        
        if "document_registry" not in namespaces:
            print("No cached documents found in Pinecone.")
            return
            
        # Query all documents in the registry
        # Since we can't fetch all vectors directly, we'll use a dummy query
        query_response = index.query(
            namespace="document_registry",
            vector=[0.0] * 1536,  # Dummy vector
            top_k=100,  # Fetch up to 100 documents
            include_metadata=True
        )
        
        if not query_response.matches:
            print("No cached documents found in Pinecone.")
            return
            
        # Print document information
        print(f"Found {len(query_response.matches)} cached documents:")
        print("\n{:<40} {:<20} {:<20} {:<15} {:<15}".format(
            "Document Name", "Namespace", "Processed At", "Node Count", "Est. Tokens"
        ))
        print("-" * 110)
        
        for i, match in enumerate(query_response.matches):
            metadata = match.metadata
            file_name = metadata.get("file_name", "Unknown")
            namespace = metadata.get("namespace", "Unknown")
            processed_at = metadata.get("processed_at", 0)
            node_count = metadata.get("node_count", 0)
            estimated_tokens = metadata.get("estimated_tokens", "N/A")
            
            # Convert timestamp to readable date
            if processed_at:
                from datetime import datetime
                processed_date = datetime.fromtimestamp(processed_at).strftime('%Y-%m-%d %H:%M:%S')
            else:
                processed_date = "Unknown"
                
            print("{:<40} {:<20} {:<20} {:<15} {:<15}".format(
                file_name, namespace, processed_date, node_count, estimated_tokens
            ))
            
    except Exception as e:
        print_result(False, f"Error listing cached documents: {e}")
        import traceback
        traceback.print_exc()

def list_local_cache():
    """List all documents in the local cache."""
    print_header("Documents in Local Cache")
    
    try:
        cache_dir = get_cache_dir()
        cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.pickle')]
        
        if not cache_files:
            print("No documents found in local cache.")
            return
        
        print(f"Found {len(cache_files)} documents in local cache:")
        print("\n{:<40} {:<20} {:<15} {:<15}".format(
            "Document Name", "Processed At", "Node Count", "Est. Tokens"
        ))
        print("-" * 90)
        
        for cache_file in cache_files:
            try:
                with open(os.path.join(cache_dir, cache_file), 'rb') as f:
                    data = pickle.load(f)
                
                metadata = data.get('metadata', {})
                file_name = metadata.get('file_name', 'Unknown')
                processed_at = metadata.get('processed_at', 0)
                node_count = metadata.get('node_count', 0)
                estimated_tokens = metadata.get('estimated_tokens', 'N/A')
                
                # Convert timestamp to readable date
                if processed_at:
                    from datetime import datetime
                    processed_date = datetime.fromtimestamp(processed_at).strftime('%Y-%m-%d %H:%M:%S')
                else:
                    processed_date = "Unknown"
                
                print("{:<40} {:<20} {:<15} {:<15}".format(
                    file_name, processed_date, node_count, estimated_tokens
                ))
            except Exception as e:
                print(f"Error reading cache file {cache_file}: {e}")
        
    except Exception as e:
        print_result(False, f"Error listing local cache: {e}")

if __name__ == "__main__":
    main() 