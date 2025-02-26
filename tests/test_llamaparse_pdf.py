#!/usr/bin/env python3
"""
Enhanced script for processing PDFs with LlamaParse and storing in Pinecone.
Features:
1. GCP Bucket integration
2. Processing state tracking
3. Optimized to avoid reprocessing
4. Single recursive query engine
5. Separate index for development/testing
"""

import os
import sys
import json
import hashlib
from datetime import datetime
import requests
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Union
from dotenv import load_dotenv

def verify_environment():
    """Verify that we're running in the correct conda environment."""
    current_env = os.environ.get('CONDA_DEFAULT_ENV')
    if current_env != 'finsight-ai':
        print("❌ Error: This script must be run in the 'finsight-ai' conda environment")
        print("Please run:")
        print("    conda activate finsight-ai")
        print("Then try again.")
        sys.exit(1)
    print("✅ Running in correct conda environment: finsight-ai")

# Verify environment before importing other dependencies
verify_environment()

from llama_parse import LlamaParse
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from google.cloud import storage
from pinecone import Pinecone, ServerlessSpec

# Constants
PROCESSING_STATE_FILE = "data/processing_state.json"
TEMP_DIR = "data/temp"

def print_header(message: str) -> None:
    """Print a formatted header message."""
    print("\n" + "="*80)
    print(f" {message}")
    print("="*80)

def print_result(success: bool, message: str) -> None:
    """Print a formatted result message."""
    status = "✅" if success else "❌"
    print(f"{status} {message}")

def load_environment() -> dict:
    """Load and validate environment variables."""
    load_dotenv()
    
    # Set development mode
    is_dev = os.environ.get('IS_DEVELOPMENT', 'true').lower() == 'true'
    
    # Set Pinecone environment and dev index name
    os.environ['PINECONE_ENVIRONMENT'] = 'us-east1-aws'
    
    # Set development names with -dev suffix
    if is_dev:
        if 'PINECONE_DEV_INDEX_NAME' not in os.environ:
            os.environ['PINECONE_DEV_INDEX_NAME'] = 'dev-financial-docs'
        if 'GOOGLE_CLOUD_BUCKET_DEV' not in os.environ:
            prod_bucket = os.environ.get('GOOGLE_CLOUD_BUCKET', 'finsight-ai-docs')
            os.environ['GOOGLE_CLOUD_BUCKET_DEV'] = f"{prod_bucket}-dev"
    
    required_vars = [
        'LLAMA_CLOUD_API_KEY',
        'OPENAI_API_KEY',
        'PINECONE_API_KEY',
        'PINECONE_INDEX_NAME',
        'GOOGLE_CLOUD_PROJECT',
        'GOOGLE_APPLICATION_CREDENTIALS',
        'GOOGLE_CLOUD_BUCKET'
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
    
    # Set environment variables with defaults
    env['IS_DEVELOPMENT'] = is_dev
    env['PINECONE_ENVIRONMENT'] = os.environ['PINECONE_ENVIRONMENT']
    env['PINECONE_DEV_INDEX_NAME'] = os.environ.get('PINECONE_DEV_INDEX_NAME')
    env['GOOGLE_CLOUD_BUCKET'] = env['GOOGLE_CLOUD_BUCKET']  # Use consistent naming
    env['GOOGLE_CLOUD_BUCKET_DEV'] = os.environ.get('GOOGLE_CLOUD_BUCKET_DEV')
    
    # Use development resources if in dev mode
    if is_dev:
        env['ACTIVE_BUCKET'] = env['GOOGLE_CLOUD_BUCKET_DEV']
        env['ACTIVE_INDEX'] = env['PINECONE_DEV_INDEX_NAME']
    else:
        env['ACTIVE_BUCKET'] = env['GOOGLE_CLOUD_BUCKET']
        env['ACTIVE_INDEX'] = env['PINECONE_INDEX_NAME']
    
    # Log configuration
    print_header("Configuration")
    print(f"🔧 Mode: {'Development' if is_dev else 'Production'}")
    print(f"🌎 Environment: {env['PINECONE_ENVIRONMENT']}")
    print(f"📊 Production Resources:")
    print(f"   - Index: {env['PINECONE_INDEX_NAME']}")
    print(f"   - Bucket: {env['GOOGLE_CLOUD_BUCKET']}")
    print(f"🧪 Development Resources:")
    print(f"   - Index: {env['PINECONE_DEV_INDEX_NAME']}")
    print(f"   - Bucket: {env['GOOGLE_CLOUD_BUCKET_DEV']}")
    print(f"✨ Active Resources:")
    print(f"   - Index: {env['ACTIVE_INDEX']}")
    print(f"   - Bucket: {env['ACTIVE_BUCKET']}")
    print(f"☁️  GCP Project: {env['GOOGLE_CLOUD_PROJECT']}")
    print(f"📄 GCP Credentials: {env['GOOGLE_APPLICATION_CREDENTIALS']}")
    
    # Verify GCP credentials exist
    if not os.path.exists(env['GOOGLE_APPLICATION_CREDENTIALS']):
        raise ValueError(f"GCP credentials file not found at: {env['GOOGLE_APPLICATION_CREDENTIALS']}")
    
    return env

class ProcessingState:
    """Manages processing state to avoid reprocessing documents."""
    
    def __init__(self, state_file: str = PROCESSING_STATE_FILE):
        self.state_file = state_file
        self.state = self._load_state()
    
    def _load_state(self) -> Dict:
        """Load processing state from file."""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {'processed_files': {}}
    
    def _save_state(self) -> None:
        """Save processing state to file."""
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def get_file_hash(self, file_path: str) -> str:
        """Calculate file hash."""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def is_processed(self, file_path: str, index_name: str) -> bool:
        """Check if file has been processed for given index."""
        file_hash = self.get_file_hash(file_path)
        return (file_hash in self.state['processed_files'] and 
                index_name in self.state['processed_files'][file_hash])
    
    def mark_processed(self, file_path: str, index_name: str, metadata: dict) -> None:
        """Mark file as processed for given index."""
        file_hash = self.get_file_hash(file_path)
        if file_hash not in self.state['processed_files']:
            self.state['processed_files'][file_hash] = {}
        
        self.state['processed_files'][file_hash][index_name] = {
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata
        }
        self._save_state()

class GCPBucketManager:
    """Manages interactions with Google Cloud Storage bucket."""
    
    def __init__(self, project_id: str, bucket_name: str):
        self.project_id = project_id
        self.bucket_name = bucket_name
        
        # Verify GCP credentials are properly set
        if not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable is not set")
        if not os.path.exists(os.environ['GOOGLE_APPLICATION_CREDENTIALS']):
            raise ValueError(f"GCP credentials file not found at: {os.environ['GOOGLE_APPLICATION_CREDENTIALS']}")
        
        print(f"🔑 Using GCP credentials from: {os.environ['GOOGLE_APPLICATION_CREDENTIALS']}")
        self.client = storage.Client(project=project_id)
        self.bucket = self._get_or_create_bucket()
    
    def _get_or_create_bucket(self) -> storage.Bucket:
        """Get bucket or create if it doesn't exist."""
        try:
            bucket = self.client.get_bucket(self.bucket_name)
            print_result(True, f"Using existing bucket: {self.bucket_name}")
            return bucket
        except Exception:
            print_result(True, f"Creating new bucket: {self.bucket_name}")
            return self._create_bucket()
    
    def _create_bucket(self) -> storage.Bucket:
        """Create a new bucket."""
        try:
            bucket = self.client.create_bucket(
                self.bucket_name,
                location="us-east1"
            )
            
            # Set lifecycle policy to delete files after 7 days if in dev bucket
            if self.bucket_name.endswith('-dev'):
                lifecycle_rules = [
                    {
                        'action': {'type': 'Delete'},
                        'condition': {'age': 7}  # 7 days
                    }
                ]
                bucket.lifecycle_rules = lifecycle_rules
                bucket.patch()
            
            print_result(True, f"Created bucket: {self.bucket_name}")
            return bucket
            
        except Exception as e:
            print_result(False, f"Error creating bucket: {str(e)}")
            raise
    
    def list_pdfs(self) -> List[str]:
        """List all PDFs in the bucket."""
        return [blob.name for blob in self.bucket.list_blobs() if blob.name.lower().endswith('.pdf')]
    
    def download_pdf(self, blob_name: str, local_path: str) -> str:
        """Download a PDF from the bucket."""
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            blob = self.bucket.blob(blob_name)
            blob.download_to_filename(local_path)
            print_result(True, f"Downloaded {blob_name} to {local_path}")
            return local_path
        except Exception as e:
            print_result(False, f"Error downloading {blob_name}: {str(e)}")
            raise

def process_pdf_with_llamaparse(pdf_path: str) -> List[Document]:
    """Process a PDF file with LlamaParse and return the extracted documents."""
    try:
        parser = LlamaParse(
            result_type="markdown",
            api_key=os.environ.get("LLAMA_CLOUD_API_KEY"),
            verbose=True
        )
        
        print_header(f"Processing {pdf_path} with LlamaParse")
        documents = parser.load_data(pdf_path)
        
        print(f"Number of documents extracted: {len(documents)}")
        for i, doc in enumerate(documents):
            print(f"\nDocument {i+1}:")
            print(f"Metadata: {doc.metadata}")
            print(f"Text length: {len(doc.text)} characters")
        
        return documents
    
    except Exception as e:
        print_result(False, f"Error processing PDF with LlamaParse: {str(e)}")
        raise

def create_pinecone_index(documents: List[Document], index_name: str, document_id: str) -> None:
    """Create embeddings and store in Pinecone."""
    try:
        print_header("Creating embeddings and storing in Pinecone")
        
        embed_model = OpenAIEmbedding(
            model="text-embedding-3-small",
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=1536,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
        
        index = pc.Index(index_name)
        
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            texts = [doc.text for doc in batch]
            embeddings = embed_model.get_text_embedding_batch(texts)
            
            vectors = []
            for j, (doc, embedding) in enumerate(zip(batch, embeddings)):
                vector_id = f"{document_id}_doc_{i+j}"
                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {
                        "text": doc.text[:1000],
                        "metadata": str(doc.metadata),
                        "document_id": document_id,
                        "timestamp": datetime.now().isoformat(),
                        "source": doc.metadata.get('source', ''),
                    }
                })
            
            index.upsert(vectors=vectors)
            print_result(True, f"Uploaded batch {i//batch_size + 1}")
        
        print_result(True, f"Successfully stored document {document_id} in Pinecone")
        
    except Exception as e:
        print_result(False, f"Error storing in Pinecone: {str(e)}")
        raise

def setup_llm_and_embeddings() -> None:
    """Setup LLM and embedding models as global settings."""
    try:
        print_header("Setting up LLM and embedding models")
        
        embed_model = OpenAIEmbedding(
            model="text-embedding-3-small",
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        llm = OpenAI(
            model="gpt-3.5-turbo-0125",
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        
        Settings.llm = llm
        Settings.embed_model = embed_model
        
        print_result(True, "Successfully setup LLM and embedding models")
        
    except Exception as e:
        print_result(False, f"Error setting up models: {str(e)}")
        raise

def create_query_engine(documents: Optional[List[Document]], document_id: Optional[Union[str, List[str]]] = None) -> VectorStoreIndex:
    """Create recursive query engine.
    
    Args:
        documents: List of documents to index, or None if querying existing documents
        document_id: Single document ID or list of document IDs to filter by
    """
    try:
        print_header("Creating recursive query engine")
        
        # If documents are provided, create nodes and index them
        if documents:
            node_parser = MarkdownElementNodeParser(
                llm=Settings.llm,
                num_workers=8
            )
            
            nodes = node_parser.get_nodes_from_documents(documents)
            base_nodes, objects = node_parser.get_nodes_and_objects(nodes)
            
            def get_page_nodes(docs, separator="\\n---\\n"):
                return [Document(text=chunk, metadata=doc.metadata.copy())
                       for doc in docs
                       for chunk in doc.text.split(separator)]
            
            page_nodes = get_page_nodes(documents)
            all_nodes = base_nodes + objects + page_nodes
            
            recursive_index = VectorStoreIndex(nodes=all_nodes)
        else:
            # If no documents provided, create empty index (will load from storage)
            recursive_index = VectorStoreIndex([])
        
        # Set up document filter
        if document_id:
            if isinstance(document_id, list):
                filter_dict = {"document_id": {"$in": document_id}}
            else:
                filter_dict = {"document_id": document_id}
        else:
            filter_dict = None
        
        # Create query engine with filters
        query_engine = recursive_index.as_query_engine(
            similarity_top_k=5,
            filters=filter_dict
        )
        
        print_result(True, "Successfully created query engine")
        return query_engine
        
    except Exception as e:
        print_result(False, f"Error creating query engine: {str(e)}")
        raise

def query_document(query: str, query_engine) -> None:
    """Query the document using recursive engine."""
    try:
        print_header(f"Querying: {query}")
        
        response = query_engine.query(query)
        print("\nResponse:")
        print(response)
        
        print("\n***********Source Nodes***********")
        for i, node in enumerate(response.source_nodes[:2]):
            print(f"\nSource {i+1}:")
            print(node.get_content()[:500] + "..." if len(node.get_content()) > 500 else node.get_content())
        
    except Exception as e:
        print_result(False, f"Error querying document: {str(e)}")
        raise

def interactive_query_mode(query_engine) -> None:
    """Enter interactive query mode."""
    print_header("Entering interactive query mode")
    print("Type 'exit' to quit")
    
    while True:
        try:
            query = input("\nEnter your query: ").strip()
            if query.lower() == 'exit':
                break
            
            if query:
                query_document(query, query_engine)
                
        except KeyboardInterrupt:
            print("\nExiting interactive mode...")
            break
        except Exception as e:
            print_result(False, f"Error: {str(e)}")

def process_pdf(pdf_path: str, document_id: str, index_name: str, state: ProcessingState) -> List[Document]:
    """Process a single PDF if not already processed."""
    if state.is_processed(pdf_path, index_name):
        print_result(True, f"Skipping {pdf_path} - already processed")
        return []
    
    documents = process_pdf_with_llamaparse(pdf_path)
    create_pinecone_index(documents, index_name, document_id)
    
    state.mark_processed(pdf_path, index_name, {
        'document_id': document_id,
        'num_documents': len(documents)
    })
    
    return documents

def main():
    """Main function to process PDFs from GCP bucket and store in Pinecone."""
    try:
        # Load environment variables
        env = load_environment()
        
        # Initialize processing state
        state = ProcessingState()
        
        # Setup LLM and embeddings
        setup_llm_and_embeddings()
        
        # Use active index
        index_name = env['ACTIVE_INDEX']
        print_header(f"Using index: {index_name}")
        
        # Initialize GCP bucket manager with active bucket
        gcp_manager = GCPBucketManager(
            env['GOOGLE_CLOUD_PROJECT'],
            env['ACTIVE_BUCKET']
        )
        
        # Handle development mode with test PDF
        if env['IS_DEVELOPMENT']:
            test_pdf_path = os.environ.get('TEST_PDF_PATH')
            if test_pdf_path and os.path.exists(test_pdf_path):
                print_header(f"Development Mode: Using test PDF at {test_pdf_path}")
                
                # Get the filename from the test PDF path
                test_pdf_filename = os.path.basename(test_pdf_path)
                
                # Check if the file already exists in the bucket
                existing_pdfs = gcp_manager.list_pdfs()
                if test_pdf_filename not in existing_pdfs:
                    print(f"📤 Uploading test PDF to development bucket: {test_pdf_filename}")
                    blob = gcp_manager.bucket.blob(test_pdf_filename)
                    blob.upload_from_filename(test_pdf_path)
                    print_result(True, f"Uploaded {test_pdf_filename} to bucket {env['ACTIVE_BUCKET']}")
                else:
                    print_result(True, f"Test PDF already exists in bucket: {test_pdf_filename}")
                
                # Download to temp directory (this ensures consistent processing path)
                local_path = os.path.join(TEMP_DIR, test_pdf_filename)
                gcp_manager.download_pdf(test_pdf_filename, local_path)
                
                # Create document_id from filename
                document_id = os.path.splitext(test_pdf_filename)[0]
                
                # Process PDF if not already processed
                documents = process_pdf(local_path, document_id, index_name, state)
                if documents:
                    all_documents = documents
                else:
                    print_result(True, "Test PDF already processed in Pinecone")
                    return
            else:
                print_result(False, f"Test PDF not found at {test_pdf_path}")
                return
        else:
            # Process PDFs from bucket
            all_documents = []
            pdf_files = gcp_manager.list_pdfs()
            
            print_header(f"Found {len(pdf_files)} PDFs in bucket {env['ACTIVE_BUCKET']}")
            for pdf_file in pdf_files:
                print(f"📄 Processing: {pdf_file}")
                local_path = os.path.join(TEMP_DIR, pdf_file)
                gcp_manager.download_pdf(pdf_file, local_path)
                
                # Create document_id from filename
                document_id = os.path.splitext(os.path.basename(pdf_file))[0]
                
                # Process PDF if not already processed
                documents = process_pdf(local_path, document_id, index_name, state)
                all_documents.extend(documents)
        
        if not all_documents:
            print_result(True, "No new documents to process")
            return
        
        # Create query engine for all documents
        query_engine = create_query_engine(all_documents)
        
        # Example queries
        example_queries = [
            "What were the purchases of marketable securities in 2020?",
            "What were the effective interest rates of all debt issuances in 2021?",
            "What was the impact of the U.S. Tax Cuts and Jobs Act of 2017 on income taxes in 2020?"
        ]
        
        print_header(f"Running example queries on index: {index_name}")
        for query in example_queries:
            query_document(query, query_engine)
        
        # Enter interactive mode
        print_header(f"Interactive query mode - Using index: {index_name}")
        interactive_query_mode(query_engine)
        
        print_header("Processing Complete")
        print_result(True, "PDF processing and RAG system ready")
        
    except Exception as e:
        print_result(False, f"Error in main function: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 