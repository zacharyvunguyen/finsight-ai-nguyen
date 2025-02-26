# FinSight AI Test Suite

This directory contains test scripts for verifying the functionality of the FinSight AI system.

## Test Scripts

### `test_services.py`

A comprehensive test suite for all external services used in the project:

- Tests GCP connectivity and services (Storage)
- Tests Pinecone connectivity and operations
- Tests OpenAI API connectivity
- Tests LlamaParse API connectivity
- Provides detailed error messages and troubleshooting steps

**Usage:**
```bash
python tests/test_services.py
```

### `test_rag_pipeline.py`

Tests the complete RAG (Retrieval Augmented Generation) pipeline:

- Tests LlamaIndex setup
- Tests LlamaParse document parsing
- Tests node creation and processing
- Tests Pinecone vector storage
- Tests the full query pipeline

**Usage:**
```bash
pytest tests/test_rag_pipeline.py -v
```

### `streamlit_app.py`

A Streamlit application for testing and demonstrating the RAG pipeline:

- Provides a user-friendly interface for querying documents
- Allows selection of specific documents to query
- Displays chat history and source attribution
- Includes an About tab with information about the application
- Features enhanced entity recognition and modern UI

**Usage:**
```bash
streamlit run tests/streamlit_app.py
```

For detailed information about the Streamlit application, see [STREAMLIT_APP.md](STREAMLIT_APP.md).

### `test_direct_pinecone_query.py`

Tests direct querying of the Pinecone vector database:

- Tests connection to Pinecone
- Tests vector similarity search
- Tests metadata filtering

**Usage:**
```bash
python tests/test_direct_pinecone_query.py
```

### `test_query_pinecone.py`

Tests the query functionality against Pinecone:

- Tests embedding generation
- Tests vector search
- Tests result processing

**Usage:**
```bash
python tests/test_query_pinecone.py
```

### `test_rag_with_openai.py`

Tests the RAG pipeline with OpenAI models:

- Tests OpenAI embedding generation
- Tests OpenAI completion models
- Tests the full RAG pipeline with OpenAI

**Usage:**
```bash
python tests/test_rag_with_openai.py
```

### `test_llamaparse_pdf.py`

Tests the LlamaParse PDF parsing functionality:

- Tests PDF upload to LlamaParse
- Tests structured data extraction
- Tests text extraction

**Usage:**
```bash
python tests/test_llamaparse_pdf.py
```

## Archived Tests

The `archive` directory contains older test scripts that have been replaced by the more comprehensive test suites above:

- `test_gcp_connection.py` - Tests GCP connectivity (replaced by `test_services.py`)
- `test_pinecone_connection.py` - Tests Pinecone connectivity (replaced by `test_services.py`)
- `create_pinecone_index.py` - Creates a Pinecone index (functionality now in main scripts)
- `test_gcs.py` - Tests GCS functionality (replaced by `test_services.py`)
- `test_gcs_upload.py` - Tests GCS upload functionality (replaced by `test_services.py`)

## Running All Tests

To run all active tests:

```bash
# Run service tests
python tests/test_services.py

# Run RAG pipeline tests
pytest tests/test_rag_pipeline.py -v

# Run Streamlit app
streamlit run tests/streamlit_app.py
``` 