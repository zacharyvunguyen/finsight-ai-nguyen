# FinSight AI

A financial document analysis tool that leverages Retrieval-Augmented Generation (RAG) to provide intelligent insights from financial reports, SEC filings, and other financial documents.

## Overview

FinSight AI allows users to:
- Upload and process financial PDF documents
- Query documents using natural language
- Receive accurate, contextually relevant answers
- View source attributions for transparency
- Compare information across different reports

## Features

- **Natural Language Querying**: Ask questions about your financial documents in plain English
- **Document Selection**: Filter queries to specific documents of interest
- **Source Attribution**: See exactly where information comes from
- **Conversation Memory**: Chat history is maintained for contextual responses
- **PDF Processing**: Upload and process PDFs with automatic embedding and storage
- **Duplicate Detection**: Avoid processing the same document multiple times

## Technical Architecture

### Core Components

1. **Frontend**: Streamlit web application
2. **Document Processing**: LlamaParse for PDF extraction
3. **Vector Database**: Pinecone for storing and retrieving document embeddings
4. **Language Model**: OpenAI GPT for generating responses
5. **Storage**: Google Cloud Storage for PDF files and Firebase for metadata
6. **Orchestration**: LangChain for the RAG pipeline

### Workflow

1. **Document Processing & Storage**:
   - PDFs are uploaded through the UI
   - Documents are checked for duplicates in Firebase
   - Files are stored in Google Cloud Storage
   - LlamaParse extracts structured text
   - Text is chunked, embedded, and stored in Pinecone
   - Metadata is registered in Firebase

2. **Query Processing**:
   - User selects documents to query
   - Question is embedded and used to retrieve relevant chunks
   - Retrieved context and question are sent to the LLM
   - Response is generated and displayed with source attribution

## Setup and Installation

### Prerequisites

- Python 3.9+
- Conda environment manager
- Accounts for:
  - OpenAI API
  - Pinecone
  - Firebase
  - Google Cloud Platform
  - LlamaParse

### Environment Variables

The following environment variables are required:

```
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_pinecone_index_name
LLAMA_CLOUD_API_KEY=your_llamaparse_api_key
GOOGLE_CLOUD_PROJECT=your_gcp_project_id
GOOGLE_CLOUD_BUCKET=your_gcs_bucket_name
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/gcp/credentials.json
FIREBASE_CREDENTIALS=path/to/your/firebase/credentials.json
```

For development, you can also set:
```
IS_DEVELOPMENT=true
PINECONE_DEV_INDEX_NAME=your_dev_index_name
GOOGLE_CLOUD_BUCKET_DEV=your_dev_bucket_name
```

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/finsight-ai.git
   cd finsight-ai
   ```

2. Create and activate the conda environment:
   ```
   conda create -n finsight-ai python=3.9
   conda activate finsight-ai
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables (create a .env file in the project root)

5. Run the application:
   ```
   streamlit run tests/streamlit_app.py
   ```

## Usage Guide

### Uploading Documents

1. Navigate to the "Upload PDFs" tab
2. Select one or more PDF files
3. Click "Process Selected PDFs"
4. Wait for processing to complete
5. Click "Refresh Available Documents" in the sidebar

### Querying Documents

1. In the sidebar, select which documents you want to query
2. Click "Update Query Engine"
3. Type your question in the chat input or select an example question
4. View the response and source documents
5. Continue the conversation with follow-up questions

## Development

### Project Structure

```
finsight-ai/
├── app/                  # Core application code
│   ├── streamlit/        # Streamlit app components
│   └── utils/            # Utility functions
├── credentials/          # Credential files (gitignored)
├── data/                 # Data storage
├── tests/                # Test scripts
│   ├── streamlit_app.py  # Main Streamlit application
│   └── test_*.py         # Test modules
└── requirements.txt      # Dependencies
```

### Key Files

- `tests/streamlit_app.py`: Main Streamlit application
- `tests/test_llamaparse_pdf.py`: PDF processing and Pinecone integration

## Troubleshooting

### Common Issues

1. **Firebase Connection Issues**:
   - Check that FIREBASE_CREDENTIALS environment variable is set
   - Verify the credentials file exists and has correct permissions

2. **Pinecone Connection Issues**:
   - Ensure PINECONE_API_KEY is set correctly
   - Verify the index exists in your Pinecone account

3. **PDF Processing Failures**:
   - Check LLAMA_CLOUD_API_KEY is valid
   - Ensure the PDF is not corrupted or password-protected

### Demo Mode

If external services are unavailable, the application will fall back to demo mode with sample documents for testing.

## License

[MIT License](LICENSE)
