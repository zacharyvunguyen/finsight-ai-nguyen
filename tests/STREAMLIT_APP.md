# FinSight AI Streamlit Application

## Overview

The FinSight AI Streamlit application provides a user-friendly interface for querying financial documents using Retrieval-Augmented Generation (RAG). It allows users to upload PDF documents, process them, and ask natural language questions about their content.

## Features

### Document Selection
- Select specific documents to query from available documents in Pinecone
- Documents are organized by type (10-K, 10-Q, 8-K, Uploaded PDFs)
- Filter queries to focus on relevant documents

### Natural Language Queries
- Ask questions in plain English about your financial documents
- Get contextually relevant answers based on the selected documents
- Follow up with additional questions that maintain conversation context

### Source Attribution
- See exactly which documents and sections provided the information
- Expand source documents to view the original context
- Verify information directly from the source material

### Conversation Memory
- Chat history is maintained throughout the session
- Follow-up questions can reference previous queries and answers
- Reset conversation when starting a new topic

### PDF Upload and Processing
- Upload PDF files directly through the interface
- Automatic processing with LlamaParse for text extraction
- Embedding generation and storage in Pinecone
- Duplicate detection to avoid processing the same document twice

### UI/UX Enhancements
- Clean, modern interface with intuitive navigation
- Responsive design that works on different screen sizes
- Clear status indicators for processing and query operations
- Example questions to help users get started

## Application Structure

The application is organized into three main tabs:

1. **Chat Tab**: The main interface for querying documents and viewing responses
2. **About Tab**: Information about the application, technologies, and advantages
3. **Upload PDFs Tab**: Interface for uploading and processing new documents

### Sidebar
- Document selection controls
- Index information display
- Refresh documents button
- Reset chat button

## Technical Implementation

### Session State Management
The application uses Streamlit's session state to maintain:
- Query engine configuration
- QA chain for document retrieval
- Chat history
- Active documents selection
- Uploaded PDF information
- Environment variables

### Document Processing Pipeline
1. PDF upload through the Streamlit interface
2. Hash calculation for duplicate detection
3. Firebase check for existing documents
4. Google Cloud Storage upload for persistence
5. LlamaParse processing for text extraction
6. Pinecone storage for vector embeddings
7. Firebase registration for metadata tracking

### Query Processing Pipeline
1. Document selection in the sidebar
2. Query engine setup with selected documents
3. User query input through chat interface
4. Retrieval of relevant document chunks from Pinecone
5. Context assembly with retrieved chunks
6. LLM response generation with OpenAI
7. Display of answer and source attribution

## Usage Guide

### Getting Started
1. Open the application in your browser
2. Click "Refresh Available Documents" in the sidebar
3. Select one or more documents to query
4. Click "Update Query Engine" to prepare the system
5. Start asking questions in the chat interface

### Uploading New Documents
1. Navigate to the "Upload PDFs" tab
2. Click "Choose PDF files" and select one or more files
3. Click "Process Selected PDFs"
4. Wait for processing to complete (status updates will be shown)
5. Return to the Chat tab and refresh available documents

### Effective Querying
- Be specific in your questions for more accurate answers
- Reference specific metrics, sections, or time periods when relevant
- Use follow-up questions to drill down into details
- Try example questions to see the system's capabilities

## Dependencies

The Streamlit application relies on several key libraries:
- `streamlit`: Web application framework
- `langchain`: RAG pipeline orchestration
- `pinecone`: Vector database for document storage
- `openai`: Language model and embeddings
- `firebase_admin`: Document metadata storage
- `google.cloud.storage`: PDF file storage
- `llama_cloud_services`: PDF parsing

## Environment Variables

The application requires several environment variables:
- `OPENAI_API_KEY`: For language model and embeddings
- `PINECONE_API_KEY`: For vector database access
- `LLAMA_CLOUD_API_KEY`: For PDF parsing
- `GOOGLE_CLOUD_BUCKET`: For PDF storage
- `FIREBASE_CREDENTIALS`: For metadata storage

## Error Handling

The application includes robust error handling:
- Fallback to demo mode if external services are unavailable
- Clear error messages for connection issues
- Graceful handling of processing failures
- Status indicators for all operations

## Future Enhancements

Planned improvements for the application:
- Multi-user support with authentication
- Enhanced visualization of financial metrics
- Comparative analysis across multiple documents
- Custom document collections and saved queries
- Export functionality for answers and sources 