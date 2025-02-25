# FinSight AI - Document Processing & Analysis

An intelligent Streamlit application for processing, analyzing, and querying PDF documents using advanced AI capabilities.

## 🌟 Features

- PDF document upload with duplicate detection to Google Cloud Storage
- Secure document handling with authenticated access only
- Structured (tables) and unstructured (text) data extraction using LlamaParse
- Vector embeddings generation and storage with Pinecone
- Context-aware document querying using LangChain and LlamaIndex
- Retrieval Augmented Generation (RAG) pipeline for intelligent document chat
- Interactive chat interface with source attribution
- Production-ready deployment on Google Cloud Platform

## 🔧 Tech Stack

- **Frontend**: Streamlit
- **Storage**: Google Cloud Storage (Project: finsight-ai-nguyen)
- **Document Processing**: LlamaParse
- **Embeddings**: OpenAI (text-embedding-3-small)
- **Vector Storage**: Pinecone
- **Language Model**: GPT-4
- **Query Engine**: LlamaIndex
- **Deployment**: Google Cloud Run & App Engine

## 📁 Project Structure

finsight-ai-nguyen/
├── app/
│   ├── main.py              # Streamlit application
│   ├── utils/
│   │   ├── gcs.py          # GCS operations
│   │   ├── parser.py       # PDF parsing
│   │   ├── embeddings.py   # Vector operations
│   │   └── chatbot.py      # LangChain integration
│   └── components/         # Streamlit components
├── config/
│   └── keys/              # GCP credentials
├── data/
│   ├── raw/              # Original PDFs
│   ├── processed/        # Processed data
│   ├── temp/             # Temporary files
│   └── test/            # Test files
├── scripts/             # Setup and utility scripts
│   ├── setup_gcp.py     # GCP setup
│   ├── setup_pinecone.py # Pinecone setup
│   ├── setup_rag_pipeline.py # RAG pipeline setup
│   └── process_pdf.py   # PDF processing
├── tests/               # Unit tests
├── deployment/         # Deployment configs
├── .env               # Environment variables
└── requirements.txt   # Python dependencies

## 🚀 Quick Start

See [QUICKSTART.md](QUICKSTART.md) for detailed setup and running instructions.

## 📋 Prerequisites

- Python 3.9+ with Anaconda
- Google Cloud Platform account (Project: finsight-ai-nguyen)
- OpenAI API key
- Pinecone API key
- LlamaParse API key

## 🔒 Security Features

- All documents are stored securely in Google Cloud Storage
- No public links are generated for uploaded documents
- Access to documents requires GCP authentication
- Duplicate detection prevents redundant storage
- Metadata is stored securely in Firestore

## 🤖 RAG Pipeline

The application uses a Retrieval Augmented Generation (RAG) pipeline to provide intelligent responses to queries about your documents:

1. **Document Processing**: PDFs are processed using LlamaParse to extract structured text
2. **Chunking**: Documents are split into semantic chunks for better retrieval
3. **Embedding**: Text chunks are converted to vector embeddings using OpenAI
4. **Storage**: Embeddings are stored in Pinecone with namespaces to avoid duplication
5. **Retrieval**: When a query is made, the most relevant chunks are retrieved
6. **Generation**: GPT-4 generates responses based on the retrieved context
7. **Source Attribution**: Responses include references to the source material

## 🏆 Milestones

- **v0.1.0** - Initial project setup
- **v0.2.0** - PDF upload functionality with GCS integration
- **v0.3.0** - Secure document handling with authenticated access only
- **v0.4.0** - RAG pipeline integration with chat interface (CURRENT)

## 📝 License

MIT License
