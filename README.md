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
- Enhanced name recognition for queries about specific individuals
- Modern, responsive UI with compact information display

## 🔧 Tech Stack

- **Frontend**: Streamlit
- **Storage**: Google Cloud Storage (Project: finsight-ai-nguyen)
- **Document Processing**: LlamaParse
- **Embeddings**: OpenAI (text-embedding-3-small)
- **Vector Storage**: Pinecone
- **Language Model**: GPT-3.5 Turbo
- **Query Engine**: LangChain
- **Deployment**: Google Cloud Run & App Engine

## 📁 Project Structure

finsight-ai-nguyen/
├── app/
│   ├── streamlit/
│   │   ├── app.py          # Streamlit application
│   │   └── README.md       # Streamlit app documentation
│   ├── main.py             # Main application entry point
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
│   ├── setup.py         # Unified setup script
│   ├── setup/           # Setup modules
│   │   ├── setup_gcp.py     # GCP setup
│   │   ├── setup_pinecone.py # Pinecone setup
│   │   └── setup_rag_pipeline.py # RAG pipeline setup
│   └── utils/           # Utility functions
│       └── common.py    # Common utility functions
├── tests/               # Unit tests
│   ├── test_llamaparse_pdf.py # LlamaParse PDF test
│   ├── test_gcs_pdf.py  # GCS PDF test
│   ├── test_rag_with_openai.py # RAG with OpenAI test
│   ├── STREAMLIT_APP.md # Streamlit app documentation
│   └── README.md        # Test suite documentation
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
6. **Generation**: GPT-3.5 Turbo generates responses based on the retrieved context
7. **Source Attribution**: Responses include references to the source material

## 📱 Streamlit Application

The project includes a Streamlit application for interacting with the RAG pipeline:

- **Document Selection**: Choose which documents to query
- **Natural Language Interface**: Ask questions in plain English
- **Source Attribution**: See exactly where information comes from
- **Modern UI**: Clean, responsive design with intuitive navigation
- **Enhanced Entity Recognition**: Improved handling of queries about specific individuals

For detailed information about the Streamlit application, see [tests/STREAMLIT_APP.md](tests/STREAMLIT_APP.md).

## 🏆 Milestones

- **v0.1.0** - Initial project setup
- **v0.2.0** - PDF upload functionality with GCS integration
- **v0.3.0** - Secure document handling with authenticated access only
- **v0.4.0** - RAG pipeline integration with chat interface (CURRENT)
- **v0.4.1** - Enhanced UI with modern design and improved UX
- **v0.4.2** - Advanced entity recognition and improved query handling

## 📝 License

MIT License
