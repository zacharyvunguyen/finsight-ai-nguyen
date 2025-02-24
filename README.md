# FinSight AI - Document Processing & Analysis

An intelligent Streamlit application for processing, analyzing, and querying PDF documents using advanced AI capabilities.

## 🌟 Features

- PDF document upload with duplicate detection to Google Cloud Storage
- Structured (tables) and unstructured (text) data extraction using LlamaParse
- Vector embeddings generation and storage with Pinecone
- Context-aware document querying using LangChain and LlamaIndex
- Production-ready deployment on Google Cloud Platform

## 🔧 Tech Stack

- **Frontend**: Streamlit
- **Storage**: Google Cloud Storage (Project: finsight-ai-nguyen)
- **Document Processing**: LlamaParse
- **Embeddings**: OpenAI (text-embedding-3-small)
- **Vector Storage**: Pinecone
- **Query Engine**: LangChain + LlamaIndex
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
│   └── test/            # Test files
├── tests/               # Unit tests
├── deployment/         # Deployment configs
├── .env               # Environment variables
└── requirements.txt   # Python dependencies

## 🚀 Quick Start

See [quickstart.md](quickstart.md) for detailed setup and running instructions.

## 📋 Prerequisites

- Python 3.9+ with Anaconda
- Google Cloud Platform account (Project: finsight-ai-nguyen)
- OpenAI API key
- Pinecone API key
- LlamaParse API key

## 📝 License

MIT License
