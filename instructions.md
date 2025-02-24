# FinSight AI - Quick Setup Guide

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.9+ with Anaconda
- Google Cloud Platform account
- Required API keys:
  - OpenAI API key
  - Pinecone API key
  - LlamaParse API key

### 2. Initial Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/finsight-ai-nguyen.git
cd finsight-ai-nguyen

# Create and activate conda environment
conda create -n finsight python=3.9
conda activate finsight

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure API Keys
1. Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

2. Edit `.env` with your API keys:
```
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
LLAMAPARSE_API_KEY=your_llamaparse_key
GCP_STORAGE_BUCKET=finsight-reports-bucket
GOOGLE_CLOUD_PROJECT=finsight-ai-nguyen
```

### 4. Setup Google Cloud
```bash
# Run the setup script
./setup_project.sh
```
This will:
- Configure GCP credentials
- Create necessary buckets
- Setup required services

### 5. Run the Application
```bash
# Start the Streamlit app
cd app
streamlit run main.py
```

## 📝 Basic Usage

### Upload Documents
1. Open app in browser (usually http://localhost:8501)
2. Click "Upload" button in the sidebar
3. Select your PDF file
4. Click "Upload & Process" button
5. Wait for processing to complete
6. You'll be automatically redirected to the Documents tab

### View Documents
1. Click "Documents" in the sidebar
2. View the list of all uploaded documents
3. Click "View" to see details of a specific document
4. Access the document in GCP Console (requires authentication)

### Query Documents
1. Go to Query tab in the sidebar
2. Type your question
3. View results with source citations

## 🔒 Security Features

### Document Access
- All documents are stored securely in Google Cloud Storage
- No public links are generated for uploaded documents
- Access to documents requires GCP authentication
- Only authenticated users can view or download files

### Best Practices
- Keep your GCP credentials secure
- Don't share your service account key
- Regularly rotate credentials
- Monitor access logs in GCP Console

## 🔧 Troubleshooting

### Common Issues
1. **Upload Fails**
   - Check PDF file format
   - Verify API keys in `.env`
   - Ensure GCP setup is complete
   - Check GCP credentials are properly loaded

2. **Query Not Working**
   - Confirm document upload finished
   - Check OpenAI API key
   - Verify internet connection

### Need Help?
- Check the logs in `app/logs`
- Review GCP Console for storage issues
- Contact support team

## 📚 Resources
- [GCP Console](https://console.cloud.google.com)
- [Streamlit Docs](https://docs.streamlit.io)
- [Project Repository](https://github.com/yourusername/finsight-ai-nguyen) 