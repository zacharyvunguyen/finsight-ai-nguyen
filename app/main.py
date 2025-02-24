import streamlit as st
import tempfile
import os
import time
import sys
from pathlib import Path
from dotenv import load_dotenv
from utils.gcs import GCSManager

# Load environment variables
load_dotenv()

# Set up Google Cloud credentials
def setup_gcp_credentials():
    """Set up Google Cloud credentials"""
    # Try to find the credentials file
    possible_paths = [
        'config/keys/finsight-ai-nguyen-89af45b3c2c0.json',  # Project root
        '../config/keys/finsight-ai-nguyen-89af45b3c2c0.json',  # From app directory
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                    'config/keys/finsight-ai-nguyen-89af45b3c2c0.json')  # Absolute path
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.abspath(path)
            st.sidebar.success(f"✅ GCP credentials loaded from: {path}")
            return True
    
    st.sidebar.error("❌ GCP credentials file not found!")
    return False

# Page configuration
st.set_page_config(
    page_title="FinSight AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean look
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        font-weight: 500;
    }
    .upload-header {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .file-info {
        margin-top: 1rem;
        padding: 1rem;
        border-radius: 5px;
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "upload"

def set_active_tab(tab):
    st.session_state.active_tab = tab

# Sidebar
with st.sidebar:
    st.title("FinSight AI")
    st.markdown("---")
    
    # Navigation
    st.button("📤 Upload", on_click=set_active_tab, args=("upload",), 
              disabled=st.session_state.active_tab=="upload")
    st.button("📋 Documents", on_click=set_active_tab, args=("documents",), 
              disabled=st.session_state.active_tab=="documents")
    st.button("❓ Query", on_click=set_active_tab, args=("query",), 
              disabled=st.session_state.active_tab=="query")
    
    st.markdown("---")
    
    # Stats
    st.subheader("Statistics")
    st.markdown(f"**Uploaded Files:** {len(st.session_state.uploaded_files)}")
    
    # Help section
    st.markdown("---")
    st.markdown("### Need Help?")
    st.markdown("Check the [documentation](https://github.com/yourusername/finsight-ai-nguyen) for more information.")

# Upload Section
def upload_section():
    st.markdown('<p class="upload-header">Upload PDF Documents</p>', unsafe_allow_html=True)
    
    # Info box
    st.markdown("""
    <div class="info-box">
        <strong>📌 Note:</strong> Upload PDF documents to analyze. 
        The system will automatically check for duplicates and process the document.
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'], key="pdf_uploader")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("Upload & Process", disabled=uploaded_file is None):
            if uploaded_file:
                process_upload(uploaded_file)
    
    with col2:
        if st.button("Clear", disabled=uploaded_file is None):
            st.session_state.pdf_uploader = None
            st.rerun()

# Documents Section
def documents_section():
    st.markdown('<p class="upload-header">Uploaded Documents</p>', unsafe_allow_html=True)
    
    # Debug information
    st.sidebar.markdown("### Debug Info")
    st.sidebar.markdown(f"Files in session: {len(st.session_state.uploaded_files)}")
    
    # Check if we need to load files from GCS
    if not st.session_state.uploaded_files:
        with st.spinner("Loading documents from storage..."):
            try:
                # Try to load files from GCS
                gcs_manager = GCSManager()
                files = gcs_manager.list_files()
                
                if files:
                    st.info(f"Found {len(files)} files in storage. Loading metadata...")
                    # We have files but no session state, let's populate it
                    for file_path in files:
                        if file_path.startswith('uploads/'):
                            file_name = file_path.replace('uploads/', '')
                            gcs_uri = f"gs://{gcs_manager.bucket_name}/{file_path}"
                            links = get_gcs_links(gcs_uri)
                            
                            # Add to session state
                            st.session_state.uploaded_files.append({
                                'name': file_name,
                                'status': 'success',
                                'gcs_uri': gcs_uri,
                                'console_url': links['console_url'],
                                'metadata': {}  # We don't have metadata without the hash
                            })
            except Exception as e:
                st.error(f"Error loading documents: {str(e)}")
    
    if not st.session_state.uploaded_files:
        st.info("No documents have been uploaded yet.")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Upload a Document"):
                st.session_state.active_tab = "upload"
                st.rerun()
        return
    
    # Display uploaded files
    st.markdown(f"### {len(st.session_state.uploaded_files)} Documents Found")
    
    # Create a table view
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.markdown("**Document Name**")
    with col2:
        st.markdown("**Status**")
    with col3:
        st.markdown("**Actions**")
    
    st.markdown("---")
    
    for idx, file_info in enumerate(st.session_state.uploaded_files):
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.markdown(f"**{file_info['name']}**")
        
        with col2:
            status = file_info['status']
            if status == 'success':
                st.markdown("✅ Success")
            elif status == 'duplicate':
                st.markdown("⚠️ Duplicate")
            else:
                st.markdown(f"❓ {status}")
        
        with col3:
            if st.button("View", key=f"view_{idx}"):
                with st.expander(f"Details for {file_info['name']}", expanded=True):
                    st.markdown(f"""
                    <div class="file-info">
                        <strong>File:</strong> {file_info['name']}<br>
                        <strong>Status:</strong> {file_info['status'].capitalize()}<br>
                        <strong>GCS URI:</strong> {file_info['gcs_uri']}<br>
                        <strong>Links:</strong><br>
                        - <a href="{file_info['console_url']}" target="_blank">View in Console</a> (requires GCP authentication)
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if 'metadata' in file_info and file_info['metadata']:
                        st.markdown("#### Metadata")
                        st.json(file_info['metadata'])
        
        st.markdown("---")

# Query Section
def query_section():
    st.markdown('<p class="upload-header">Query Documents</p>', unsafe_allow_html=True)
    
    st.info("Query functionality will be implemented in the next phase.")
    
    # Placeholder for future query interface
    st.text_input("Ask a question about your documents")
    st.button("Submit Query", disabled=True)

def process_upload(uploaded_file):
    with st.spinner('Uploading and processing file...'):
        # Create progress bar
        progress_bar = st.progress(0)
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
            
            # Update progress
            progress_bar.progress(25)
            time.sleep(0.5)  # Simulate processing time

        try:
            # Upload to GCS
            gcs_manager = GCSManager()
            progress_bar.progress(50)
            time.sleep(0.5)  # Simulate processing time
            
            result = gcs_manager.upload_file(tmp_path, uploaded_file.name)
            progress_bar.progress(75)
            time.sleep(0.5)  # Simulate processing time
            
            if result['status'] == 'success':
                # Get GCS links
                gcs_uri = result.get('gcs_uri', '')
                links = get_gcs_links(gcs_uri)
                
                # Add to session state
                file_info = {
                    'name': uploaded_file.name,
                    'status': 'success',
                    'gcs_uri': gcs_uri,
                    'console_url': links['console_url'],
                    'metadata': result.get('metadata', {})
                }
                st.session_state.uploaded_files.append(file_info)
                
                # Show success message
                st.success("File uploaded successfully!")
                st.markdown(f"""
                <div class="file-info">
                    <strong>File:</strong> {uploaded_file.name}<br>
                    <strong>Status:</strong> Success<br>
                    <strong>GCS URI:</strong> {gcs_uri}<br>
                    <strong>Console Link:</strong> <a href="{links['console_url']}" target="_blank">View in Console</a> (requires GCP authentication)
                </div>
                """, unsafe_allow_html=True)
                
                # Automatically switch to documents tab
                st.session_state.active_tab = "documents"
                
            elif result['status'] == 'duplicate':
                # Get metadata for duplicate
                metadata = gcs_manager.get_file_metadata(result['file_hash'])
                gcs_uri = metadata.get('gcs_uri', '')
                links = get_gcs_links(gcs_uri)
                
                # Add to session state if not already there
                file_exists = False
                for file in st.session_state.uploaded_files:
                    if file.get('gcs_uri') == gcs_uri:
                        file_exists = True
                        break
                
                if not file_exists:
                    file_info = {
                        'name': uploaded_file.name,
                        'status': 'duplicate',
                        'gcs_uri': gcs_uri,
                        'console_url': links['console_url'],
                        'metadata': metadata
                    }
                    st.session_state.uploaded_files.append(file_info)
                
                # Show warning
                st.warning("This file has already been uploaded.")
                st.markdown(f"""
                <div class="file-info">
                    <strong>File:</strong> {uploaded_file.name}<br>
                    <strong>Status:</strong> Duplicate<br>
                    <strong>GCS URI:</strong> {gcs_uri}<br>
                    <strong>Console Link:</strong> <a href="{links['console_url']}" target="_blank">View in Console</a> (requires GCP authentication)
                </div>
                """, unsafe_allow_html=True)
                
                # Display metadata
                with st.expander("View Metadata"):
                    st.json(metadata)
                    
                # Automatically switch to documents tab
                st.session_state.active_tab = "documents"
                
            else:
                st.error(f"Upload failed: {result.get('message', 'Unknown error')}")
                
            progress_bar.progress(100)
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)

def get_gcs_links(gcs_uri):
    """Generate GCS links for console access only (no public links)"""
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT', 'finsight-ai-nguyen')
    
    # Remove gs:// prefix and split into bucket and path
    bucket_path = gcs_uri.replace('gs://', '')
    bucket_name = bucket_path.split('/')[0]
    file_path = '/'.join(bucket_path.split('/')[1:])
    
    # Format GCS console URL (requires GCP authentication)
    console_url = f"https://console.cloud.google.com/storage/browser/{bucket_name}/{file_path}?project={project_id}"
    
    return {
        'console_url': console_url,
        'bucket_url': f"https://console.cloud.google.com/storage/browser/{bucket_name}?project={project_id}"
    }

# Main app
def main():
    # Setup GCP credentials
    setup_gcp_credentials()
    
    if st.session_state.active_tab == "upload":
        upload_section()
    elif st.session_state.active_tab == "documents":
        documents_section()
    elif st.session_state.active_tab == "query":
        query_section()

if __name__ == "__main__":
    main()
