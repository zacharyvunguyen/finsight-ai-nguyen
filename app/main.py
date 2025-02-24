import streamlit as st
from utils.gcs import GCSManager
import tempfile
import os

def upload_section():
    st.header("Upload PDF Documents")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
    
    if uploaded_file:
        with st.spinner('Uploading and processing file...'):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            try:
                # Upload to GCS
                gcs_manager = GCSManager()
                result = gcs_manager.upload_file(tmp_path, uploaded_file.name)

                if result['status'] == 'success':
                    st.success(f"File uploaded successfully! GCS URI: {result['gcs_uri']}")
                elif result['status'] == 'duplicate':
                    st.warning("This file has already been uploaded.")
                    # Show existing file metadata
                    metadata = gcs_manager.get_file_metadata(result['file_hash'])
                    st.json(metadata)
                else:
                    st.error(f"Upload failed: {result['message']}")

            finally:
                # Clean up temporary file
                os.unlink(tmp_path)

# Add this to your main Streamlit app
if __name__ == "__main__":
    upload_section()
