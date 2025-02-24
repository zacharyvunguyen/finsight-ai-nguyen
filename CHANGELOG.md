# Changelog

All notable changes to the FinSight AI project will be documented in this file.

## [0.3.0] - 2024-02-24

### 🔒 Security Milestone: Secure Document Handling

#### Added
- Secure document handling with authenticated access only
- Enhanced document listing with better UI
- Debug information in sidebar for troubleshooting
- Automatic loading of existing documents from GCS
- Improved document details view
- Automatic navigation to Documents tab after upload

#### Changed
- Removed all public links to documents
- Modified GCS links to require authentication
- Updated documentation with security best practices
- Improved UI with cleaner document listing
- Enhanced error handling and user feedback

#### Security
- Documents now require GCP authentication to access
- No public URLs are generated or displayed
- Added clear authentication requirements notices
- Improved session state management

## [0.2.0] - 2024-02-23

### 🚀 Feature Milestone: PDF Upload Functionality

#### Added
- PDF upload functionality with GCS integration
- Duplicate detection using file hashing
- Metadata storage in Firestore
- Basic Streamlit interface
- GCS file management utilities

#### Changed
- Improved project structure
- Enhanced error handling
- Added logging for better debugging

## [0.1.0] - 2024-02-22

### 🎉 Initial Release

#### Added
- Initial project setup
- Basic directory structure
- Configuration files
- Documentation
- GCP project setup 