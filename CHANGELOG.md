# Changelog

All notable changes to the FinSight AI project will be documented in this file.

## [0.4.3] - 2024-03-02

### 🧰 Feature Milestone: Code Organization and Cleanup

#### Added
- Unified setup script (`scripts/setup.py`) for easier project setup
- Common utility functions in `scripts/utils/common.py`
- Proper package structure with `__init__.py` files

#### Changed
- Reorganized scripts into logical directories (setup, utils)
- Moved test scripts from scripts/ to tests/ directory
- Updated README with new project structure
- Improved code organization and maintainability

#### Removed
- Redundant code duplicated across multiple scripts

## [0.4.2] - 2024-03-01

### 🧠 Feature Milestone: Advanced Entity Recognition

#### Added
- Enhanced name recognition for queries about specific individuals
- Fallback mechanism for entity queries with regex pattern matching
- Increased context retrieval from 5 to 8 documents
- Improved QA prompt for better handling of specific entity queries

#### Changed
- Updated query processing logic to handle entity-specific queries
- Enhanced response generation for queries about people mentioned in documents
- Improved error handling for queries with no results

## [0.4.1] - 2024-02-28

### 🎨 Feature Milestone: Enhanced UI/UX

#### Added
- Modern, responsive UI with clean design
- Tabbed interface with Chat and About sections
- Visual status indicators for operations
- Compact document and source display
- Example questions in grid layout

#### Changed
- Moved chat input to bottom of screen for better user experience
- Removed bulky information boxes for cleaner interface
- Reduced font sizes for better information density
- Improved document type recognition with specific icons
- Enhanced source attribution with expandable sections

#### Improved
- Overall application responsiveness
- Information density and readability
- Visual hierarchy and organization
- User onboarding with better welcome screen

## [0.4.0] - 2024-02-26

### 🤖 Feature Milestone: RAG Pipeline Integration

#### Added
- Retrieval Augmented Generation (RAG) pipeline
- Interactive chat interface with source attribution
- Document selection functionality
- Conversation memory for contextual responses
- Example questions for user guidance

#### Changed
- Updated project structure for better organization
- Enhanced documentation with RAG pipeline details
- Improved error handling and user feedback

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