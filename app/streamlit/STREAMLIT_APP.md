# FinSight AI - Streamlit Application

## Overview

The FinSight AI Streamlit application provides an intuitive interface for querying financial documents using a Retrieval Augmented Generation (RAG) pipeline. This document outlines the application's features, design, and functionality.

## Features

### Core Functionality
- **Document Selection**: Select specific financial documents to query from Pinecone
- **Natural Language Queries**: Ask questions about financial documents in plain English
- **Source Attribution**: View the exact sources used to generate responses
- **Conversation Memory**: Maintain context across multiple questions
- **Example Questions**: Pre-defined questions to help users get started

### UI/UX Enhancements
- **Modern Interface**: Clean, professional design with intuitive navigation
- **Tabbed Layout**: Separate Chat and About sections for better organization
- **Responsive Design**: Adapts to different screen sizes
- **Visual Feedback**: Status indicators for operations (success, warning, error)
- **Compact Information Display**: Efficient use of screen space

### Advanced Features
- **Name Recognition**: Enhanced handling of queries about specific individuals
- **Fallback Mechanism**: Secondary search for names if initial query yields no results
- **Document Type Recognition**: Automatic identification of document types (10-K, 10-Q, 8-K)
- **Expandable Sources**: Collapsible sections for source documents to save space

## Application Structure

### Main Components
1. **Sidebar**
   - Index information display
   - Document selection interface
   - Refresh documents button
   - Update query engine button
   - Reset chat button

2. **Chat Tab**
   - Active documents display
   - Chat history
   - Example questions
   - Chat input (positioned at bottom)
   - Source attribution section

3. **About Tab**
   - Application overview
   - Technologies used
   - Key components
   - Advantages of RAG approach
   - How it works section

## Technical Implementation

### Session State Management
The application uses Streamlit's session state to maintain:
- Query engine configuration
- QA chain
- Chat history
- Conversation memory
- Environment variables
- Active documents
- Active tab selection

### RAG Pipeline Integration
1. **Document Retrieval**: Fetches document chunks from Pinecone based on semantic similarity
2. **Context Assembly**: Combines retrieved chunks with conversation history
3. **Response Generation**: Uses OpenAI's language model to generate responses
4. **Source Attribution**: Displays source documents with metadata

### Enhanced Query Processing
- **Custom Prompt Engineering**: Financial-specific prompt template
- **Increased Context Window**: Retrieves 8 document chunks (increased from 5)
- **Name Recognition**: Regex pattern to identify names in queries
- **Fallback Mechanism**: Secondary query for specific entities

## Design Elements

### Visual Components
- **Status Indicators**: Color-coded text for success (green), warning (orange), error (red)
- **Document Cards**: Compact display of document information with type-specific icons
- **Source Sections**: Expandable sections for source documents
- **Example Question Grid**: Organized grid layout for example questions

### CSS Styling
- **Typography**: Consistent font sizing and styling
- **Color Scheme**: Professional blue-based color palette
- **Spacing**: Appropriate whitespace for readability
- **Component Styling**: Custom styling for tabs, buttons, and information displays

## Usage Guide

### Getting Started
1. Click "Refresh Available Documents" in the sidebar
2. Select one or more documents from the list
3. Click "Update Query Engine" to prepare the system
4. Start asking questions in the chat input or use example questions

### Example Queries
- "What was the company's revenue for the last fiscal year?"
- "What are the main risk factors mentioned in the financial report?"
- "Summarize the business overview and main operations."
- "What are the key financial metrics and their trends over time?"
- "Explain the company's strategy and future outlook."
- "Who is [Person Name]?" (e.g., "Who is Arthur D. Levinson?")

### Interpreting Results
- **Main Response**: The answer to your question based on document content
- **Source Documents**: The specific document chunks used to generate the response
- **Metadata**: Information about each source document (document ID, page, etc.)

## Recent Improvements

### UI/UX Enhancements
- Moved chat input to bottom of screen for better user experience
- Removed bulky blue information boxes for cleaner interface
- Implemented more compact document and source display
- Added visual status indicators with minimal space usage
- Reduced font sizes for better information density

### Functional Improvements
- Enhanced name recognition for queries about specific individuals
- Implemented fallback mechanism for entity queries
- Increased context retrieval from 5 to 8 documents
- Improved QA prompt for better handling of specific entity queries

## Running the Application

```bash
# Navigate to the tests directory
cd tests

# Run the Streamlit application
streamlit run streamlit_app.py
```

## Dependencies
- Streamlit
- LangChain
- OpenAI
- Pinecone
- Python 3.9+

## Notes
- This application is designed for testing and demonstration purposes
- For production use, additional security measures should be implemented
- The application requires valid API keys for OpenAI and Pinecone 