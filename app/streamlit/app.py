#!/usr/bin/env python3
"""
Streamlit app for querying documents from Pinecone using RAG.
A simplified version that focuses on the chat interface.
"""

import os
import streamlit as st
from test_llamaparse_pdf import (
    load_environment,
    setup_llm_and_embeddings,
    create_query_engine
)
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document as LangchainDocument
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# Set page config - MUST be the first Streamlit command
st.set_page_config(
    page_title="FinSight AI - Chat",
    page_icon="��",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #333;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .compact-text {
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    .source-section {
        background-color: #f9f9f9;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }
    .example-button {
        width: 100%;
        text-align: left !important;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        white-space: pre-wrap;
        font-size: 1rem;
    }
    .status-success {
        color: #4CAF50;
        font-weight: 500;
    }
    .status-warning {
        color: #FF9800;
        font-weight: 500;
    }
    .status-error {
        color: #F44336;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'query_engine' not in st.session_state:
    st.session_state.query_engine = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )
if 'env' not in st.session_state:
    st.session_state.env = load_environment()
if 'active_documents' not in st.session_state:
    st.session_state.active_documents = set()
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Chat"

# Add this function to get the LLM
def get_llm(temperature=0.7):
    """Get a ChatOpenAI LLM instance."""
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=temperature,
        api_key=os.environ.get("OPENAI_API_KEY")
    )

def create_qa_chain(query_engine, temperature=0.7):
    """Create a Langchain QA chain with custom prompt and memory."""
    template = """You are a helpful financial analyst assistant. Use the following pieces of context and chat history to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Always include specific numbers, dates, and facts from the context when available.
    If the answer is found in a specific section or page, mention that.
    When the information comes from different documents, clearly indicate which document it's from.
    Use the chat history to provide more contextual and relevant answers.
    When comparing with previous answers or years, make explicit references to the data points you're comparing.
    
    For questions about specific people, companies, or entities:
    - If the person or entity is mentioned in the context, provide all available details about them
    - For board members, executives, or signatories, include their title, role, and any other information provided
    - If dates are associated with their position or actions, include those dates
    
    Chat History: {chat_history}
    
    Context: {context}

    Question: {question}

    Helpful Answer:"""

    QA_PROMPT = PromptTemplate(
        template=template,
        input_variables=["context", "chat_history", "question"]
    )

    # Get LLM directly instead of from query_engine
    llm = get_llm(temperature=temperature)
    
    # Get the Pinecone index name from environment
    index_name = st.session_state.env.get('ACTIVE_INDEX', 'dev-financial-reports')
    
    # Initialize Pinecone with new API
    pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY")
    )
    
    # Get the Pinecone index
    index = pc.Index(index_name)
    
    # Create embeddings model
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.environ.get("OPENAI_API_KEY")
    )
    
    # Create a filter for the selected documents if any are specified
    filter_dict = None
    if hasattr(query_engine, 'selected_docs') and query_engine.selected_docs:
        filter_dict = {"document_id": {"$in": query_engine.selected_docs}}
    
    # Create the vector store
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        text_key="text"
    )
    
    # Get the retriever from the vector store with filter
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 8,  # Increased from 5 to 8 to get more context
            "filter": filter_dict
        } if filter_dict else {"k": 8}
    )
    
    # Create the chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=st.session_state.memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={
            "prompt": QA_PROMPT,
        },
        verbose=True
    )
    
    return chain

def display_chat_history():
    """Display the chat history in a conversational format."""
    for query, answer in st.session_state.chat_history:
        # Display user message
        with st.chat_message("user"):
            st.markdown(query)
        
        # Display assistant response
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(answer)

def display_about_tab():
    """Display information about the app, technologies, and advantages."""
    st.markdown('<div class="main-header">About FinSight AI</div>', unsafe_allow_html=True)
    
    # What is FinSight AI section
    st.markdown('<div class="sub-header">What is FinSight AI?</div>', unsafe_allow_html=True)
    
    with st.container():
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            FinSight AI is an advanced financial document analysis tool that leverages the power of Retrieval-Augmented Generation (RAG) 
            to provide intelligent insights from financial reports, SEC filings, and other financial documents. The application allows 
            users to query their financial documents in natural language and receive accurate, contextually relevant answers.
            """)
        
        with col2:
            st.image("https://img.icons8.com/fluency/240/financial-analytics.png", width=150)
    
    # Technologies Used section
    st.markdown('<div class="sub-header">Technologies Used</div>', unsafe_allow_html=True)
    
    tech_col1, tech_col2 = st.columns(2)
    
    with tech_col1:
        st.markdown("### Core Technologies")
        
        tech_data = {
            "LangChain": "Orchestrates the RAG pipeline and manages conversation flow",
            "OpenAI": "Powers the language model (GPT-3.5 Turbo) and embedding model",
            "Pinecone": "Vector database for storing and retrieving document embeddings",
            "Streamlit": "Web application framework for the user interface"
        }
        
        for tech, desc in tech_data.items():
            st.markdown(f"**{tech}**: {desc}")
    
    with tech_col2:
        st.markdown("### Key Components")
        
        components = [
            "**Document Processing**: Financial documents are parsed, chunked, and embedded",
            "**Vector Storage**: Document chunks and their embeddings are stored in Pinecone",
            "**Retrieval System**: Relevant document chunks are retrieved based on semantic similarity",
            "**Language Model**: OpenAI's GPT model generates responses based on retrieved context",
            "**Conversation Memory**: Chat history is maintained to provide contextual responses"
        ]
        
        for component in components:
            st.markdown(f"- {component}")
    
    # Advantages section with expanders
    st.markdown('<div class="sub-header">Advantages</div>', unsafe_allow_html=True)
    
    advantages = {
        "Enhanced Accuracy": [
            "RAG combines the knowledge from your documents with the capabilities of large language models",
            "Responses are grounded in your actual financial data, reducing hallucinations"
        ],
        "Specialized Financial Knowledge": [
            "Custom prompt engineering focused on financial analysis",
            "Ability to extract and interpret complex financial metrics and trends"
        ],
        "Document-Specific Insights": [
            "Filter queries to specific documents of interest",
            "Compare information across different reports or time periods"
        ],
        "Efficient Information Retrieval": [
            "Instantly access information from large financial documents",
            "Natural language interface eliminates the need for complex search queries"
        ],
        "Transparency": [
            "Source attribution shows exactly where information comes from",
            "Access to the original context for verification"
        ]
    }
    
    adv_cols = st.columns(3)
    
    for i, (title, points) in enumerate(advantages.items()):
        with adv_cols[i % 3]:
            with st.expander(f"**{i+1}. {title}**", expanded=True):
                for point in points:
                    st.markdown(f"- {point}")
    
    # How It Works section
    st.markdown('<div class="sub-header">How It Works</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="source-section">', unsafe_allow_html=True)
    
    steps = [
        "Documents are processed and stored in Pinecone with their embeddings",
        "When you ask a question, the system finds the most relevant document chunks",
        "These chunks are sent to the language model along with your question",
        "The model generates a response based on the retrieved information",
        "The system displays the answer along with the source documents"
    ]
    
    for i, step in enumerate(steps):
        st.markdown(f"**{i+1}.** {step}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("""
    This approach ensures that responses are accurate, relevant, and directly tied to your financial documents.
    """)

# Main app
tabs = st.tabs(["💬 Chat", "ℹ️ About"])

# Setup sidebar
with st.sidebar:
    st.markdown('<div class="main-header">FinSight AI</div>', unsafe_allow_html=True)
    
    # Display index information
    st.markdown('<div class="sub-header">Index Information</div>', unsafe_allow_html=True)
    if st.session_state.env:
        index_name = st.session_state.env['ACTIVE_INDEX']
        
        # Simplified index information display
        st.markdown(f"📊 **Active Index:** {index_name}")
        st.markdown(f"🌐 **Environment:** {st.session_state.env.get('PINECONE_ENVIRONMENT', 'Production')}")
        st.markdown(f"📏 **Dimension:** 1536 (OpenAI Embeddings)")
        st.markdown(f"📐 **Metric:** Cosine Similarity")
    
    # Document selection
    st.markdown('<div class="sub-header">Document Selection</div>', unsafe_allow_html=True)
    
    # Get document IDs from Pinecone
    if st.button("🔄 Refresh Available Documents", use_container_width=True):
        with st.spinner("Loading documents from Pinecone..."):
            try:
                # Setup LLM and embeddings if not already done
                setup_llm_and_embeddings()
                
                # Get the Pinecone index name from environment
                index_name = st.session_state.env.get('ACTIVE_INDEX', 'dev-financial-reports')
                
                # Initialize Pinecone with new API
                pc = Pinecone(
                    api_key=os.environ.get("PINECONE_API_KEY")
                )
                
                # Get the Pinecone index
                index = pc.Index(index_name)
                
                # Query for document IDs
                try:
                    # Use stats to get metadata
                    stats = index.describe_index_stats()
                    
                    # Use a query to get unique document IDs
                    query_response = index.query(
                        vector=[0] * 1536,  # Dummy vector
                        top_k=100,
                        include_metadata=True
                    )
                    
                    # Extract unique document IDs from metadata
                    document_ids = set()
                    for match in query_response.matches:
                        if 'document_id' in match.metadata:
                            document_ids.add(match.metadata['document_id'])
                    
                    if document_ids:
                        st.session_state.available_documents = list(document_ids)
                        st.markdown(f'<span class="status-success">✓ Found {len(document_ids)} documents in Pinecone</span>', unsafe_allow_html=True)
                    else:
                        # For testing, use a sample document ID
                        st.session_state.available_documents = ["10-K-2021-(As-Filed)_doc_11"]
                        st.markdown('<span class="compact-text">Using sample document for testing</span>', unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f'<span class="status-error">✗ Error querying Pinecone: {str(e)}</span>', unsafe_allow_html=True)
                    # For testing, use a sample document ID
                    st.session_state.available_documents = ["10-K-2021-(As-Filed)_doc_11"]
                    st.markdown('<span class="compact-text">Using sample document for testing</span>', unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f'<span class="status-error">✗ Error loading documents: {str(e)}</span>', unsafe_allow_html=True)
                st.session_state.available_documents = ["10-K-2021-(As-Filed)_doc_11"]
                st.markdown('<span class="compact-text">Using sample document for testing</span>', unsafe_allow_html=True)
    
    # Select documents to query
    if 'available_documents' in st.session_state and st.session_state.available_documents:
        selected_docs = st.multiselect(
            "Select documents to query:",
            st.session_state.available_documents,
            default=list(st.session_state.active_documents) if st.session_state.active_documents else None
        )
        
        if st.button("🚀 Update Query Engine", use_container_width=True):
            if selected_docs:
                with st.spinner("Setting up query engine..."):
                    try:
                        # Setup LLM and embeddings
                        setup_llm_and_embeddings()
                        
                        # Store the selected documents
                        class QueryEngineWrapper:
                            def __init__(self, selected_docs):
                                self.selected_docs = selected_docs
                        
                        # Create a simple wrapper to hold the selected documents
                        query_engine = QueryEngineWrapper(selected_docs)
                        st.session_state.query_engine = query_engine
                        
                        # Create QA chain with the selected documents
                        qa_chain = create_qa_chain(query_engine)
                        st.session_state.qa_chain = qa_chain
                        st.session_state.active_documents = set(selected_docs)
                        st.markdown(f'<span class="status-success">✓ Ready to query {len(selected_docs)} documents</span>', unsafe_allow_html=True)
                    except Exception as e:
                        st.markdown(f'<span class="status-error">✗ Error: {str(e)}</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-warning">⚠ Please select at least one document</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="compact-text">Click \'Refresh Available Documents\' to load documents from Pinecone</span>', unsafe_allow_html=True)
    
    # Reset chat button
    st.divider()
    if st.button("🗑️ Reset Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.memory.clear()
        st.markdown('<span class="status-success">✓ Chat history cleared!</span>', unsafe_allow_html=True)

# Chat tab
with tabs[0]:
    st.markdown('<div class="main-header">FinSight AI - Document Chat</div>', unsafe_allow_html=True)
    
    # Main chat interface
    if st.session_state.query_engine and st.session_state.qa_chain:
        # Display active documents with more details
        st.markdown('<div class="sub-header">Active Documents</div>', unsafe_allow_html=True)
        
        # Get index name for reference
        index_name = st.session_state.env.get('ACTIVE_INDEX', 'dev-financial-reports')
        
        # Display document information in a more compact format
        doc_cols = st.columns(3)
        
        for i, doc in enumerate(st.session_state.active_documents):
            # Extract document type from ID if possible
            doc_type = "Financial Report"
            doc_icon = "📄"
            if "10-K" in doc:
                doc_type = "Annual Report (10-K)"
                doc_icon = "📊"
            elif "10-Q" in doc:
                doc_type = "Quarterly Report (10-Q)"
                doc_icon = "📈"
            elif "8-K" in doc:
                doc_type = "Current Report (8-K)"
                doc_icon = "📰"
            
            with doc_cols[i % 3]:
                st.markdown(f"{doc_icon} **{doc_type}**")
                st.markdown(f"ID: `{doc}`")
        
        st.divider()
        
        # Display chat history
        display_chat_history()
        
        # Example questions
        st.markdown('<div class="sub-header">Example Questions</div>', unsafe_allow_html=True)
        
        example_queries = [
            "What was the company's revenue for the last fiscal year?",
            "What are the main risk factors mentioned in the financial report?",
            "Summarize the business overview and main operations.",
            "What are the key financial metrics and their trends over time?",
            "Explain the company's strategy and future outlook."
        ]
        
        example_cols = st.columns(3)
        for i, example_query in enumerate(example_queries):
            with example_cols[i % 3]:
                if st.button(f"🔍 {example_query}", key=f"example_{i}", use_container_width=True):
                    # Set the query as a session state variable and rerun
                    st.session_state.example_selected = example_query
                    st.rerun()
        
        # Check if an example was selected and process it
        if 'example_selected' in st.session_state and st.session_state.example_selected:
            query = st.session_state.example_selected
            
            # Clear the selection for next time
            selected_query = st.session_state.example_selected
            st.session_state.example_selected = None
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(selected_query)
            
            # Generate and display assistant response
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Thinking..."):
                    try:
                        message_placeholder = st.empty()
                        st_callback = StreamlitCallbackHandler(st.container())
                        result = st.session_state.qa_chain(
                            {"question": selected_query},
                            callbacks=[st_callback]
                        )
                        message_placeholder.markdown(result['answer'])
                        st.session_state.chat_history.append((selected_query, result['answer']))
                        
                        # Display sources in a more compact format
                        if 'source_documents' in result:
                            st.markdown("### 📚 Sources")
                            
                            for i, doc in enumerate(result['source_documents'][:3]):
                                with st.expander(f"Source {i+1} from `{doc.metadata.get('document_id', 'Unknown')}`", expanded=i==0):
                                    st.markdown(doc.page_content)
                                    st.markdown("**Metadata:**")
                                    st.json(doc.metadata)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        # Add a spacer before the chat input
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Chat input - moved to bottom
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Display user message immediately
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate and display assistant response
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Thinking..."):
                    try:
                        message_placeholder = st.empty()
                        st_callback = StreamlitCallbackHandler(st.container())
                        
                        # Process the query
                        result = st.session_state.qa_chain(
                            {"question": prompt},
                            callbacks=[st_callback]
                        )
                        
                        answer = result['answer']
                        
                        # Check if the answer indicates no information was found for a person query
                        if ("I don't have information" in answer or "no information" in answer) and any(name in prompt.upper() for name in ["WHO IS", "ABOUT", "TELL ME ABOUT"]):
                            # This might be a person query that didn't get good results
                            # Extract potential names from the query
                            import re
                            potential_names = re.findall(r'[A-Z][a-zA-Z]+ [A-Z]\. [A-Z][a-zA-Z]+|[A-Z][a-zA-Z]+ [A-Z][a-zA-Z]+', prompt)
                            
                            if potential_names:
                                # Try a more direct search for the name
                                name_to_search = potential_names[0]
                                
                                # Create a more specific query focused on finding the person
                                specific_query = f"Find information about {name_to_search} including their role, title, or position"
                                
                                # Run a second query with this more specific prompt
                                second_result = st.session_state.qa_chain(
                                    {"question": specific_query},
                                    callbacks=[st_callback]
                                )
                                
                                # If the second query found information, use that instead
                                if "I don't have information" not in second_result['answer'] and "no information" not in second_result['answer']:
                                    answer = second_result['answer']
                                    result['source_documents'] = second_result['source_documents']
                        
                        message_placeholder.markdown(answer)
                        st.session_state.chat_history.append((prompt, answer))
                        
                        # Display sources in a more compact format
                        if 'source_documents' in result:
                            st.markdown("### 📚 Sources")
                            
                            for i, doc in enumerate(result['source_documents'][:3]):
                                with st.expander(f"Source {i+1} from `{doc.metadata.get('document_id', 'Unknown')}`", expanded=i==0):
                                    st.markdown(doc.page_content)
                                    st.markdown("**Metadata:**")
                                    st.json(doc.metadata)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    else:
        # Welcome screen
        st.markdown('<div class="main-header">👋 Welcome to FinSight AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Your Financial Document Assistant</div>', unsafe_allow_html=True)
        
        welcome_cols = st.columns([2, 1])
        
        with welcome_cols[0]:
            st.markdown("""
            **Getting Started:**
            1. Click **Refresh Available Documents** in the sidebar to load documents from Pinecone
            2. Select one or more documents to query
            3. Click **Update Query Engine** to prepare the system
            4. Start asking questions about your documents!
            """)
            
            st.markdown("### What you can do:")
            
            features = [
                "💬 Chat with your financial documents",
                "🔍 Get insights from specific documents",
                "📊 Extract key financial metrics and trends",
                "📈 Compare information across reports",
                "❓ Ask questions in natural language"
            ]
            
            for feature in features:
                st.markdown(f"- {feature}")
        
        with welcome_cols[1]:
            st.image("https://img.icons8.com/color/240/financial-growth.png", width=150)

# About tab
with tabs[1]:
    display_about_tab() 