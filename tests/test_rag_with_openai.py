#!/usr/bin/env python3
"""
Test script for implementing a RAG pipeline using direct Pinecone queries and OpenAI.
This script:
1. Connects to Pinecone
2. Generates embeddings for a query using OpenAI
3. Directly queries Pinecone with the embeddings
4. Extracts relevant content from the results
5. Uses OpenAI to generate a response based on the retrieved content
"""

import os
import sys
import json
import argparse
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Add the parent directory to the path so we can import from scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import common utility functions
from scripts.utils.common import print_header, print_result

# Import functions from the setup_rag_pipeline script with updated paths
from scripts.setup.setup_rag_pipeline import (
    load_environment,
    verify_pinecone_setup,
    verify_openai_api
)

def extract_text_from_match(match):
    """Extract text content from a Pinecone match."""
    metadata = match['metadata']
    text = ""
    page_number = None
    file_name = None
    
    # Print raw metadata for debugging
    print("\n=== RAW METADATA ===")
    import pprint
    pprint.pprint(metadata)
    print("===================\n")
    
    # Extract page number and file name if available
    if 'page_number' in metadata:
        page_number = metadata['page_number']
    elif 'pg' in metadata:
        page_number = metadata['pg']
        
    if 'file_name' in metadata:
        file_name = metadata['file_name']
    elif 'fn' in metadata:
        file_name = metadata['fn']
    
    # Try to extract the text content from node_content
    if '_node_content' in metadata:
        try:
            node_content = json.loads(metadata['_node_content'])
            print("\n=== NODE CONTENT ===")
            pprint.pprint(node_content)
            print("===================\n")
            
            # First try to get text from obj.__data__.text (for tables and structured data)
            if 'obj' in node_content and '__data__' in node_content['obj'] and 'text' in node_content['obj']['__data__']:
                text = node_content['obj']['__data__']['text']
                print(f"Found text in obj.__data__.text: {text[:100]}...")
            # Then try to get text directly
            elif 'text' in node_content:
                text = node_content['text']
                print(f"Found text directly in node_content: {text[:100]}...")
                
            # Try to extract table data if available
            if 'obj' in node_content and '__data__' in node_content['obj'] and 'metadata' in node_content['obj']['__data__']:
                obj_metadata = node_content['obj']['__data__']['metadata']
                print("\n=== OBJ METADATA ===")
                pprint.pprint(obj_metadata)
                print("===================\n")
                
                if 'table_df' in obj_metadata:
                    text += f"Table data: {obj_metadata['table_df']}\n"
                    print(f"Found table_df: {obj_metadata['table_df'][:100]}...")
                if 'table_summary' in obj_metadata:
                    text += f"Table summary: {obj_metadata['table_summary']}\n"
                    print(f"Found table_summary: {obj_metadata['table_summary'][:100]}...")
        except json.JSONDecodeError:
            print("Error: Could not decode _node_content as JSON")
            print(f"Raw _node_content: {metadata['_node_content'][:200]}...")
        except Exception as e:
            print(f"Error extracting text: {e}")
    
    # If we couldn't extract text from node_content, use content_preview
    if not text and 'content_preview' in metadata:
        text = metadata['content_preview']
        print(f"Using content_preview: {text[:100]}...")
    
    # Add source information
    source_info = ""
    if page_number is not None:
        source_info += f" [Page: {page_number}]"
    if file_name is not None:
        source_info += f" [File: {file_name}]"
    
    # Add any additional metadata that might be useful
    if 'GROSS PATIENT SERVICE REVENUE' in text or 'GPSR' in text:
        print(f"Found relevant financial data about GROSS PATIENT SERVICE REVENUE!{source_info}")
        
    return text, source_info, {"page_number": page_number, "file_name": file_name}

def query_pinecone_directly(query, openai_client, pinecone_index, namespace="doc_b7712a3a4e"):
    """Query Pinecone directly with a user query."""
    print_header(f"Processing Query: {query}")
    
    # Generate embeddings for query
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=query,
        encoding_format="float"
    )
    query_embedding = response.data[0].embedding
    
    # Query Pinecone
    print("Querying Pinecone...")
    results = pinecone_index.query(
        vector=query_embedding,
        top_k=5,  # Limit to fewer results for detailed inspection
        namespace=namespace,
        include_metadata=True
    )
    
    # Process results
    print(f"Found {len(results['matches'])} matches")
    
    # Store the most relevant context texts with their relevance scores
    relevant_contexts = []
    
    # Extract text from matches
    for i, match in enumerate(results['matches']):
        print(f"\n\n==== MATCH {i+1} (Score: {match['score']}) ====")
        text, source_info, source_metadata = extract_text_from_match(match)
        if text:
            # Only add if it contains relevant financial information
            if any(term in text.upper() for term in ["REVENUE", "PATIENT", "FINANCIAL", "GROSS", "GPSR", "INPATIENT", "OUTPATIENT"]):
                # Calculate relevance score based on match score and presence of key terms
                relevance_score = match['score']
                
                # Add to relevant contexts
                context_entry = {
                    "text": f"Document {i+1} (Score: {match['score']}){source_info}:\n{text}",
                    "score": relevance_score,
                    "metadata": source_metadata
                }
                relevant_contexts.append(context_entry)
                print(f"Match {i+1} (Score: {match['score']}){source_info}: {text[:200]}...")
    
    # Sort contexts by relevance score and take the top 10 most relevant
    relevant_contexts.sort(key=lambda x: x["score"], reverse=True)
    top_contexts = relevant_contexts[:5]  # Limit to top 5 most relevant contexts
    
    return top_contexts

def setup_langchain_retrieval(query, openai_api_key, pinecone_api_key, index_name, namespace="doc_b7712a3a4e"):
    """Set up LangChain retrieval from Pinecone."""
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    
    # Initialize Pinecone vector store
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        pinecone_api_key=pinecone_api_key,
        namespace=namespace
    )
    
    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )
    
    # Define prompt template
    template = """
    You are a financial data extraction tool. Your ONLY task is to extract EXACT financial numbers from the provided context.
    
    DO NOT add any explanations, interpretations, or additional text. ONLY return the exact numbers and account codes as they appear in the document.
    
    For example, if the query asks about "401000 - Inpatient Medicare Trad Intensive Care Revenue", you should ONLY return something like:
    "401000 - Inpatient Medicare Trad Intensive Care Revenue: $11,807,381 $16,322,926 ($4,515,544) (27.7%)"
    
    DO NOT make up any numbers. If the exact data is not found, simply state "No exact match found for [query term]".
    
    DO NOT say "I don't have enough information" or provide any other explanations.
    
    ONLY return the raw financial data exactly as it appears in the document.
    
    If page numbers are available, include them in a minimal format: "[Page X]".
    
    Context:
    {context}
    
    Query: {question}
    
    Extract ONLY the exact financial numbers and account codes related to this query. Return the raw data exactly as it appears in the document with no additional text or explanations.
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    # Create LLM
    llm = ChatOpenAI(
        temperature=0.1,
        model="gpt-3.5-turbo-0125",
        openai_api_key=openai_api_key
    )
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain

def main():
    """Main function to test RAG with direct Pinecone queries and OpenAI."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Query financial documents using RAG')
    parser.add_argument('--query', type=str, required=True, help='The query to search for in the documents')
    args = parser.parse_args()
    
    user_query = args.query
    
    print_header("Testing RAG with Direct Pinecone Queries and OpenAI")
    print(f"User Query: {user_query}")
    
    # Load environment variables
    load_dotenv()
    
    # Verify OpenAI API
    if not verify_openai_api():
        print_result(False, "OpenAI API verification failed")
        return
    
    # Verify Pinecone setup
    index_name = os.getenv("PINECONE_INDEX_NAME")
    if not verify_pinecone_setup(index_name):
        print_result(False, "Pinecone setup verification failed")
        return
    
    try:
        # Initialize OpenAI client
        print_header("Initializing OpenAI Client")
        openai_client = OpenAI()
        print_result(True, "OpenAI client initialized")
        
        # Initialize Pinecone client
        print_header("Initializing Pinecone Client")
        pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        pinecone_index = pinecone_client.Index(index_name)
        print_result(True, "Pinecone client initialized")
        
        # Get index stats
        stats = pinecone_index.describe_index_stats()
        print(f"Index stats: {stats}")
        
        print_header("Using LangChain for Retrieval")
        qa_chain = setup_langchain_retrieval(
            user_query,
            os.getenv("OPENAI_API_KEY"),
            os.getenv("PINECONE_API_KEY"),
            index_name
        )
        
        # Get response from LangChain
        print(f"Querying with LangChain: {user_query}")
        langchain_response = qa_chain.invoke({"query": user_query})
        
        # Print response
        print("\nLangChain Response:")
        print(langchain_response["result"])
        
        # Also query Pinecone directly for comparison
        print_header("Also Querying Pinecone Directly")
        top_contexts = query_pinecone_directly(user_query, openai_client, pinecone_index)
        
        # Combine the top contexts
        context = "\n\n".join([ctx["text"] for ctx in top_contexts])
        
        # Limit context length to approximately 10,000 characters to avoid token limit
        if len(context) > 10000:
            context = context[:10000] + "...\n(Context truncated due to length constraints)"
        
        # Generate response using OpenAI
        print_header(f"Generating Direct Response for: {user_query}")
        
        system_prompt = """
        You are a financial data extraction tool. Your ONLY task is to extract EXACT financial numbers from the provided context.
        
        DO NOT add any explanations, interpretations, or additional text. ONLY return the exact numbers and account codes as they appear in the document.
        
        For example, if the query asks about "401000 - Inpatient Medicare Trad Intensive Care Revenue", you should ONLY return something like:
        "401000 - Inpatient Medicare Trad Intensive Care Revenue: $11,807,381 $16,322,926 ($4,515,544) (27.7%)"
        
        DO NOT make up any numbers. If the exact data is not found, simply state "No exact match found for [query term]".
        
        DO NOT say "I don't have enough information" or provide any other explanations.
        
        ONLY return the raw financial data exactly as it appears in the document.
        
        If page numbers are available, include them in a minimal format: "[Page X]".
        """
        
        user_prompt = f"""
        Context:
        {context}
        
        Query: {user_query}
        
        Extract ONLY the exact financial numbers and account codes related to this query. Return the raw data exactly as it appears in the document with no additional text or explanations.
        """
        
        print("Sending request to OpenAI...")
        completion = openai_client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )
        
        # Print response
        direct_response = completion.choices[0].message.content
        print("\nDirect Response:")
        print(direct_response)
        
        # Compare responses
        print_header("Comparing Responses")
        print("Both approaches should provide similar information. If one approach provides more detailed or accurate information, consider using that approach for future queries.")
        
        print_result(True, "RAG pipeline with LangChain and direct Pinecone queries executed successfully")
        
    except Exception as e:
        print_result(False, f"Error during RAG pipeline execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 