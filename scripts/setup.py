#!/usr/bin/env python3
"""
Unified setup script for the FinSight AI project.
This script serves as an entry point for setting up all components of the project:
1. Google Cloud Platform resources
2. Pinecone vector database
3. RAG pipeline configuration
4. Local environment verification

Usage:
    python scripts/setup.py --all                # Set up everything
    python scripts/setup.py --gcp                # Set up only GCP resources
    python scripts/setup.py --pinecone           # Set up only Pinecone
    python scripts/setup.py --rag                # Set up only RAG pipeline
    python scripts/setup.py --verify             # Verify environment
"""

import os
import sys
import argparse
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def print_header(message):
    """Print a formatted header message."""
    print("\n" + "="*80)
    print(f" {message}")
    print("="*80)

def print_result(success, message):
    """Print a formatted result message."""
    status = "✅" if success else "❌"
    print(f"{status} {message}")

def verify_environment():
    """Verify that the environment is properly set up."""
    print_header("Verifying Environment")
    
    # Check Python version
    python_version = sys.version.split()[0]
    print(f"Python version: {python_version}")
    
    # Check if .env file exists
    env_file = project_root / ".env"
    if env_file.exists():
        print_result(True, "Found .env file")
    else:
        print_result(False, "Missing .env file. Please create one based on .env.example")
        return False
    
    # Check if required directories exist
    required_dirs = ["data", "data/temp", "data/raw", "data/processed"]
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print_result(True, f"Created directory: {dir_name}")
        else:
            print_result(True, f"Found directory: {dir_name}")
    
    return True

def setup_gcp():
    """Set up Google Cloud Platform resources."""
    print_header("Setting up Google Cloud Platform")
    
    try:
        from scripts.setup.setup_gcp import setup_gcp as run_gcp_setup
        success = run_gcp_setup()
        return success
    except ImportError as e:
        print_result(False, f"Failed to import GCP setup module: {e}")
        return False
    except Exception as e:
        print_result(False, f"Error setting up GCP: {e}")
        return False

def setup_pinecone():
    """Set up Pinecone vector database."""
    print_header("Setting up Pinecone")
    
    try:
        from scripts.setup.setup_pinecone import setup_pinecone as run_pinecone_setup
        success = run_pinecone_setup()
        return success
    except ImportError as e:
        print_result(False, f"Failed to import Pinecone setup module: {e}")
        return False
    except Exception as e:
        print_result(False, f"Error setting up Pinecone: {e}")
        return False

def setup_rag():
    """Set up RAG pipeline."""
    print_header("Setting up RAG Pipeline")
    
    try:
        from scripts.setup.setup_rag_pipeline import init_pinecone, verify_pinecone_setup, verify_openai_api
        
        # Initialize Pinecone
        init_pinecone()
        
        # Verify Pinecone setup
        pinecone_success = verify_pinecone_setup()
        
        # Verify OpenAI API
        openai_success = verify_openai_api()
        
        return pinecone_success and openai_success
    except ImportError as e:
        print_result(False, f"Failed to import RAG setup module: {e}")
        return False
    except Exception as e:
        print_result(False, f"Error setting up RAG pipeline: {e}")
        return False

def main():
    """Main entry point for the setup script."""
    parser = argparse.ArgumentParser(description="Set up FinSight AI project components")
    parser.add_argument("--all", action="store_true", help="Set up all components")
    parser.add_argument("--gcp", action="store_true", help="Set up GCP resources")
    parser.add_argument("--pinecone", action="store_true", help="Set up Pinecone")
    parser.add_argument("--rag", action="store_true", help="Set up RAG pipeline")
    parser.add_argument("--verify", action="store_true", help="Verify environment")
    
    args = parser.parse_args()
    
    # If no arguments are provided, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    # Verify environment first
    if args.verify or args.all:
        env_ok = verify_environment()
        if not env_ok and not args.verify:
            print("Environment verification failed. Please fix the issues before continuing.")
            return
    
    # Set up GCP
    if args.gcp or args.all:
        gcp_ok = setup_gcp()
        print_result(gcp_ok, "GCP setup")
    
    # Set up Pinecone
    if args.pinecone or args.all:
        pinecone_ok = setup_pinecone()
        print_result(pinecone_ok, "Pinecone setup")
    
    # Set up RAG
    if args.rag or args.all:
        rag_ok = setup_rag()
        print_result(rag_ok, "RAG pipeline setup")
    
    print_header("Setup Complete")
    print("You can now run the Streamlit app with:")
    print("    streamlit run app/streamlit/app.py")

if __name__ == "__main__":
    main() 