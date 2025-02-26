#!/usr/bin/env python3
"""
Common utility functions used across multiple scripts.
This module provides:
1. Standardized printing functions
2. Environment loading and validation
3. Common file operations
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

def print_header(message: str) -> None:
    """Print a formatted header message."""
    print("\n" + "="*80)
    print(f" {message}")
    print("="*80)

def print_result(success: bool, message: str) -> None:
    """Print a formatted result message."""
    status = "✅" if success else "❌"
    print(f"{status} {message}")

def load_environment(required_vars: Optional[List[str]] = None) -> Dict[str, str]:
    """
    Load and validate environment variables.
    
    Args:
        required_vars: List of required environment variable names
        
    Returns:
        Dictionary of environment variables
        
    Raises:
        ValueError: If any required variables are missing
    """
    load_dotenv()
    
    if required_vars is None:
        return {key: val for key, val in os.environ.items()}
    
    env = {}
    missing_vars = []
    
    for var in required_vars:
        value = os.environ.get(var)
        if not value:
            missing_vars.append(var)
        else:
            env[var] = value
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    return env

def get_file_hash(file_path: str) -> str:
    """
    Calculate SHA-256 hash of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Hex digest of the file hash
    """
    sha256_hash = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        # Read the file in chunks to handle large files
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    
    return sha256_hash.hexdigest()

def ensure_directory_exists(directory_path: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory
    """
    Path(directory_path).mkdir(parents=True, exist_ok=True)

def save_json(data: Any, file_path: str) -> None:
    """
    Save data as JSON to a file.
    
    Args:
        data: Data to save
        file_path: Path to the output file
    """
    ensure_directory_exists(os.path.dirname(file_path))
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(file_path: str) -> Any:
    """
    Load JSON data from a file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Loaded data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    with open(file_path, 'r') as f:
        return json.load(f)

def verify_environment() -> bool:
    """
    Verify that we're running in the correct conda environment.
    
    Returns:
        True if running in the correct environment, False otherwise
    """
    current_env = os.environ.get('CONDA_DEFAULT_ENV')
    if current_env != 'finsight-ai':
        print("❌ Error: This script must be run in the 'finsight-ai' conda environment")
        print("Please run:")
        print("    conda activate finsight-ai")
        print("Then try again.")
        return False
    print("✅ Running in correct conda environment: finsight-ai")
    return True 