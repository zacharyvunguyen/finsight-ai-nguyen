import os
import sys
import time
from dotenv import load_dotenv
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def print_header(title):
    """Print a formatted header"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")

def print_result(status, details=None):
    """Print a formatted test result"""
    status_icon = "✅" if status else "❌"
    print(f"{status_icon} {'Success' if status else 'Failed'}")
    if details:
        for line in details:
            print(f"   • {line}")

def test_pinecone_connection():
    """Test connectivity to Pinecone"""
    print_header("Pinecone Connection Test")
    
    # Load environment variables
    load_dotenv()
    
    try:
        print("\n📍 Testing Pinecone...")
        
        # Import the Pinecone library - using only the GRPC client as recommended
        try:
            from pinecone.grpc import PineconeGRPC
            print("   • Successfully imported Pinecone GRPC client")
        except ImportError:
            print("   • Failed to import Pinecone GRPC client")
            print("   • Attempting to provide installation instructions...")
            print("\n⚠️ Pinecone SDK Installation Instructions:")
            print("   1. Uninstall any existing pinecone packages:")
            print("      pip uninstall pinecone-client -y")
            print("      pip uninstall pinecone -y")
            print("   2. Install the new Pinecone package with GRPC support:")
            print("      pip install \"pinecone[grpc]\"")
            raise ImportError("Pinecone GRPC client not installed correctly")
        
        # Get all environment variables
        pinecone_api_key = os.getenv('PINECONE_API_KEY')
        pinecone_index_name = os.getenv('PINECONE_INDEX_NAME')
        pinecone_cloud = os.getenv('PINECONE_CLOUD')
        pinecone_region = os.getenv('PINECONE_REGION')
        pinecone_dimension = os.getenv('PINECONE_DIMENSION', '1536')
        pinecone_metric = os.getenv('PINECONE_METRIC', 'cosine')
        
        # Print configuration for debugging
        print(f"   • API Key: {'*' * 5}{pinecone_api_key[-5:] if pinecone_api_key else 'Not set'}")
        print(f"   • Index Name: {pinecone_index_name or 'Not set'}")
        print(f"   • Cloud: {pinecone_cloud or 'Not set'}")
        print(f"   • Region: {pinecone_region or 'Not set'}")
        print(f"   • Dimension: {pinecone_dimension or 'Not set'}")
        print(f"   • Metric: {pinecone_metric or 'Not set'}")
        
        # Validate required environment variables
        missing_vars = []
        if not pinecone_api_key:
            missing_vars.append('PINECONE_API_KEY')
        if not pinecone_index_name:
            missing_vars.append('PINECONE_INDEX_NAME')
            
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        # Initialize Pinecone with the GRPC client
        print("   • Initializing Pinecone GRPC client...")
        pc = PineconeGRPC(api_key=pinecone_api_key)
        print("   • Successfully initialized Pinecone client")
        
        # List indexes
        print("   • Listing indexes...")
        indexes = [index.name for index in pc.list_indexes()]
        print(f"   • Found {len(indexes)} indexes: {', '.join(indexes) if indexes else 'None'}")
        
        # Check if our index exists
        if pinecone_index_name in indexes:
            # Connect to the index
            print(f"   • Connecting to index '{pinecone_index_name}'...")
            index = pc.Index(pinecone_index_name)
            
            # Get index stats
            print("   • Getting index stats...")
            stats = index.describe_index_stats()
            vector_count = stats.get('total_vector_count', 0)
            
            # Get index description for more details
            index_description = pc.describe_index(pinecone_index_name)
            
            print_result(True, [
                f"Connected to Pinecone successfully",
                f"Index '{pinecone_index_name}' exists",
                f"Index contains {vector_count} vectors",
                f"Dimension: {index_description.dimension}",
                f"Metric: {index_description.metric}",
                f"Status: {index_description.status}"
            ])
            
            # Try a simple query to verify full functionality
            print("   • Testing query functionality...")
            try:
                # Create a test vector of the right dimension
                dimension = int(pinecone_dimension)
                test_vector = [0.0] * dimension
                
                # Query the index with the test vector
                query_response = index.query(
                    vector=test_vector,
                    top_k=1,
                    include_metadata=True
                )
                
                print("   • Query successful")
                print_result(True, [
                    "Full Pinecone functionality verified",
                    "Query operation successful"
                ])
            except Exception as query_error:
                print(f"   • Query test failed: {str(query_error)}")
                print_result(False, [
                    "Connected to index but query operation failed",
                    f"Error: {str(query_error)}"
                ])
        else:
            print_result(False, [
                f"Connected to Pinecone successfully",
                f"Index '{pinecone_index_name}' does not exist",
                f"Available indexes: {', '.join(indexes) if indexes else 'None'}",
                f"Run 'python tests/create_pinecone_index.py' to create the index"
            ])
    except ImportError as e:
        print_result(False, [
            f"Import error: {str(e)}",
            "Make sure you have installed the correct Pinecone package:",
            "pip uninstall pinecone-client -y",
            "pip uninstall pinecone -y",
            "pip install \"pinecone[grpc]\""
        ])
    except Exception as e:
        print_result(False, [
            f"Error: {str(e)}",
            "Check your API key and index name in the .env file",
            "Make sure your Pinecone account is active and the index exists"
        ])
    
    print("\n🔍 Troubleshooting Tips:")
    print("   • Verify your API key in the .env file")
    print("   • Check if your Pinecone index exists and is accessible")
    print("   • Make sure you've installed the correct package:")
    print("     pip uninstall pinecone-client -y")
    print("     pip uninstall pinecone -y")
    print("     pip install \"pinecone[grpc]\"")
    print("   • Verify internet connectivity")
    print("   • Check if your Pinecone account is active")

if __name__ == "__main__":
    test_pinecone_connection() 