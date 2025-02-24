import os
import sys
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def print_header(title):
    """Print a formatted header"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")

def create_pinecone_index():
    """Check and create Pinecone index if it doesn't exist"""
    print_header("Pinecone Index Creation")
    
    # Load environment variables
    load_dotenv()
    
    try:
        # Import the Pinecone library
        from pinecone.grpc import PineconeGRPC
        from pinecone import ServerlessSpec
        
        # Get environment variables - ensure all values come from .env
        pinecone_api_key = os.getenv('PINECONE_API_KEY')
        pinecone_index_name = os.getenv('PINECONE_INDEX_NAME')
        pinecone_cloud = os.getenv('PINECONE_CLOUD')
        pinecone_region = os.getenv('PINECONE_REGION')
        pinecone_dimension = int(os.getenv('PINECONE_DIMENSION', '1536'))  # Default to 1536 if not specified
        pinecone_metric = os.getenv('PINECONE_METRIC', 'cosine')  # Default to cosine if not specified
        
        # Validate required environment variables
        missing_vars = []
        if not pinecone_api_key:
            missing_vars.append('PINECONE_API_KEY')
        if not pinecone_index_name:
            missing_vars.append('PINECONE_INDEX_NAME')
        if not pinecone_cloud:
            missing_vars.append('PINECONE_CLOUD')
        if not pinecone_region:
            missing_vars.append('PINECONE_REGION')
            
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        # Print configuration
        print(f"API Key: {'*' * 5}{pinecone_api_key[-5:] if pinecone_api_key else 'Not set'}")
        print(f"Index Name: {pinecone_index_name}")
        print(f"Cloud: {pinecone_cloud}")
        print(f"Region: {pinecone_region}")
        print(f"Dimension: {pinecone_dimension}")
        print(f"Metric: {pinecone_metric}")
        
        # Initialize Pinecone
        print("\nInitializing Pinecone client...")
        pc = PineconeGRPC(api_key=pinecone_api_key)
        
        # List existing indexes
        print("Listing existing indexes...")
        indexes = [index.name for index in pc.list_indexes()]
        print(f"Found {len(indexes)} indexes: {', '.join(indexes) if indexes else 'None'}")
        
        # Check if our index exists
        if pinecone_index_name in indexes:
            print(f"\n✅ Index '{pinecone_index_name}' already exists!")
            
            # Get index details
            index = pc.Index(pinecone_index_name)
            stats = index.describe_index_stats()
            vector_count = stats.get('total_vector_count', 0)
            print(f"Index contains {vector_count} vectors")
            
            # Get index description
            index_description = pc.describe_index(pinecone_index_name)
            print(f"Index dimension: {index_description.dimension}")
            print(f"Index metric: {index_description.metric}")
            print(f"Index status: {index_description.status}")
            
            return True
        else:
            print(f"\n⚠️ Index '{pinecone_index_name}' does not exist. Creating it now...")
            
            try:
                # Create the index using values from .env
                pc.create_index(
                    name=pinecone_index_name,
                    dimension=pinecone_dimension,
                    metric=pinecone_metric,
                    spec=ServerlessSpec(
                        cloud=pinecone_cloud,
                        region=pinecone_region
                    )
                )
                print(f"\n✅ Successfully created index '{pinecone_index_name}'!")
                print(f"Dimension: {pinecone_dimension}")
                print(f"Metric: {pinecone_metric}")
                print(f"Cloud: {pinecone_cloud}")
                print(f"Region: {pinecone_region}")
                
                # Wait for the index to be ready
                print("\nWaiting for the index to be ready...")
                import time
                time.sleep(10)  # Give it some time to initialize
                
                # Verify the index was created
                new_indexes = [index.name for index in pc.list_indexes()]
                if pinecone_index_name in new_indexes:
                    print(f"✅ Index '{pinecone_index_name}' is now available!")
                    return True
                else:
                    print(f"❌ Index '{pinecone_index_name}' was not found after creation. It may still be initializing.")
                    return False
                
            except Exception as e:
                print(f"❌ Failed to create index: {str(e)}")
                print("\nPossible reasons:")
                print("  • You may not have permissions to create indexes")
                print("  • Your account may have reached its index limit")
                print("  • The region or cloud provider may not be available for your account")
                print("  • There might be an issue with your API key")
                return False
    
    except ImportError as e:
        print(f"❌ Import error: {str(e)}")
        print("Make sure you have installed the correct Pinecone package:")
        print("pip uninstall pinecone-client -y")
        print("pip uninstall pinecone -y")
        print("pip install \"pinecone[grpc]\"")
        return False
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    create_pinecone_index() 