import os
import sys
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from rich.console import Console
from rich.panel import Panel

# Load environment variables
load_dotenv()

# Initialize rich console for better output formatting
console = Console()

def test_vector_store():
    """Test if the vector store is working correctly."""
    console.print(Panel.fit("Testing Vector Store Functionality", style="green"))
    
    # Initialize embedding model - same as in rag_analyzer.py
    console.print("Initializing embedding model...", style="yellow")
    embeddings = HuggingFaceEmbeddings(model_name="Snowflake/snowflake-arctic-embed-l")
    
    # Initialize Vector DB with the same path as rag_analyzer, explicitly using the Security_Analysis collection
    console.print("Connecting to vector database...", style="yellow")
    vector_db = Chroma(
        embedding_function=embeddings, 
        persist_directory="./chroma_db",
        collection_name="Security_Analysis"  # Explicitly use the collection with data
    )
    
    # Check if the vector database has any collections
    console.print("Checking vector database collections...", style="yellow")
    try:
        # Get collection info - handle v0.6.0+ API changes
        collection_names = vector_db._client.list_collections()
        collection_count = len(collection_names)
        console.print(f"Found {collection_count} collections in the database", style="green")
        
        # List collection names
        for i, collection_name in enumerate(collection_names):
            console.print(f"Collection {i+1}: {collection_name}")
            
            # Try to get more information about each collection
            try:
                collection = vector_db._client.get_collection(collection_name)
                count = collection.count()
                console.print(f"  - Contains {count} documents", style="blue")
            except Exception as e:
                console.print(f"  - Error getting collection details: {e}", style="red")
    except Exception as e:
        console.print(f"Error checking collections: {e}", style="red")
        return False
    
    # Check if our specified collection exists and has documents
    try:
        console.print("\nChecking Security_Analysis collection...", style="yellow")
        security_collection = vector_db._client.get_collection("Security_Analysis")
        doc_count = security_collection.count()
        console.print(f"Security_Analysis collection contains {doc_count} documents", style="green")
        if doc_count == 0:
            console.print("Security_Analysis collection is empty!", style="red")
    except Exception as e:
        console.print(f"Error checking Security_Analysis collection: {e}", style="red")
    
    # Perform a simple similarity search using the retriever
    console.print("\nPerforming test query on vector database using retriever...", style="yellow")
    console.print("Using collection: Security_Analysis", style="cyan")
    
    # List of test queries to try
    test_queries = [
        "value investing principles",
        "stock market analysis",
        "technical analysis indicators",
        "portfolio diversification"
    ]
    
    # Try each query
    for query in test_queries:
        try:
            # Create retriever
            retriever = vector_db.as_retriever(search_kwargs={"k": 3})
            
            # Query the database
            console.print(f"\nQuery: '{query}'", style="cyan")
            results = retriever.invoke(query)
            
            # Check if we got any results
            if results and len(results) > 0:
                console.print(f"✅ Success! Retrieved {len(results)} documents", style="green")
                
                # Show sample of first result
                console.print("\nSample from first result:", style="cyan")
                first_doc = results[0]
                # Get the first 200 characters of content to preview
                preview = first_doc.page_content[:200] + "..." if len(first_doc.page_content) > 200 else first_doc.page_content
                console.print(preview)
                
                # Show metadata
                console.print("\nMetadata:", style="cyan")
                for key, value in first_doc.metadata.items():
                    console.print(f"  {key}: {value}")
            else:
                console.print("❌ No results returned", style="red")
        except Exception as e:
            console.print(f"❌ Error during query: {e}", style="red")
    
    console.print("\nVector store test completed.", style="green")
    
    # Check if we need to update the rag_analyzer.py file
    console.print("\nAnalyzing issue in rag_analyzer.py...", style="yellow")
    console.print("The issue might be that rag_analyzer.py is not specifying the 'Security_Analysis' collection name.", style="cyan")
    console.print("You may need to update agents/rag_analyzer.py to explicitly use the Security_Analysis collection:", style="cyan")
    console.print('vector_db = Chroma(embedding_function=embeddings, persist_directory="./chroma_db", collection_name="Security_Analysis")', style="green")
    
    return True

if __name__ == "__main__":
    success = test_vector_store()
    if not success:
        sys.exit(1) 