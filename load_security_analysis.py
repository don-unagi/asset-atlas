import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def load_security_analysis():
    """Load the security analysis book into the vector database."""
    print("Loading Security Analysis book into vector database...")
    
    # Load the text file
    file_path = "security-analysis.txt"
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    
    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1250,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    
    print(f"Split the document into {len(chunks)} chunks")
    
    # Initialize embedding model
    embeddings = HuggingFaceEmbeddings(model_name="Snowflake/snowflake-arctic-embed-l")
    
    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db",
        collection_name="Security_Analysis"
    )
    
    print(f"Created vector store with {len(chunks)} documents")
    print("Vector store persisted to ./chroma_db")
    
    return vectorstore

if __name__ == "__main__":
    load_security_analysis() 