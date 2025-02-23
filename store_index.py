import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

# Pinecone + LangChain + Embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings

# Local imports from helper.py
from src.helper import load_pdf_file, text_split

# Load API keys from .env
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Check if API keys are loaded correctly
if not PINECONE_API_KEY:
    raise ValueError("Error: PINECONE_API_KEY not found in environment variables.")
if not OPENAI_API_KEY:
    raise ValueError("Error: OPENAI_API_KEY not found in environment variables.")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "quickstart"

def create_index_if_not_exists():
    """
    Creates a Pinecone serverless index if it doesn't already exist.
    """
    try:
        # Check existing indexes before creating a new one
        existing_indexes = [index["name"] for index in pc.list_indexes()]
        
        if index_name not in existing_indexes:
            pc.create_index(
                name=index_name,
                dimension=384,  # Must match the embedding dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print(f"✅ Index '{index_name}' created.")
        else:
            print(f"⚠️ Index '{index_name}' already exists. Skipping creation.")
    
    except Exception as e:
        print(f"⚠️ Index creation skipped due to error: {e}")

def load_or_create_docsearch():
    """
    Loads PDF data, splits it, and creates (or reuses) a Pinecone vector store.
    Returns both the docsearch and the embeddings object.
    """
    # 1. Load data from PDF folder
    documents = load_pdf_file('Data/')

    # 2. Split into text chunks
    text_chunks = text_split(documents)

    # 3. Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    # 4. Check if index exists before creating docsearch
    existing_indexes = [index["name"] for index in pc.list_indexes()]
    
    if index_name in existing_indexes:
        print(f"⚠️ Using existing Pinecone index: {index_name}")
        docsearch = PineconeVectorStore.from_existing_index(index_name, embedding=embeddings)
    else:
        print(f"✅ Creating new Pinecone index and storing embeddings.")
        docsearch = PineconeVectorStore.from_documents(
            documents=text_chunks,
            index_name=index_name,
            embedding=embeddings,
        )

    return docsearch, embeddings
