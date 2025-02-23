import os
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_pdf_file(folder_path: str):
    """
    Loads all PDF files from the given folder using PyPDFLoader.
    """
    loader = DirectoryLoader(
        folder_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    return loader.load()

def text_split(extracted_data):
    """
    Splits the extracted documents into smaller text chunks for processing.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks
