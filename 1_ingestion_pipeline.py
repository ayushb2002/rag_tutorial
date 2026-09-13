import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# Step-1: Load the context files
def load_documents(docs_path="docs"):
    """
    Loads context documents in memory.
    """
    
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your files.")

    # Only look for .txt files and look recursively in folders and sub-folders.
    # Since we are working with text files, we are using TextLoader as loader class. The loader class changes as the file extension changes.
    loader = DirectoryLoader(
        path=docs_path,
        glob="**/*.pdf",
        recursive=True,
        loader_cls=PyPDFLoader
    )

    documents = loader.load()
    
    if len(documents) == 0:
        raise FileNotFoundError(f"No .pdf files were found at path {docs_path}. Please add your documents.")

    return documents

# Step-2: Chunk the context files
def split_documents(documents, chunk_size=800, chunk_overlap=150):
    """
    Splits documents into chunks with overlap
    """
    
    print("Splitting documents into chunks...")
    
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="\n"
    )
    
    chunks = text_splitter.split_documents(documents)
    
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    
    return chunks

# Creates and Persists a Chroma DB vector storage
def create_vector_store(chunks, persist_directory="db"):
    """
    Creates a vector store from chunks and persists it to disk.
    """
    print("Creating embeddings and storing in ChromaDB")
    
    embeddings_model = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
    
    print("----Creating a vector storage----")
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    print("----Finished creating vector store----")
    print(f"Vector store created and persisted at {persist_directory}")
    
    return vector_store

def main():
    # 1. Read the context files
    documents = load_documents(docs_path="docs")
    # 2. Break the context files into chunks
    chunks = split_documents(documents)
    # 3. Create the vector store
    vector_store = create_vector_store(chunks=chunks, persist_directory="db")
    # 4. Print the result
    print("Ingestion pipeline is complete!")

if __name__ == "__main__":
    main()
