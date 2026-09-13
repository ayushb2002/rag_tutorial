from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

############
# Method 1: Basic Similarity Search 
############

def basic_similarity_search(db, print_chunks=True):
    """Basic Similarity search with k = 3"""
    
    print("Basic Similarity Search (k = 3)...\n")

    retriever = db.as_retriever(search_kwargs={"k": 3})

    docs = retriever.invoke(query)
    print(f"Retrieved {len(docs)} documents.\n")
    
    if print_chunks:
        for i, doc in enumerate(docs):
            print(f"Document {i+1}:")
            print(f"{doc.page_content}\n")
            
        print("-"*60)

############
# Method 2: Similarity with Score Threshold 
############

def similarity_with_score_threshold(db, print_chunks=True):
    """Similarity with k = 3 and score_threshold = 0.3"""
    
    print("Similarity with Score Threshold...\n")

    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 3,
            "score_threshold": 0.3 # Only returns docs with similarity >= 0.3
        }
    )

    docs = retriever.invoke(query)
    print(f"Retrieved {len(docs)} documents (threshold: 0.3): \n")
    
    if print_chunks:
        for i, doc in enumerate(docs):
            print(f"Document {i}: ")
            print(f"{doc.page_content}\n")
            
        print("-"*60)

############
# Method 3: Maximum Marginal Relevance (MMR) 
############

def maximum_marginal_relevance(db, print_chunks=True):
    """Retrieval method which uses mmr with the following config: {k=3, fetch_k=10, lambda_mult=0.5}"""
    
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,             # Final number of docs
            "fetch_k": 10,      # Initial pool to select from
            "lambda_mult": 0.5  # 0=max diversity, 1=max relevance
        }
    )

    docs = retriever.invoke(query)
    print(f"Retrieved {len(docs)} documents (lambda: 0.5): \n")

    if print_chunks:
        for i, doc in enumerate(docs):
            print(f"Document {i}: ")
            print(f"{doc.page_content}\n")
            
        print("-"*60)
        
if __name__ == "__main__":
    
    # Setup
    persistent_directory = "db/pdf"
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    db = Chroma(
        persist_directory=persistent_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    # query = "Which is the oldest dish in the Indian cuisine?"
    
    while True:
        query = str(input("Enter query on world cuisines: "))
        print(f"\nYour query: {query}\n")
        
        print(f"""
            Select one of the following retrieval options:
            \t1. Basic Similarity Search
            \t2. Similarity Search with Score Threshold
            \t3. Maximum Marginal Relevance
        """)
        opt = str(input("Enter choice: "))
        
        if opt == "1":
            basic_similarity_search(db)
            print("\n")
        elif opt == "2":
            similarity_with_score_threshold(db)
            print("\n")
        elif opt == "3":
            maximum_marginal_relevance(db)
            print("\n")
        else:
            break
        
        opt = input("\nExit (Y/N)?: ")
        if opt in ["Y", "y"]:
            break