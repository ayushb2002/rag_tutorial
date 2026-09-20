from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List

load_dotenv()

persistent_directory = "db/pdf"
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o", temperature=0)

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# Structured Outputs
class QueryVariations(BaseModel):
    queries: List[str]
    
# Feeding Original Query
original_query = "Which modern day dish is ranked the best in Japanese cuisine?"
print(f"Original Query: {original_query}\n")

# ----------------------------------------------------------
# Step 1: Generating multiple query variations
# ----------------------------------------------------------

llm_with_tools = llm.with_structured_output(QueryVariations)

prompt = f"""Generate 3 different variations of this query that would help me retrieve relevant documents:

Original query: {original_query}

Return 3 alternative queries that rephrase or approach the same question from different angles.
"""

response = llm_with_tools.invoke(prompt)
query_variations = response.queries

# print("Generated Query Variations:")
# for i, variations in enumerate(query_variations, 1):
#     print(f"{i}. {variations}")
    
# print("\n"+"="*60)

# ----------------------------------------------------------
# Step 2: Search with Each query variations and store results
# ----------------------------------------------------------

retriever = db.as_retriever(search_kwargs={"k": 5})
all_retrieval_results = []

for i, query in enumerate(query_variations, 1):
    print(f"\n===RESULT FOR QUERY {i}: {query} ===")
    
    docs = retriever.invoke(query)
    all_retrieval_results.append(docs)
    
    print(f"Retrieved {len(docs)} documents:\n")
    
    for j, doc in enumerate(docs, 1):
        print(f"Document: {j}:")
        print(f"{doc.page_content[:150]}...\n")
        
    print("-"*50)
print("\n"+"="*60)
print("Multi-Query Retrieval Complete!")