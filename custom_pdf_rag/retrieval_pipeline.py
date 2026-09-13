from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

# OpenAI model used to create the vector embeddings
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Loading the vector embeddings from ChromaDB
db = Chroma(
    persist_directory="db/pdf",
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# Enter a query
query = "Which is the most famous dish in the Indian cuisine?"

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 10,
        "score_threshold": 0.3
    }
)

relevant_docs = retriever.invoke(query)

llm_query = f"""
Based on the following documents, please provide me the answer to this question: {query}

Documents (.md format):
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents. 
If you can't find the answer in the documents, say "Based on the given context, I don't have enough information to answer the question".
"""

# Load ChatOpenAI model
model = ChatOpenAI(model="gpt-4o-mini")

# Define the message for the model
message = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=llm_query),
]

# Invoke the model with llm query
result = model.invoke(message)

print("\n----Generated Response----")
# print(f"Full Result: \n{result}")
print(f"\n{result.content}")