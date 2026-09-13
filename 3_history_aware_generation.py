from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


persistent_directory = "db"
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

model = ChatOpenAI(model="gpt-4o-mini")

chat_history = []

def start_chat():
    print("Ask me any question! Type 'quit' to exit.")

    while True:
        question = input("\n Your query: ")    

        if question.lower() == "quit":
            print("Goodbye!")
            break
        
        ask_question(question)
        
def ask_question(user_question):
    print(f"\n----User query: {user_question}----")
    
    # Step-1: Make the question clear using conversation history
    if chat_history:
        # Ask AI to make the question standalone
        message = [
            SystemMessage(content="Given the chat history, rewrite the new questions to be standalone and searchable. Just return the rewritten question.")
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]
        
        result = model.invoke(message)
        search_question = result.content.strip()
        print(f"Searching for: {search_question}")
        
    else:
        search_question = user_question
    
    # Step-2: Find relevant documents
    
    retriever = db.as_retriever(search_kwargs={"k":3})
    docs = retriever.invoke(search_question)
    
    print(f"Found {len(docs)} relevant documents.")
    for i, doc in enumerate(docs, 1):
        lines = doc.page_content.split('\n')[:2]
        preview = '\n'.join(lines)
        print(f"Doc {i}. {preview}")

    # Step-3: Create the final prompt
    
    llm_query = f"""
    Based on the following documents, please provide me the answer to this question: {user_question}

    Documents:
    {chr(10).join([f"- {doc.page_content}" for doc in docs])}

    Please provide a clear, helpful answer using only the information from these documents. 
    If you can't find the answer in the documents, say "Based on the given context, I don't have enough information to answer the question".
    """

    # Define the message for the model
    message = [
        SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and conversation history.")
    ] + chat_history + [
        HumanMessage(content=llm_query),
    ]

    # Invoke the model with llm query
    result = model.invoke(message)
    answer = result.content
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))
    
    
    print(f"Answer: {result.content}")
    
    
if __name__ == "__main__":
    start_chat()