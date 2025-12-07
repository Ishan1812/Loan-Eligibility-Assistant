import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

# Define paths
DATA_PATH = "data/bank_policies/loan_rules.txt"
DB_PATH = "faiss_index"

def build_rag_index():
    """
    Ingests the policy document, chunks it, and builds a FAISS index.
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Policy file not found at {DATA_PATH}")

    print("Loading policies...")
    loader = TextLoader(DATA_PATH)
    documents = loader.load()

    print("Splitting text...")
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    print("Creating embeddings and vector store...")
    # Using Ollama Embeddings with Mistral
    embeddings = OllamaEmbeddings(model="mistral")
    db = FAISS.from_documents(docs, embeddings)

    print(f"Saving index to {DB_PATH}...")
    db.save_local(DB_PATH)
    print("RAG Index built successfully.")

def query_policies(question: str) -> str:
    """
    Retrieves relevant policy rules for a given question.
    """
    if not os.path.exists(DB_PATH):
        print("Index not found, building it now...")
        build_rag_index()

    embeddings = OllamaEmbeddings(model="mistral")
    db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    
    retriever = db.as_retriever(search_kwargs={"k": 2})
    docs = retriever.invoke(question)
    
    return "\n\n".join([doc.page_content for doc in docs])

if __name__ == "__main__":
    # If run directly, build the index
    try:
        build_rag_index()
        # Test query
        print("\nTest Query: 'What is the minimum salary?'")
        print(query_policies("What is the minimum salary?"))
    except Exception as e:
        print(f"Error: {e}")
        print("Ensure you have Ollama installed and 'mistral' model pulled.")
