from rag_engine import build_rag_index

if __name__ == "__main__":
    print("Starting RAG Index Build Process...")
    try:
        build_rag_index()
        print("Build Complete.")
    except Exception as e:
        print(f"Build Failed: {e}")
