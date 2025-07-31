from rag_qdrant import AgenticRAG

def main():
    # Initialize the RAG system
    rag = AgenticRAG()
    
    # Verify collections first
    print("🔍 Verifying collections...")
    rag.verify_collections()
    
    # Ask if we should recreate collections
    recreate = input("\nDo you want to recreate the collections? This will delete all data. (y/n): ").lower() == 'y'
    
    if recreate:
        print("\n♻️  Recreating collections...")
        rag._ensure_collections_exist(recreate=True)
        print("✅ Collections recreated successfully!")
        
        # Verify again after recreation
        print("\n🔍 Verifying collections after recreation...")
        rag.verify_collections()
    
    print("\nDone!")

if __name__ == "__main__":
    main()
