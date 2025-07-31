# from rag_qdrant2 import AgenticRAG

# # Initialize the RAG system
# rag = AgenticRAG()  # This will use the updated class with clean_none_content_documents

# # Clean up documents with None content
# num_removed = rag.clean_none_content_documents()
# print(f"Removed {num_removed} documents with None or empty content")


from qdrant_client import QdrantClient
import os

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

client.delete_collection(collection_name="electronic_manuals")
