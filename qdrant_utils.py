from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import OpenAIEmbeddings

def store_embeddings_with_metadata(chunk_tuples: list[tuple[str, dict]], collection_name="electronic_manuals"):
    texts = [c[0] for c in chunk_tuples]
    metadatas = [c[1] for c in chunk_tuples]

    # Initialize the client
    client = QdrantClient(host="localhost", port=6333)

    # Create the collection if it doesn't exist
    try:
        client.get_collection(collection_name)
    except Exception:
        # Collection doesn't exist, create it
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=1536,  # OpenAI embeddings size
                distance=models.Distance.COSINE
            )
        )

    # Now initialize Qdrant with the existing collection
    qdrant = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=OpenAIEmbeddings()
    )
    qdrant.add_texts(texts=texts, metadatas=metadatas)
