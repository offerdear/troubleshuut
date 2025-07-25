import os
import sys
from rag_2 import AgenticRAG, extract_text_from_pdf, split_text
from qdrant_client.http.models import PointStruct
from datetime import datetime
import uuid

def ingest_pdf(pdf_path, device_name):
    print(f"Ingesting {pdf_path} as device '{device_name}'...")
    text = extract_text_from_pdf(pdf_path)
    chunks = split_text(text)
    rag = AgenticRAG()
    documents = chunks
    payloads = [
        {
            'type': 'text_chunk',
            'source': pdf_path,
            'device': device_name,
            'chunk_index': i,
            'timestamp': datetime.now().isoformat()
        }
        for i, chunk in enumerate(chunks)
    ]
    ids = [str(uuid.uuid4()) for _ in chunks]
    embeddings = rag.get_embeddings(documents)
    points = [PointStruct(id=ids[i], vector=embeddings[i], payload=payloads[i]) for i in range(len(ids))]
    rag.qdrant_client.upsert(collection_name=rag.knowledge_collection, points=points)
    print(f"✅ Ingested {len(chunks)} chunks for {device_name}")

if __name__ == "__main__":
    pdf_dir = "."  # or your manuals directory
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            device_name = os.path.splitext(filename)[0].replace("_User_Guide", "").replace("_", " ").strip()
            ingest_pdf(os.path.join(pdf_dir, filename), device_name) 