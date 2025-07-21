import pandas as pd
from pymilvus import connections, MilvusClient, Collection, CollectionSchema, FieldSchema, DataType, utility
from openai import OpenAI
import json
import os
from dotenv import load_dotenv
from typing import List, Dict
import numpy as np

# Load environment variables
load_dotenv()

class RAGSystem:
    def __init__(self):
        """Initialize the RAG system with your credentials"""
        
        # Get credentials from environment
        openai_api_key = os.getenv("OPENAI_API_KEY")
        milvus_host = os.getenv("MILVUS_HOST", "localhost")
        milvus_port = os.getenv("MILVUS_PORT", "19530")
        
        if not openai_api_key:
            raise ValueError("Missing OPENAI_API_KEY in environment variables.")
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=openai_api_key)
        
        # Connect to Milvus
        self.client = MilvusClient("./milvus_demo.db")  # Uses local file-based storage
        # connections.connect("default", host=milvus_host, port=milvus_port)
        
        # Collection name for storing documents
        self.collection_name = "documents"
        self.collection = None
        
        print("✅ RAG System initialized successfully!")
        print(f"✅ Connected to Milvus at: {milvus_host}:{milvus_port}")
        
    def test_connections(self):
        """Test both OpenAI and Milvus connections"""
        try:
            # Test OpenAI
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=["test"]
            )
            print("✅ OpenAI connection successful!")
            
            # Test Milvus
            print(f"✅ Milvus connection successful!")
            
            return True
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def setup_milvus_collection(self):
        """Create the collection in Milvus for storing documents"""
        
        # Drop existing collection if it exists
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            print("🗑️ Deleted existing collection")
        
        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536)  # OpenAI embedding dimension
        ]
        
        schema = CollectionSchema(fields, description="Document collection for RAG")
        
        # Create collection
        self.collection = Collection(self.collection_name, schema)
        
        # Create index for vector search
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        
        self.collection.create_index("embedding", index_params)
        
        print("✅ Milvus collection created successfully!")
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts using OpenAI"""
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [embedding.embedding for embedding in response.data]
    
    def process_csv_and_ingest(self, csv_file_path: str, text_column: str, chunk_size: int = 500):
        """Process a CSV file and ingest it into Milvus"""
        
        # Load CSV
        df = pd.read_csv(csv_file_path)
        print(f"📊 Loaded CSV with {len(df)} rows")
        print(f"📋 Columns: {list(df.columns)}")
        
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in CSV. Available columns: {list(df.columns)}")
        
        # Prepare documents for ingestion
        documents = {
            "content": [],
            "source": [],
            "metadata": [],
            "embedding": []
        }
        
        texts_for_embedding = []
        
        for idx, row in df.iterrows():
            # Get the text content
            text_content = str(row[text_column])
            
            # Simple chunking by character count
            chunks = self._chunk_text(text_content, chunk_size)
            
            for chunk_idx, chunk in enumerate(chunks):
                # Create metadata
                metadata = {
                    "row_index": idx,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(chunks)
                }
                
                # Add other columns as metadata
                for col in df.columns:
                    if col != text_column:
                        metadata[col] = str(row[col])
                
                documents["content"].append(chunk)
                documents["source"].append(f"row_{idx}_chunk_{chunk_idx}")
                documents["metadata"].append(json.dumps(metadata))
                
                texts_for_embedding.append(chunk)
        
        print(f"📝 Created {len(documents['content'])} document chunks")
        
        # Get embeddings in batches
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts_for_embedding), batch_size):
            batch = texts_for_embedding[i:i+batch_size]
            embeddings = self.get_embeddings(batch)
            all_embeddings.extend(embeddings)
            print(f"🔄 Generated embeddings for batch {i//batch_size + 1}/{(len(texts_for_embedding)-1)//batch_size + 1}")
        
        documents["embedding"] = all_embeddings
        
        # Insert into Milvus
        self.collection.insert(documents)
        self.collection.flush()
        
        print(f"✅ Successfully ingested {len(documents['content'])} documents into Milvus!")
    
    def _chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """Simple text chunking by character count"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i+chunk_size])
        
        return chunks
    
    def search_similar_documents(self, query: str, limit: int = 5) -> List[Dict]:
        """Search for similar documents using vector similarity"""
        
        # Get query embedding
        query_embedding = self.get_embeddings([query])[0]
        
        # Load collection
        self.collection.load()
        
        # Search parameters
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        
        # Search in Milvus
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            output_fields=["content", "source", "metadata"]
        )
        
        documents = []
        for hit in results[0]:
            documents.append({
                "content": hit.entity.get("content"),
                "source": hit.entity.get("source"),
                "metadata": hit.entity.get("metadata"),
                "distance": hit.distance
            })
        
        return documents
    
    def generate_answer(self, query: str, context_limit: int = 5) -> str:
        """Generate an answer using RAG: retrieve relevant docs and generate response"""
        
        print(f"🔍 Searching for: {query}")
        
        # Retrieve relevant documents
        relevant_docs = self.search_similar_documents(query, context_limit)
        
        if not relevant_docs:
            return "No relevant documents found for your query."
        
        # Prepare context
        context = "\n\n".join([doc["content"] for doc in relevant_docs])
        print(f"📚 Found {len(relevant_docs)} relevant documents")
        
        # Generate response using OpenAI
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on the provided context. If you cannot find the answer in the context, say so clearly."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
                }
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    def close(self):
        """Close the Milvus connection"""
        connections.disconnect("default")

def main():
    rag = None
    try:
        # Initialize the RAG system
        rag = RAGSystem()
        
        # Test connections
        if not rag.test_connections():
            return
        
        # Set up Milvus collection
        rag.setup_milvus_collection()
        
        # Get CSV file info from user
        csv_file = input("Enter path to your CSV file: ")
        
        # Load CSV to show columns
        df = pd.read_csv(csv_file)
        print(f"\nAvailable columns: {list(df.columns)}")
        
        text_column = input("Enter the name of the text column to use: ")
        
        # Process and ingest CSV
        rag.process_csv_and_ingest(csv_file, text_column)
        
        print("\n🎉 RAG system is ready! You can now ask questions.")
        print("Type 'quit' to exit.\n")
        
        # Interactive Q&A loop
        while True:
            query = input("Ask a question: ")
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            answer = rag.generate_answer(query)
            print(f"\n💡 Answer: {answer}\n")
            print("-" * 50)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure:")
        print("1. Your .env file is in the same directory")
        print("2. All API keys are correct")
        print("3. Your CSV file path is correct")
        print("4. Milvus is running (docker run -p 19530:19530 milvusdb/milvus:latest)")
    
    finally:
        # Close the connection properly
        if rag:
            rag.close()

if __name__ == "__main__":
    main()