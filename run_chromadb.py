import pandas as pd
import chromadb
from openai import OpenAI
import os
from dotenv import load_dotenv
from typing import List, Dict
import uuid

# Load environment variables
load_dotenv()

class RAGSystem:
    def __init__(self, persist_directory: str = "./chroma_db"):
        """Initialize the RAG system with ChromaDB"""
        
        # Get OpenAI API key from environment
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not openai_api_key:
            raise ValueError("Missing OPENAI_API_KEY environment variable. Check your .env file.")
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=openai_api_key)
        
        # Initialize ChromaDB client with persistence
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        
        # Collection name for storing documents
        self.collection_name = "documents"
        self.collection = None
        
        print("✅ RAG System initialized successfully!")
        print(f"✅ ChromaDB will persist data to: {persist_directory}")
        
    def test_connections(self):
        """Test OpenAI connection"""
        try:
            # Test OpenAI
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=["test"]
            )
            print("✅ OpenAI connection successful!")
            
            # Test ChromaDB
            self.chroma_client.heartbeat()
            print("✅ ChromaDB connection successful!")
            
            return True
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def setup_chromadb_collection(self):
        """Create or get the ChromaDB collection"""
        
        # Delete existing collection if it exists
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
            print("🗑️ Deleted existing collection")
        except:
            pass
        
        # Create new collection
        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        print("✅ ChromaDB collection created successfully!")
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts using OpenAI"""
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [embedding.embedding for embedding in response.data]
    
    def process_csv_and_ingest(self, csv_file_path: str, text_column: str, chunk_size: int = 500):
        """Process a CSV file and ingest it into ChromaDB"""
        
        # Load CSV
        df = pd.read_csv(csv_file_path)
        print(f"📊 Loaded CSV with {len(df)} rows")
        print(f"📋 Columns: {list(df.columns)}")
        
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in CSV. Available columns: {list(df.columns)}")
        
        # Prepare documents for ingestion
        documents = []
        metadatas = []
        ids = []
        texts_for_embedding = []
        
        for idx, row in df.iterrows():
            # Get the text content
            text_content = str(row[text_column])
            
            # Simple chunking by character count
            chunks = self._chunk_text(text_content, chunk_size)
            
            for chunk_idx, chunk in enumerate(chunks):
                # Create metadata
                metadata = {
                    "row_index": int(idx),
                    "chunk_index": chunk_idx,
                    "total_chunks": len(chunks),
                    "source": f"row_{idx}_chunk_{chunk_idx}"
                }
                
                # Add other columns as metadata
                for col in df.columns:
                    if col != text_column:
                        metadata[col] = str(row[col])
                
                # Generate unique ID
                doc_id = str(uuid.uuid4())
                
                documents.append(chunk)
                metadatas.append(metadata)
                ids.append(doc_id)
                texts_for_embedding.append(chunk)
        
        print(f"📝 Created {len(documents)} document chunks")
        
        # Get embeddings in batches
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts_for_embedding), batch_size):
            batch = texts_for_embedding[i:i+batch_size]
            embeddings = self.get_embeddings(batch)
            all_embeddings.extend(embeddings)
            print(f"🔄 Generated embeddings for batch {i//batch_size + 1}/{(len(texts_for_embedding)-1)//batch_size + 1}")
        
        # Ingest into ChromaDB
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=all_embeddings
        )
        
        print(f"✅ Successfully ingested {len(documents)} documents into ChromaDB!")
    
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
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        documents = []
        if results["documents"] and results["documents"][0]:  # Check if results exist
            for i, doc in enumerate(results["documents"][0]):
                documents.append({
                    "content": doc,
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i]
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
        
        # Show similarity scores
        for i, doc in enumerate(relevant_docs):
            print(f"  Document {i+1}: similarity = {1 - doc['distance']:.3f}")
        
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
    
    def get_collection_info(self):
        """Get information about the current collection"""
        if self.collection:
            count = self.collection.count()
            print(f"📊 Collection '{self.collection_name}' contains {count} documents")
            return count
        return 0
    
    def load_existing_collection(self):
        """Load an existing collection if it exists"""
        try:
            self.collection = self.chroma_client.get_collection(name=self.collection_name)
            count = self.collection.count()
            print(f"✅ Loaded existing collection with {count} documents")
            return True
        except:
            print("ℹ️ No existing collection found")
            return False

def main():
    rag = None
    try:
        # Initialize the RAG system
        rag = RAGSystem()
        
        # Test connections
        if not rag.test_connections():
            return
        
        # Check if there's an existing collection
        if not rag.load_existing_collection():
            # Set up new ChromaDB collection
            rag.setup_chromadb_collection()
            
            # Get CSV file info from user
            csv_file = input("Enter path to your CSV file: ")
            
            # Load CSV to show columns
            df = pd.read_csv(csv_file)
            print(f"\nAvailable columns: {list(df.columns)}")
            
            text_column = input("Enter the name of the text column to use: ")
            
            # Process and ingest CSV
            rag.process_csv_and_ingest(csv_file, text_column)
        else:
            print("Using existing collection. You can start asking questions immediately.")
        
        # Show collection info
        rag.get_collection_info()
        
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
        print("2. OPENAI_API_KEY is correct")
        print("3. Your CSV file path is correct")
        print("4. ChromaDB is installed: pip install chromadb")
    
    finally:
        # ChromaDB doesn't need explicit closing
        print("👋 Goodbye!")

if __name__ == "__main__":
    main()