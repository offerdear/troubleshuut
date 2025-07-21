import pandas as pd
import weaviate
from weaviate.auth import AuthApiKey
import weaviate.classes as wvc
from weaviate.classes.config import Property, DataType, Configure
from openai import OpenAI
import json
import os
from dotenv import load_dotenv
from typing import List, Dict

# Load environment variables
load_dotenv()

class RAGSystem:
    def __init__(self):
        """Initialize the RAG system with your credentials"""
        
        # Get credentials from environment
        openai_api_key = os.getenv("OPENAI_API_KEY")
        weaviate_url = os.getenv("WEAVIATE_URL")
        weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
        
        if not all([openai_api_key, weaviate_url, weaviate_api_key]):
            raise ValueError("Missing required environment variables. Check your .env file.")
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=openai_api_key)
        
        # Initialize Weaviate client with authentication
        self.weaviate_client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=AuthApiKey(weaviate_api_key)
        )
        
        # Class name for storing documents in Weaviate
        self.class_name = "Document"
        
        print("✅ RAG System initialized successfully!")
        print(f"✅ Connected to Weaviate at: {weaviate_url}")
        
    def test_connections(self):
        """Test both OpenAI and Weaviate connections"""
        try:
            # Test OpenAI
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=["test"]
            )
            print("✅ OpenAI connection successful!")
            
            # Test Weaviate
            ready = self.weaviate_client.is_ready()
            print("✅ Weaviate connection successful!")
            
            return True
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def setup_weaviate_schema(self):
        """Create the schema in Weaviate for storing documents"""
        
        # Delete existing collection if it exists
        try:
            self.weaviate_client.collections.delete(self.class_name)
            print("🗑️ Deleted existing collection")
        except:
            pass
        
        # Create the collection with v4 API syntax
        self.weaviate_client.collections.create(
            name=self.class_name,
            description="A document chunk for RAG",
            properties=[
                Property(
                    name="content",
                    data_type=DataType.TEXT,
                    description="The actual content of the document chunk"
                ),
                Property(
                    name="source", 
                    data_type=DataType.TEXT,
                    description="Source information"
                ),
                Property(
                    name="metadata",
                    data_type=DataType.TEXT, 
                    description="Additional metadata as JSON string"
                )
            ],
            # Configure vectorizer for OpenAI embeddings
            vectorizer_config=Configure.Vectorizer.none()
        )
        
        print("✅ Weaviate collection created successfully!")
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts using OpenAI"""
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [embedding.embedding for embedding in response.data]
    
    def process_csv_and_ingest(self, csv_file_path: str, text_column: str, chunk_size: int = 500):
        """Process a CSV file and ingest it into Weaviate"""
        
        # Load CSV
        df = pd.read_csv(csv_file_path)
        print(f"📊 Loaded CSV with {len(df)} rows")
        print(f"📋 Columns: {list(df.columns)}")
        
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in CSV. Available columns: {list(df.columns)}")
        
        # Prepare documents for ingestion
        documents = []
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
                
                documents.append({
                    "content": chunk,
                    "source": f"row_{idx}_chunk_{chunk_idx}",
                    "metadata": json.dumps(metadata)
                })
                
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
        
        # Ingest into Weaviate
        collection = self.weaviate_client.collections.get(self.class_name)
        with collection.batch.dynamic() as batch:
            for doc, embedding in zip(documents, all_embeddings):
                batch.add_object(
                    properties=doc,
                    vector=embedding
                )
        
        print(f"✅ Successfully ingested {len(documents)} documents into Weaviate!")
    
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
        
        # Search in Weaviate
        collection = self.weaviate_client.collections.get(self.class_name)
        response = collection.query.near_vector(
            near_vector=query_embedding,
            limit=limit,
            return_metadata=wvc.query.MetadataQuery(distance=True)
        )
        
        return [{"content": obj.properties["content"], 
                "source": obj.properties["source"], 
                "metadata": obj.properties["metadata"]} for obj in response.objects]
    
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
        """Close the Weaviate connection"""
        self.weaviate_client.close()

def main():
    rag = None
    try:
        # Initialize the RAG system
        rag = RAGSystem()
        
        # Test connections
        if not rag.test_connections():
            return
        
        # Set up Weaviate schema
        rag.setup_weaviate_schema()
        
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
    
    finally:
        # Close the connection properly
        if rag:
            rag.close()

if __name__ == "__main__":
    main()