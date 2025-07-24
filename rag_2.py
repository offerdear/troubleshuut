import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from openai import OpenAI
import os
from dotenv import load_dotenv
from typing import List, Dict, Optional, Tuple
import uuid
import json
from datetime import datetime
import re
import sys
from PyPDF2 import PdfReader

# Load environment variables
load_dotenv()

class AgenticRAG:
    def __init__(self, persist_directory: str = "./qdrant_db"):
        """Initialize the Agentic RAG system with Qdrant"""
        
        # Get OpenAI API key from environment
        openai_api_key = os.getenv("OPENAI_API_KEY")
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        if not openai_api_key:
            raise ValueError("Missing OPENAI_API_KEY environment variable. Check your .env file.")
        if not qdrant_url or not qdrant_api_key:
            raise ValueError("Missing QDRANT_URL or QDRANT_API_KEY in environment variables.")
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=openai_api_key)
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        
        # Collections
        self.knowledge_collection = "knowledge"
        self.context_collection = "context"
        
        # Conversation state
        self.conversation_history: List[Dict] = []
        self.current_session_id = str(uuid.uuid4())
        self.active_procedures: Dict[str, Dict] = {}
        
        print("✅ Agentic RAG System initialized successfully!")
        print(f"✅ Session ID: {self.current_session_id}")
        print(f"✅ Qdrant cloud URL: {qdrant_url}")
        
        self._ensure_collections_exist()

    def _ensure_collections_exist(self):
        # Create collections if they don't exist
        for collection_name in [self.knowledge_collection, self.context_collection]:
            if not self.qdrant_client.collection_exists(collection_name):
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
                )
                print(f"✅ Created Qdrant collection: {collection_name}")
            else:
                print(f"✅ Qdrant collection exists: {collection_name}")

    def test_connections(self):
        """Test OpenAI and Qdrant connections"""
        try:
            # Test OpenAI
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=["test"]
            )
            print("✅ OpenAI connection successful!")
            # Test Qdrant
            self.qdrant_client.get_collections()
            print("✅ Qdrant connection successful!")
            return True
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [embedding.embedding for embedding in response.data]

    def ingest_diagnostic_procedure(self, procedure_text: str, procedure_name: str = None):
        """Parse and ingest a diagnostic procedure into the knowledge base"""
        if not procedure_name:
            procedure_name = "diagnostic_procedure_" + str(uuid.uuid4())[:8]
        
        # Parse procedure structure
        procedure_data = self._parse_procedure(procedure_text)
        
        # Create chunks for different parts of the procedure
        chunks = []
        
        # Symptom description
        if 'symptom' in procedure_data:
            chunks.append({
                'type': 'symptom',
                'content': procedure_data['symptom'],
                'procedure': procedure_name
            })
        
        # Context questions
        if 'context_questions' in procedure_data:
            for question in procedure_data['context_questions']:
                chunks.append({
                    'type': 'context_question',
                    'content': question,
                    'procedure': procedure_name
                })
        
        # Diagnostic steps
        if 'diagnostic_flow' in procedure_data:
            for step_name, step_content in procedure_data['diagnostic_flow'].items():
                if isinstance(step_content, list):
                    step_text = f"{step_name}: " + "; ".join(step_content)
                else:
                    step_text = f"{step_name}: {step_content}"
                chunks.append({
                    'type': 'diagnostic_step',
                    'content': step_text,
                    'step_name': step_name,
                    'procedure': procedure_name
                })
        
        # Success indicators and resolutions
        for key in ['success_indicators', 'common_resolutions']:
            if key in procedure_data:
                chunks.append({
                    'type': key,
                    'content': "; ".join(procedure_data[key]) if isinstance(procedure_data[key], list) else str(procedure_data[key]),
                    'procedure': procedure_name
                })
        
        # Ingest chunks
        documents = []
        payloads = []
        ids = []
        
        for chunk in chunks:
            doc_id = str(uuid.uuid4())
            documents.append(chunk['content'])
            
            metadata = {
                'type': chunk['type'],
                'procedure': procedure_name,
                'timestamp': datetime.now().isoformat()
            }
            metadata.update({k: v for k, v in chunk.items() if k not in ['content', 'type']})
            
            payloads.append(metadata)
            ids.append(doc_id)
        
        # Get embeddings and add to collection
        embeddings = self.get_embeddings(documents)
        
        points = [PointStruct(id=ids[i], vector=embeddings[i], payload=payloads[i]) for i in range(len(ids))]
        self.qdrant_client.upsert(collection_name=self.knowledge_collection, points=points)
        
        print(f"✅ Ingested procedure '{procedure_name}' with {len(chunks)} chunks")
        
        return procedure_name
    
    def _parse_procedure(self, text: str) -> Dict:
        """Parse diagnostic procedure text into structured data"""
        procedure = {}
        
        # Extract symptom
        symptom_match = re.search(r'Symptom:\s*(.+?)(?=\n|\r|$)', text)
        if symptom_match:
            procedure['symptom'] = symptom_match.group(1).strip()
        
        # Extract context questions
        context_section = re.search(r'Context_Questions:\s*(.*?)(?=Diagnostic_Flow:|$)', text, re.DOTALL)
        if context_section:
            questions = re.findall(r'-\s*([^-\n]+)', context_section.group(1))
            procedure['context_questions'] = [q.strip() for q in questions]
        
        # Extract diagnostic flow
        flow_section = re.search(r'Diagnostic_Flow:\s*(.*?)(?=Success_Indicators:|Common_Resolutions:|$)', text, re.DOTALL)
        if flow_section:
            flow = {}
            current_step = None
            for line in flow_section.group(1).split('\n'):
                line = line.strip()
                if re.match(r'^\d+\.\s+(\w+):', line):
                    step_match = re.match(r'^\d+\.\s+(\w+):\s*(.*)', line)
                    if step_match:
                        current_step = step_match.group(1)
                        flow[current_step] = [step_match.group(2)] if step_match.group(2) else []
                elif line.startswith('- ') and current_step:
                    flow[current_step].append(line[2:])
            procedure['diagnostic_flow'] = flow
        
        # Extract success indicators
        success_match = re.search(r'Success_Indicators:\s*\[(.*?)\]', text, re.DOTALL)
        if success_match:
            indicators = [s.strip().strip('"') for s in success_match.group(1).split(',')]
            procedure['success_indicators'] = indicators
        
        # Extract common resolutions
        resolution_match = re.search(r'Common_Resolutions:\s*\[(.*?)\]', text, re.DOTALL)
        if resolution_match:
            resolutions = [s.strip().strip('"') for s in resolution_match.group(1).split(',')]
            procedure['common_resolutions'] = resolutions
        
        return procedure
    
    def save_context(self, user_input: str, response: str, metadata: Dict = None):
        """Save conversation context to persistent storage"""
        
        context_entry = {
            'user_input': user_input,
            'response': response,
            'timestamp': datetime.now().isoformat(),
            'session_id': self.current_session_id
        }
        
        if metadata:
            context_entry.update(metadata)
        
        # Add to conversation history
        self.conversation_history.append(context_entry)
        
        # Create searchable context content
        context_content = f"User: {user_input}\nAssistant: {response}"
        
        # Save to Qdrant for future retrieval
        doc_id = str(uuid.uuid4())
        embedding = self.get_embeddings([context_content])[0]
        
        point = PointStruct(id=doc_id, vector=embedding, payload=context_entry)
        self.qdrant_client.upsert(collection_name=self.context_collection, points=[point])
    
    def get_relevant_context(self, query: str, limit: int = 3) -> List[Dict]:
        """Retrieve relevant conversation context"""
        
        query_embedding = self.get_embeddings([query])[0]
        
        search_result = self.qdrant_client.search(
            collection_name=self.context_collection,
            query_vector=query_embedding,
            limit=limit
        )
        contexts = []
        for hit in search_result:
            # Filter by session_id in Python
            if hit.payload.get("session_id") == self.current_session_id:
                contexts.append({
                    "content": hit.payload.get("user_input", "") + "\n" + hit.payload.get("response", ""),
                    "metadata": hit.payload,
                    "relevance": hit.score
                })
        return contexts
    
    def search_knowledge_base(self, query: str, limit: int = 5) -> List[Dict]:
        """Search the knowledge base for relevant information"""
        
        query_embedding = self.get_embeddings([query])[0]
        
        search_result = self.qdrant_client.search(
            collection_name=self.knowledge_collection,
            query_vector=query_embedding,
            limit=limit
        )
        
        knowledge = []
        for hit in search_result:
            knowledge.append({
                "content": hit.payload.get("content", ""),
                "metadata": hit.payload,
                "relevance": hit.score
            })
        
        return knowledge
    
    def generate_agentic_response(self, user_input: str) -> str:
        """Generate context-aware response with agentic behavior"""
        
        print(f"🔍 Processing: {user_input}")
        
        # Get relevant context from conversation
        context_history = self.get_relevant_context(user_input, limit=3)
        
        # Search knowledge base
        knowledge_items = self.search_knowledge_base(user_input, limit=5)
        
        # Determine if this is part of a diagnostic procedure
        procedure_context = self._analyze_procedure_state(user_input, knowledge_items)
        
        # Build prompt with context
        system_prompt = self._build_system_prompt(procedure_context)
        user_prompt = self._build_user_prompt(user_input, context_history, knowledge_items)
        
        # Generate response
        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=800,
            temperature=0.3
        )
        
        assistant_response = response.choices[0].message.content
        
        # Save this interaction to context
        self.save_context(user_input, assistant_response, {
            "knowledge_items_used": len(knowledge_items),
            "context_items_used": len(context_history),
            "procedure_active": bool(procedure_context)
        })
        
        return assistant_response
    
    def _analyze_procedure_state(self, user_input: str, knowledge_items: List[Dict]) -> Optional[Dict]:
        """Analyze if user is in the middle of a diagnostic procedure"""
        
        # Look for diagnostic-related keywords in knowledge items
        diagnostic_items = [item for item in knowledge_items 
                          if item['metadata'].get('type') in ['diagnostic_step', 'context_question', 'symptom']]
        
        if diagnostic_items:
            # Check if there's an active procedure
            procedures = set(item['metadata'].get('procedure') for item in diagnostic_items)
            if procedures:
                return {
                    'active_procedures': list(procedures),
                    'current_step_type': diagnostic_items[0]['metadata'].get('type'),
                    'diagnostic_items': diagnostic_items
                }
        
        return None
    
    def _build_system_prompt(self, procedure_context: Optional[Dict]) -> str:
        """Build system prompt based on current context"""
        
        base_prompt = """You are an intelligent diagnostic assistant with agentic capabilities. Your role is to:

1. Maintain context across conversations
2. Guide users through systematic diagnostic procedures
3. Ask relevant follow-up questions
4. Provide step-by-step troubleshooting guidance
5. Remember what has been tried and what worked/didn't work

Key behaviors:
- Be conversational and helpful
- Ask clarifying questions when needed
- Suggest next steps based on diagnostic procedures
- Keep track of what's been tried
- Celebrate successes and learn from failures"""

        if procedure_context:
            base_prompt += f"""

ACTIVE DIAGNOSTIC CONTEXT:
- Currently working on: {', '.join(procedure_context['active_procedures'])}
- Current step type: {procedure_context['current_step_type']}
- Follow the diagnostic flow systematically
- Ask context questions before proceeding with steps
- Guide the user through each verification step"""

        return base_prompt
    
    def _build_user_prompt(self, user_input: str, context_history: List[Dict], knowledge_items: List[Dict]) -> str:
        """Build user prompt with relevant context and knowledge"""
        
        prompt_parts = []
        
        # Add conversation context
        if context_history:
            prompt_parts.append("RECENT CONVERSATION CONTEXT:")
            for ctx in context_history[-2:]:  # Last 2 relevant exchanges
                prompt_parts.append(f"- {ctx['content'][:200]}...")
            prompt_parts.append("")
        
        # Add relevant knowledge
        if knowledge_items:
            prompt_parts.append("RELEVANT KNOWLEDGE:")
            for item in knowledge_items:
                item_type = item['metadata'].get('type', 'unknown')
                content = item['content'][:300]
                prompt_parts.append(f"[{item_type.upper()}] {content}")
            prompt_parts.append("")
        
        # Add current user input
        prompt_parts.append(f"CURRENT USER INPUT: {user_input}")
        prompt_parts.append("")
        prompt_parts.append("Please provide a helpful, context-aware response:")
        
        return "\n".join(prompt_parts)
    
    def get_system_status(self):
        """Get current system status"""
        knowledge_count = self.qdrant_client.count(self.knowledge_collection) if self.qdrant_client else 0
        context_count = self.qdrant_client.count(self.context_collection) if self.qdrant_client else 0
        
        print(f"📊 System Status:")
        print(f"  Knowledge items: {knowledge_count}")
        print(f"  Context entries: {context_count}")
        print(f"  Current session: {self.current_session_id}")
        print(f"  Conversation turns: {len(self.conversation_history)}")

    def ingest_text_chunks(self, chunks: list, source: str = "pdf"):
        """Ingest a list of text chunks into the knowledge base (Qdrant collection)."""
        documents = []
        payloads = []
        ids = []
        for i, chunk in enumerate(chunks):
            doc_id = str(uuid.uuid4())
            documents.append(chunk)
            metadata = {
                'type': 'text_chunk',
                'source': source,
                'chunk_index': i,
                'timestamp': datetime.now().isoformat()
            }
            payloads.append(metadata)
            ids.append(doc_id)
        embeddings = self.get_embeddings(documents)
        points = [PointStruct(id=ids[i], vector=embeddings[i], payload=payloads[i]) for i in range(len(ids))]
        self.qdrant_client.upsert(collection_name=self.knowledge_collection, points=points)
        print(f"✅ Ingested {len(chunks)} text chunks into the knowledge base from {source}.")

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def split_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def main():
    rag = None
    try:
        # PDF ingestion mode
        if len(sys.argv) == 3 and sys.argv[1] == "ingest_pdf":
            pdf_path = sys.argv[2]
            print(f"Extracting text from {pdf_path}...")
            text = extract_text_from_pdf(pdf_path)
            print(f"Extracted {len(text)} characters. Splitting into chunks...")
            chunks = split_text(text)
            print(f"Split into {len(chunks)} chunks. Ingesting into Qdrant...")
            rag = AgenticRAG()
            rag.ingest_text_chunks(chunks, source=pdf_path)
            print("Done.")
            return
        # Initialize the Agentic RAG system
        rag = AgenticRAG()
        
        # Test connections
        if not rag.test_connections():
            return
        
        # Show system status
        rag.get_system_status()
        
        print("\n🎉 Agentic RAG system is ready!")
        print("This system maintains conversation context and provides guided diagnostics.")
        print("Type 'quit' to exit, 'status' for system info.\n")
        
        # Interactive conversation loop
        while True:
            user_input = input("💬 You: ")
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            elif user_input.lower() == 'status':
                rag.get_system_status()
                continue
            
            response = rag.generate_agentic_response(user_input)
            print(f"🤖 Assistant: {response}\n")
            print("-" * 60)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("👋 Goodbye!")

if __name__ == "__main__":
    main()
