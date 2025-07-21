import pandas as pd
import chromadb
from openai import OpenAI
import os
from dotenv import load_dotenv
from typing import List, Dict, Optional, Tuple
import uuid
import json
from datetime import datetime
import re

# Load environment variables
load_dotenv()

class AgenticRAG:
    def __init__(self, persist_directory: str = "./chroma_db"):
        """Initialize the Agentic RAG system with ChromaDB"""
        
        # Get OpenAI API key from environment
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not openai_api_key:
            raise ValueError("Missing OPENAI_API_KEY environment variable. Check your .env file.")
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=openai_api_key)
        
        # Initialize ChromaDB client with persistence
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        
        # Collections
        self.knowledge_collection = None
        self.context_collection = None
        
        # Conversation state
        self.conversation_history: List[Dict] = []
        self.current_session_id = str(uuid.uuid4())
        self.active_procedures: Dict[str, Dict] = {}
        
        print("✅ Agentic RAG System initialized successfully!")
        print(f"✅ Session ID: {self.current_session_id}")
        print(f"✅ ChromaDB will persist data to: {persist_directory}")
        
    def test_connections(self):
        """Test OpenAI and ChromaDB connections"""
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
    
    def setup_collections(self):
        """Create or get ChromaDB collections"""
        
        # Knowledge collection for documents/procedures
        try:
            self.knowledge_collection = self.chroma_client.get_collection("knowledge")
            print("✅ Loaded existing knowledge collection")
        except:
            self.knowledge_collection = self.chroma_client.create_collection(
                name="knowledge",
                metadata={"hnsw:space": "cosine"}
            )
            print("✅ Created new knowledge collection")
        
        # Context collection for conversation history
        try:
            self.context_collection = self.chroma_client.get_collection("context")
            print("✅ Loaded existing context collection")
        except:
            self.context_collection = self.chroma_client.create_collection(
                name="context",
                metadata={"hnsw:space": "cosine"}
            )
            print("✅ Created new context collection")
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts using OpenAI"""
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
        metadatas = []
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
            
            metadatas.append(metadata)
            ids.append(doc_id)
        
        # Get embeddings and add to collection
        embeddings = self.get_embeddings(documents)
        
        self.knowledge_collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )
        
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
        
        # Save to ChromaDB for future retrieval
        doc_id = str(uuid.uuid4())
        embedding = self.get_embeddings([context_content])[0]
        
        self.context_collection.add(
            documents=[context_content],
            metadatas=[context_entry],
            ids=[doc_id],
            embeddings=[embedding]
        )
    
    def get_relevant_context(self, query: str, limit: int = 3) -> List[Dict]:
        """Retrieve relevant conversation context"""
        
        if self.context_collection.count() == 0:
            return []
        
        query_embedding = self.get_embeddings([query])[0]
        
        results = self.context_collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where={"session_id": self.current_session_id},
            include=["documents", "metadatas", "distances"]
        )
        
        contexts = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                contexts.append({
                    "content": doc,
                    "metadata": results["metadatas"][0][i],
                    "relevance": 1 - results["distances"][0][i]
                })
        
        return contexts
    
    def search_knowledge_base(self, query: str, limit: int = 5) -> List[Dict]:
        """Search the knowledge base for relevant information"""
        
        if self.knowledge_collection.count() == 0:
            return []
        
        query_embedding = self.get_embeddings([query])[0]
        
        results = self.knowledge_collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            include=["documents", "metadatas", "distances"]
        )
        
        knowledge = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                knowledge.append({
                    "content": doc,
                    "metadata": results["metadatas"][0][i],
                    "relevance": 1 - results["distances"][0][i]
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
        knowledge_count = self.knowledge_collection.count() if self.knowledge_collection else 0
        context_count = self.context_collection.count() if self.context_collection else 0
        
        print(f"📊 System Status:")
        print(f"  Knowledge items: {knowledge_count}")
        print(f"  Context entries: {context_count}")
        print(f"  Current session: {self.current_session_id}")
        print(f"  Conversation turns: {len(self.conversation_history)}")

def main():
    rag = None
    try:
        # Initialize the Agentic RAG system
        rag = AgenticRAG()
        
        # Test connections
        if not rag.test_connections():
            return
        
        # Setup collections
        rag.setup_collections()
        
        # Check if user wants to add a diagnostic procedure
        add_procedure = input("Do you want to add a diagnostic procedure? (y/n): ").lower() == 'y'
        
        if add_procedure:
            procedure_input = input("Enter the diagnostic procedure text (or file path): ")
            
            # Check if it's a file path
            if os.path.exists(procedure_input):
                with open(procedure_input, 'r') as f:
                    procedure_text = f.read()
            else:
                procedure_text = procedure_input
            
            procedure_name = input("Enter a name for this procedure (optional): ") or None
            rag.ingest_diagnostic_procedure(procedure_text, procedure_name)
        
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
