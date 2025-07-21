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
        # Filter out empty texts
        valid_texts = [text.strip() for text in texts if text and text.strip()]
        
        if not valid_texts:
            raise ValueError("No valid text content found to create embeddings")
        
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=valid_texts
            )
            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            print(f"❌ Error creating embeddings: {e}")
            print(f"📝 Text inputs: {valid_texts[:2]}...")  # Show first 2 for debugging
            raise
    
    def ingest_knowledge(self, content: str, content_name: str = None, content_type: str = "general"):
        """Ingest any type of knowledge content into the knowledge base"""
        
        if not content_name:
            content_name = f"{content_type}_content_" + str(uuid.uuid4())[:8]
        
        # Handle different content types
        if content_type == "diagnostic_procedure":
            procedure_data = self._parse_procedure(content)
            chunks = self._create_diagnostic_chunks(procedure_data, content_name)
        elif content_type == "troubleshooting_guide":
            chunks = self._parse_troubleshooting_guide(content, content_name)
        elif content_type == "faq":
            chunks = self._parse_faq(content, content_name)
        else:
            # Generic content parsing
            chunks = self._parse_generic_content(content, content_name)
        
        # Ingest chunks
        documents = []
        metadatas = []
        ids = []
        
        for chunk in chunks:
            doc_id = str(uuid.uuid4())
            documents.append(chunk['content'])
            
            metadata = {
                'type': chunk.get('type', 'general'),
                'content_name': content_name,
                'content_type': content_type,
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
        
        print(f"✅ Ingested {content_type} content '{content_name}' with {len(chunks)} chunks")
        
        return content_name
    
    def ingest_diagnostic_procedure(self, procedure_text: str, procedure_name: str = None):
        """Legacy method - use ingest_knowledge instead"""
        return self.ingest_knowledge(procedure_text, procedure_name, "diagnostic_procedure")
    
    def _create_diagnostic_chunks(self, procedure_data: Dict, content_name: str) -> List[Dict]:
        """Create chunks from parsed diagnostic procedure data"""
        chunks = []
        
        # Symptom description
        if 'symptom' in procedure_data:
            chunks.append({
                'type': 'symptom',
                'content': procedure_data['symptom'],
                'procedure': content_name
            })
        
        # Context questions
        if 'context_questions' in procedure_data:
            for question in procedure_data['context_questions']:
                chunks.append({
                    'type': 'context_question',
                    'content': question,
                    'procedure': content_name
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
                    'procedure': content_name
                })
        
        # Success indicators and resolutions
        for key in ['success_indicators', 'common_resolutions']:
            if key in procedure_data:
                chunks.append({
                    'type': key,
                    'content': "; ".join(procedure_data[key]) if isinstance(procedure_data[key], list) else str(procedure_data[key]),
                    'procedure': content_name
                })
        
        return chunks
    
    def _parse_troubleshooting_guide(self, content: str, content_name: str) -> List[Dict]:
        """Parse troubleshooting guide content"""
        chunks = []
        
        # Split by common troubleshooting patterns
        sections = re.split(r'\n\s*(?:Problem|Issue|Solution|Step|Fix)\s*[:\-]?\s*', content, flags=re.IGNORECASE)
        
        for i, section in enumerate(sections):
            if section.strip():
                chunks.append({
                    'type': 'troubleshooting_step',
                    'content': section.strip(),
                    'step_number': i
                })
        
        # If no clear structure, create chunks by paragraphs
        if len(chunks) <= 1:
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            chunks = []
            for i, paragraph in enumerate(paragraphs):
                chunks.append({
                    'type': 'troubleshooting_info',
                    'content': paragraph,
                    'paragraph_number': i
                })
        
        return chunks
    
    def _parse_faq(self, content: str, content_name: str) -> List[Dict]:
        """Parse FAQ content"""
        chunks = []
        
        # Split by Q&A patterns
        qa_pattern = r'(?:Q|Question)\s*[:\-]?\s*(.*?)(?:A|Answer)\s*[:\-]?\s*(.*?)(?=(?:Q|Question)|$)'
        matches = re.findall(qa_pattern, content, re.DOTALL | re.IGNORECASE)
        
        for i, (question, answer) in enumerate(matches):
            chunks.append({
                'type': 'faq_question',
                'content': question.strip(),
                'qa_pair': i
            })
            chunks.append({
                'type': 'faq_answer',
                'content': answer.strip(),
                'qa_pair': i
            })
        
        # If no Q&A pattern found, treat as general FAQ content
        if not chunks:
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            for i, line in enumerate(lines):
                chunks.append({
                    'type': 'faq_item',
                    'content': line,
                    'item_number': i
                })
        
        return chunks
    
    def _parse_generic_content(self, content: str, content_name: str) -> List[Dict]:
        """Parse generic content by splitting into logical chunks"""
        chunks = []
        
        # Split by paragraphs first
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        for i, paragraph in enumerate(paragraphs):
            # Further split long paragraphs by sentences
            if len(paragraph) > 500:
                sentences = re.split(r'[.!?]+', paragraph)
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk + sentence) > 300 and current_chunk:
                        chunks.append({
                            'type': 'content_chunk',
                            'content': current_chunk.strip(),
                            'chunk_number': len(chunks)
                        })
                        current_chunk = sentence
                    else:
                        current_chunk += sentence + ". "
                
                if current_chunk.strip():
                    chunks.append({
                        'type': 'content_chunk',
                        'content': current_chunk.strip(),
                        'chunk_number': len(chunks)
                    })
            else:
                chunks.append({
                    'type': 'content_chunk',
                    'content': paragraph,
                    'chunk_number': i
                })
        
        return chunks
    
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
        
        # Check if user wants to add content to the knowledge base
        add_content = input("Do you want to add content to the knowledge base? (y/n): ").lower() == 'y'
        
        if add_content:
            print("\nContent types available:")
            print("1. diagnostic_procedure - Structured diagnostic procedures")
            print("2. troubleshooting_guide - Step-by-step troubleshooting guides")
            print("3. faq - Frequently asked questions")
            print("4. general - Any other type of content")
            
            content_type_map = {
                '1': 'diagnostic_procedure',
                '2': 'troubleshooting_guide', 
                '3': 'faq',
                '4': 'general'
            }
            
            content_type_choice = input("\nSelect content type (1-4): ").strip()
            content_type = content_type_map.get(content_type_choice, 'general')
            
            content_input = input("Enter the content text (or file path): ")
            
            # Check if it's a file path
            content_text = None
            if os.path.exists(content_input):
                try:
                    with open(content_input, 'r') as f:
                        content_text = f.read()
                    print(f"✅ Successfully read file: {content_input}")
                    if not content_text.strip():
                        print("⚠️ Warning: File appears to be empty")
                        content_text = None
                except Exception as e:
                    print(f"❌ Error reading file: {e}")
                    content_text = None
            else:
                content_text = content_input
                if not content_text.strip():
                    print("⚠️ Warning: No content provided")
                    content_text = None
            
            if content_text and content_text.strip():
                content_name = input("Enter a name for this content (optional): ") or None
                rag.ingest_knowledge(content_text, content_name, content_type)
            else:
                print("❌ Skipping ingestion due to empty or invalid content")
        
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
