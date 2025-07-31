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
        self.device_type: Optional[str] = None  # 'phone', 'tablet', or 'computer'
        self.phone_type: Optional[str] = None   # 'iphone' or 'samsung' or None
        self.computer_type: Optional[str] = None  # 'macbook_pro' or 'windows' or None
        
        print("✅ Agentic RAG System initialized successfully!")
        print(f"✅ Session ID: {self.current_session_id}")
        print(f"✅ Qdrant cloud URL: {qdrant_url}")
        
        self._ensure_collections_exist()

    def _ensure_collections_exist(self, recreate: bool = False):
        """Ensure collections exist with proper indexes.
        
        Args:
            recreate: If True, will delete and recreate the collections.
        """
        for collection_name in [self.knowledge_collection, self.context_collection]:
            # Delete existing collection if recreate is True
            if recreate and self.qdrant_client.collection_exists(collection_name):
                print(f"♻️  Deleting existing collection: {collection_name}")
                self.qdrant_client.delete_collection(collection_name)
            
            # Create collection if it doesn't exist
            if not self.qdrant_client.collection_exists(collection_name):
                print(f"🆕 Creating new collection: {collection_name}")
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
                )
                
                # Create indexes for knowledge collection
                if collection_name == self.knowledge_collection:
                    print("🛠️  Creating indexes for knowledge collection...")
                    # Create index for device_type
                    self.qdrant_client.create_payload_index(
                        collection_name=collection_name,
                        field_name="device_type",
                        field_schema="keyword"
                    )
                    # Create index for phone_type
                    self.qdrant_client.create_payload_index(
                        collection_name=collection_name,
                        field_name="phone_type",
                        field_schema="keyword"
                    )
                    # Create index for computer_type
                    self.qdrant_client.create_payload_index(
                        collection_name=collection_name,
                        field_name="computer_type",
                        field_schema="keyword"
                    )
                    print("✅ Created indexes for device_type, phone_type, and computer_type")
            else:
                print(f"✅ Using existing collection: {collection_name}")

    def get_collection_info(self, collection_name: str) -> dict:
        """Get information about a collection including index status"""
        try:
            collection_info = self.qdrant_client.get_collection(collection_name)
            return {
                "exists": True,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors": collection_info.indexed_vectors_count,
                "payload_schema": collection_info.payload_schema
            }
        except Exception as e:
            return {"exists": False, "error": str(e)}

    def verify_collections(self):
        """Verify that collections exist and have the correct indexes"""
        print("\n🔍 Verifying collections...")
        for collection_name in [self.knowledge_collection, self.context_collection]:
            info = self.get_collection_info(collection_name)
            print(f"\nCollection: {collection_name}")
            if not info.get('exists'):
                print("  ❌ Does not exist")
                continue
                
            print(f"  ✅ Exists")
            print(f"  📊 Vectors count: {info.get('vectors_count', 0)}")
            
            if collection_name == self.knowledge_collection:
                schema = info.get('payload_schema', {})
                print("  🔍 Checking indexes:")
                for field in ['device_type', 'phone_type', 'computer_type']:
                    if field in schema:
                        print(f"    ✅ {field} index exists")
                    else:
                        print(f"    ❌ {field} index is missing")
        print("")

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
        
        filter = None
        if self.device_type:
            filter_conditions = [
                FieldCondition(
                    key="device_type",
                    match=MatchValue(
                        value=self.device_type
                    )
                )
            ]
            if self.device_type == 'phone' and self.phone_type:
                filter_conditions.append(
                    FieldCondition(
                        key="phone_type",
                        match=MatchValue(
                            value=self.phone_type
                        )
                    )
                )
            elif self.device_type == 'tablet' and self.phone_type:
                filter_conditions.append(
                    FieldCondition(
                        key="phone_type",
                        match=MatchValue(
                            value=self.phone_type
                        )
                    )
                )
            elif self.device_type == 'computer' and self.computer_type:
                filter_conditions.append(
                    FieldCondition(
                        key="computer_type",
                        match=MatchValue(
                            value=self.computer_type
                        )
                    )
                )
            filter = Filter(
                must=filter_conditions
            )
        
        if filter:
            search_result = self.qdrant_client.search(
                collection_name=self.knowledge_collection,
                query_vector=query_embedding,
                limit=limit,
                query_filter=filter
            )
        else:
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
    
    def _format_numbered_lists(self, text: str) -> str:
        """Format numbered lists to have proper line breaks between items"""
        import re
        
        # Pattern to match numbered lists (1. 2. 3. etc.)
        # This will match patterns like "1. text 2. text 3. text"
        pattern = r'(\d+\.\s+[^1-9]*?)(?=\d+\.|$)'
        
        def replace_list(match):
            # Get the matched text
            item = match.group(1).strip()
            # Add line break after each item
            return item + '\n'
        
        # Apply the formatting
        formatted_text = re.sub(pattern, replace_list, text)
        
        # Also handle patterns like "1) text 2) text 3) text"
        pattern2 = r'(\d+\)\s+[^1-9]*?)(?=\d+\)|$)'
        formatted_text = re.sub(pattern2, replace_list, formatted_text)
        
        # Clean up any extra line breaks at the end
        formatted_text = formatted_text.rstrip('\n')
        
        return formatted_text

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
        
        # Format the response to improve numbered list formatting
        formatted_response = self._format_numbered_lists(assistant_response)
        
        # Save this interaction to context
        self.save_context(user_input, formatted_response, {
            "knowledge_items_used": len(knowledge_items),
            "context_items_used": len(context_history),
            "procedure_active": bool(procedure_context)
        })
        
        return formatted_response
    
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
        
        # Add device context to the base prompt
        device_context = ""
        if self.device_type:
            device_context = f"\n\nDEVICE CONTEXT:\n- Device Type: {self.device_type.capitalize()}"
            if self.device_type == 'phone' and self.phone_type:
                device_context += f"\n- Phone Type: {self.phone_type.capitalize()}"
            elif self.device_type == 'tablet' and self.phone_type:
                device_context += f"\n- Tablet Type: {self.phone_type.capitalize()}"
            elif self.device_type == 'computer' and self.computer_type:
                device_context += f"\n- Computer Type: {self.computer_type.capitalize()}"
            elif self.device_type == 'computer' and not self.computer_type:
                device_context += f"\n- Computer Type: Unknown"
        
        base_prompt = f"""You are an intelligent diagnostic assistant with agentic capabilities. Your role is to:

1. Maintain context across conversations
2. Guide users through systematic diagnostic procedures
3. Ask relevant follow-up questions
4. Provide step-by-step troubleshooting guidance
5. Remember what has been tried and what worked/didn't work
6. Remember the user's device type and phone type (if applicable)

Key behaviors:
- Be conversational and helpful
- Ask clarifying questions when needed
- Suggest next steps based on diagnostic procedures
- Keep track of what's been tried
- Celebrate successes and learn from failures
- Never ask about device type or phone type again once it's been provided"""

        # Add device context
        base_prompt += device_context

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
