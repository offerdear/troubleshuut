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
    def __init__(self, persist_directory: str = "./qdrant_db", products_csv_path: str = "products.csv"):
        """Initialize the Agentic RAG system with Qdrant and dynamic product categories"""

        # Get OpenAI API key from environment
        openai_api_key = os.getenv("OPENAI_API_KEY")
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        print(qdrant_api_key)

        if not openai_api_key:
            raise ValueError("Missing OPENAI_API_KEY environment variable. Check your .env file.")
        # if not qdrant_url or not qdrant_api_key:
        if not qdrant_url:
            raise ValueError("Missing QDRANT_URL or QDRANT_API_KEY in environment variables.")

        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=openai_api_key)

        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

        # Collections
        self.knowledge_collection = "knowledge"
        self.context_collection = "context"

        # Load and process products data
        self.products_csv_path = products_csv_path
        self.products_df = None
        self.categories = {}
        self.brands_by_category = {}
        self.load_products_data()

        # Conversation state
        self.conversation_history: List[Dict] = []
        self.current_session_id = str(uuid.uuid4())
        self.active_procedures: Dict[str, Dict] = {}

        # Dynamic device attributes
        self.selected_category: Optional[str] = None
        self.selected_brand: Optional[str] = None
        self.selected_product: Optional[str] = None

        print("✅ Agentic RAG System initialized successfully!")
        print(f"✅ Session ID: {self.current_session_id}")
        print(f"✅ Qdrant cloud URL: {qdrant_url}")
        print(f"✅ Loaded {len(self.categories)} product categories")

        self._ensure_collections_exist()

    def load_products_data(self):
        """Load products data from CSV and extract categories and brands"""
        try:
            self.products_df = pd.read_csv(self.products_csv_path)
            print(f"✅ Loaded {len(self.products_df)} products from CSV")

            # Extract categories and their associated brands
            self.categories = {}
            self.brands_by_category = {}

            for _, row in self.products_df.iterrows():
                category = row['Category']
                brand = row['Brand']
                product_name = row['Product_Name']

                if category not in self.categories:
                    self.categories[category] = []
                    self.brands_by_category[category] = set()

                self.categories[category].append({
                    'name': product_name,
                    'brand': brand,
                    'product_id': row['Product_ID']
                })
                self.brands_by_category[category].add(brand)

            # Convert sets to sorted lists for consistency
            for category in self.brands_by_category:
                self.brands_by_category[category] = sorted(list(self.brands_by_category[category]))

            print(f"✅ Extracted categories: {list(self.categories.keys())}")

        except Exception as e:
            print(f"❌ Error loading products data: {e}")
            # Initialize empty data structures if CSV fails to load
            self.products_df = pd.DataFrame()
            self.categories = {}
            self.brands_by_category = {}

    def get_categories(self) -> List[str]:
        """Get all available product categories"""
        return list(self.categories.keys())

    def get_brands_for_category(self, category: str) -> List[str]:
        """Get all brands available for a specific category"""
        return self.brands_by_category.get(category, [])

    def get_products_for_category_brand(self, category: str, brand: str = None) -> List[Dict]:
        """Get all products for a specific category and optionally brand"""
        if category not in self.categories:
            return []

        products = self.categories[category]
        if brand:
            products = [p for p in products if p['brand'] == brand]

        return products

    def _ensure_collections_exist(self, recreate: bool = False):
        """Ensure collections exist with proper indexes for dynamic fields"""
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

                # Create dynamic indexes for knowledge collection
                if collection_name == self.knowledge_collection:
                    print("🛠️  Creating indexes for knowledge collection...")
                    # Create index for category
                    self.qdrant_client.create_payload_index(
                        collection_name=collection_name,
                        field_name="category",
                        field_schema="keyword"
                    )
                    # Create index for brand
                    self.qdrant_client.create_payload_index(
                        collection_name=collection_name,
                        field_name="brand",
                        field_schema="keyword"
                    )
                    # Create index for product_id
                    self.qdrant_client.create_payload_index(
                        collection_name=collection_name,
                        field_name="product_id",
                        field_schema="keyword"
                    )
                    print("✅ Created indexes for category, brand, and product_id")
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
                for field in ['category', 'brand', 'product_id']:
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
        """Search the knowledge base for relevant information with dynamic filtering"""

        query_embedding = self.get_embeddings([query])[0]

        filter = None
        filter_conditions = []

        # Add category filter if selected
        if self.selected_category:
            filter_conditions.append(
                FieldCondition(
                    key="category",
                    match=MatchValue(value=self.selected_category)
                )
            )

        # Add brand filter if selected
        if self.selected_brand:
            filter_conditions.append(
                FieldCondition(
                    key="brand",
                    match=MatchValue(value=self.selected_brand)
                )
            )

        # Add product filter if selected
        if self.selected_product:
            filter_conditions.append(
                FieldCondition(
                    key="product_id",
                    match=MatchValue(value=self.selected_product)
                )
            )

        if filter_conditions:
            filter = Filter(must=filter_conditions)

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

        print('knowledge',knowledge)
        return knowledge

    def _format_numbered_lists(self, text: str) -> str:
        """Format numbered lists to have proper line breaks between items"""
        import re

        # Pattern to match numbered lists (1. 2. 3. etc.)
        pattern = r'(\d+\.\s+[^1-9]*?)(?=\d+\.|$)'

        def replace_list(match):
            item = match.group(1).strip()
            return item + '\n'

        formatted_text = re.sub(pattern, replace_list, text)

        # Handle patterns like "1) text 2) text 3) text"
        pattern2 = r'(\d+\)\s+[^1-9]*?)(?=\d+\)|$)'
        formatted_text = re.sub(pattern2, replace_list, formatted_text)

        formatted_text = formatted_text.rstrip('\n')
        return formatted_text

    def generate_agentic_response(self, user_input: str) -> str:
        """Generate context-aware response with agentic behavior"""

        print(f"🔍 Processing: {user_input}")

        # Get relevant context from conversation
        context_history = self.get_relevant_context(user_input, limit=3)
        print('context_history:',context_history)

        # Search knowledge base
        knowledge_items = self.search_knowledge_base(user_input, limit=5)
        print('knowledge_items:',knowledge_items)

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
            "procedure_active": bool(procedure_context),
            "selected_category": self.selected_category,
            "selected_brand": self.selected_brand,
            "selected_product": self.selected_product
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

        # Build product context
        product_context = ""
        if self.selected_category or self.selected_brand or self.selected_product:
            product_context = f"\n\nPRODUCT CONTEXT:"
            if self.selected_category:
                product_context += f"\n- Category: {self.selected_category}"
            if self.selected_brand:
                product_context += f"\n- Brand: {self.selected_brand}"
            if self.selected_product:
                # Find product details
                if self.products_df is not None and not self.products_df.empty:
                    product_row = self.products_df[self.products_df['Product_ID'] == self.selected_product]
                    if not product_row.empty:
                        product_name = product_row.iloc[0]['Product_Name']
                        product_context += f"\n- Product: {product_name} (ID: {self.selected_product})"

        # Available categories for context
        categories_context = ""
        if self.categories:
            categories_list = ", ".join(self.categories.keys())
            categories_context = f"\n\nAVAILABLE CATEGORIES: {categories_list}"

        base_prompt = f"""You are an intelligent diagnostic assistant with agentic capabilities for various consumer products. Your role is to:

1. Maintain context across conversations
2. Guide users through systematic diagnostic procedures
3. Ask relevant follow-up questions
4. Provide step-by-step troubleshooting guidance
5. Remember what has been tried and what worked/didn't work
6. Adapt to different product categories, brands, and specific products

Key behaviors:
- Be conversational and helpful
- Ask clarifying questions when needed
- Suggest next steps based on diagnostic procedures
- Keep track of what's been tried
- Celebrate successes and learn from failures
- Adapt your expertise to the specific product category and brand
- Never ask about product category, brand, or specific product again once it's been provided"""

        # Add product context
        base_prompt += product_context
        base_prompt += categories_context

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
        print(f"  Categories loaded: {len(self.categories)}")
        print(f"  Selected category: {self.selected_category}")
        print(f"  Selected brand: {self.selected_brand}")
        print(f"  Selected product: {self.selected_product}")
