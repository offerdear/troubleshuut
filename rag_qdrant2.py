# agentic_rag_langchain.py

import os
import uuid
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Qdrant
from langchain_core.documents import Document

from qdrant_client import QdrantClient
from flask import Flask, request, jsonify

load_dotenv()

class AgenticRAG:
    def __init__(self, products_csv_path: str = "products.csv"):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.knowledge_collection = "electronic_manuals"
        self.context_collection = "context"

        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.3)
        self.qdrant_client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)

        self.vectorstore_knowledge = Qdrant(
            client=self.qdrant_client,
            collection_name=self.knowledge_collection,
            embeddings=self.embeddings
        )

        self.vectorstore_context = Qdrant(
            client=self.qdrant_client,
            collection_name=self.context_collection,
            embeddings=self.embeddings
        )

        self.products_csv_path = products_csv_path
        self.products_df = pd.DataFrame()
        self.categories = {}
        self.brands_by_category = {}
        self.load_products_data()

        self.current_session_id = str(uuid.uuid4())
        self.conversation_history: List[Dict] = []

        self.selected_category = None
        self.selected_brand = None
        self.selected_product = None

    def load_products_data(self):
        try:
            self.products_df = pd.read_csv(self.products_csv_path)
            for _, row in self.products_df.iterrows():
                cat, brand, name = row['Category'], row['Brand'], row['Product_Name']
                if cat not in self.categories:
                    self.categories[cat] = []
                    self.brands_by_category[cat] = set()
                self.categories[cat].append({'name': name, 'brand': brand, 'product_id': row['Product_ID']})
                self.brands_by_category[cat].add(brand)
            for cat in self.brands_by_category:
                self.brands_by_category[cat] = sorted(list(self.brands_by_category[cat]))
        except Exception as e:
            print(f"❌ Error loading products.csv: {e}")

    def get_categories(self):
        return list(self.categories.keys())

    def get_brands_for_category(self, category):
        return self.brands_by_category.get(category, [])

    def get_products_for_category_brand(self, category, brand=None):
        if category not in self.categories:
            return []
        products = self.categories[category]
        return [p for p in products if brand is None or p['brand'] == brand]

    def search_knowledge_base(self, query: str, limit: int = 5) -> List[Dict]:
        metadata_filter = {}
        if self.selected_category:
            metadata_filter["category"] = self.selected_category
        if self.selected_brand:
            metadata_filter["brand"] = self.selected_brand
        if self.selected_product:
            metadata_filter["product_id"] = self.selected_product

        retriever = self.vectorstore_knowledge.as_retriever(
            search_kwargs={"k": limit, "filter": metadata_filter or None}
        )
        # results = self.vectorstore_knowledge.similarity_search(query, k=1)

        docs = retriever.invoke(query)
        # print("docs",docs)
        return [
            {"content": d.page_content, "metadata": d.metadata}
            for d in docs if isinstance(d.page_content, str)        ]


    def get_relevant_context(self, query: str, limit: int = 3) -> List[Dict]:
        retriever = self.vectorstore_context.as_retriever(
            search_kwargs={"k": limit}
        )
        docs = retriever.invoke(query)
        return [
            {"content": d.page_content, "metadata": d.metadata}
            for d in docs
            if d.page_content is not None and
               d.metadata.get("session_id") == self.current_session_id
    ]


    def save_context(self, user_input: str, response: str, metadata: Dict = None):
    # Validate input
        if not user_input or not response:
            raise ValueError("user_input and response cannot be None or empty")

        context_entry = {
            'user_input': user_input,
            'response': response,
            'timestamp': datetime.now().isoformat(),
            # rest of your code...
        }
    # rest of your method...
    def generate_agentic_response(self, user_input: str, category=None, brand=None, product_id=None) -> str:
        self.selected_category = category
        self.selected_brand = brand
        self.selected_product = product_id

        context_history = self.get_relevant_context(user_input)
        # print("context_history",context_history)
        knowledge_items = self.search_knowledge_base(user_input)
        print("knowledge_items",knowledge_items)

        # print('self.selected_product',self.selected_product,'self.selected_brand',self.selected_brand,'self.selected_category',self.selected_category)
        prompt_parts = [
            """You are a helpful and friendly customer support assistant specializing in troubleshooting consumer electronics.

Use the following product manual excerpts and history to answer the user's question regarding the following product and brand. Focus on providing clear, concise, and actionable steps. If the answer cannot be found, admit it politely.
""",
            "\n\nRECENT CONTEXT:" + "\n".join([ctx['content'] for ctx in context_history]),
            "\n\nKNOWLEDGE:" + "\n".join([k['content'] for k in knowledge_items]),
            f"\n\nUSER INPUT: {user_input}",
            f"\n\nSELECTED CATEGORY: {self.selected_category}",
            f"\n\nSELECTED BRAND: {self.selected_brand}",
            f"\n\nSELECTED PRODUCT: {self.selected_product}"
        ]

        prompt = "\n".join(prompt_parts)
        response = self.llm.invoke(prompt)

        self.save_context(user_input, response.content, {
            "knowledge_items_used": len(knowledge_items),
            "context_items_used": len(context_history),
            "selected_category": self.selected_category,
            "selected_brand": self.selected_brand,
            "selected_product": self.selected_product
        })

        return response.content
    def clean_none_content_documents(self, batch_size: int = 100) -> int:
        """
        Remove documents with None or empty content from the context collection.
        Returns the number of documents removed.
        """
        from qdrant_client.http import models as rest

        # First, find all document IDs with None or empty content
        scroll_filter = rest.Filter(
            must_not=[
                rest.FieldCondition(
                    key="metadata.text",  # Adjust this key if your content is stored differently
                    is_null=True
            )
        ]
    )

        # Scroll through all points to find ones with None content
        points = self.qdrant_client.scroll(
            collection_name=self.context_collection,
            scroll_filter=scroll_filter,
            limit=batch_size,
            with_vectors=False,
        with_payload=True
    )

        # Collect IDs of documents to delete
        ids_to_delete = []
        for point in points[0]:  # points is a tuple of (points, offset)
            if not point.payload.get("text"):  # Adjust this key if your content is stored differently
                ids_to_delete.append(point.id)

        # Delete the documents in batches
        if ids_to_delete:
            self.qdrant_client.delete(
                collection_name=self.context_collection,
                points_selector=ids_to_delete
            )

        return len(ids_to_delete)
