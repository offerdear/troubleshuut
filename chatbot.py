from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from qdrant_client import QdrantClient

def get_chat_response(user_query: str):
    qdrant_client = QdrantClient(host="localhost", port=6333)
    qdrant_store = Qdrant(
        client=qdrant_client,
        collection_name="product_manuals",
        embeddings=OpenAIEmbeddings()
    )

    retriever = qdrant_store.as_retriever(search_kwargs={"k": 5})
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0.3),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    result = qa_chain(user_query)
    return {
        "answer": result["result"],
        "sources": [doc.metadata for doc in result["source_documents"]]
    }
