from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from processor import extract_chunks_from_manual
from qdrant_utils import store_embeddings_with_metadata
from chatbot import get_chat_response
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from qdrant_client import QdrantClient

app = Flask(__name__)

CORS(app)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# existing prompt template
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful and friendly customer support assistant specializing in troubleshooting consumer electronics.

Use the following product manual excerpts to answer the user's question. Focus on providing clear, concise, and actionable steps. If the answer cannot be found, admit it politely.

========
{context}
========

Customer question: {question}

Your helpful troubleshooting answer:
""".strip()
)



@app.route("/upload", methods=["POST"])
def upload_batch():
    if 'files' not in request.files:
        return jsonify({"error": "No files part"}), 400

    files = request.files.getlist("files")
    product_id = request.form.get("product_id")  # optional per-batch tag

    all_chunks = []
    for file in files:
        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        with open(filepath, "rb") as f:
            file_bytes = f.read()

        chunks = extract_chunks_from_manual(file_bytes, filename, product_id)
        all_chunks.extend(chunks)

    store_embeddings_with_metadata(all_chunks)
    return jsonify({"status": "success", "chunks_stored": len(all_chunks)})


@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("query", "")  # <-- match frontend field

    if not user_input:
        return jsonify({"error": "No question provided"}), 400

    # Connect to Qdrant
    qdrant = Qdrant(
        client=QdrantClient(host="localhost", port=6333),
        collection_name="product_manuals",
        embeddings=OpenAIEmbeddings()
    )

    # Build RAG chain
    retriever = qdrant.as_retriever(search_kwargs={"k": 5})
    llm = ChatOpenAI(temperature=0.3)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    result = qa_chain(user_input)
    return jsonify({
        "answer": result["result"]
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
