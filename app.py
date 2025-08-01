from flask import Flask, render_template, request, jsonify, send_file
from rag_qdrant2 import AgenticRAG
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from processor import extract_chunks_from_manual
from qdrant_utils import store_embeddings_with_metadata
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

import os

app = Flask(__name__)
rag = AgenticRAG()
app.config['UPLOAD_FOLDER'] = 'uploads'


ALLOWED_EXTENSIONS = {'pdf', 'txt', 'docx'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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




@app.route('/')
def home():
    return render_template('storefront.html')

@app.route('/upload', methods=['GET'])
def upload_page():
    # Show the upload form
    return render_template('upload.html')

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     # Validate file
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file uploaded'}), 400
#     file = request.files['file']
#     if not file or not allowed_file(file.filename):
#         return jsonify({'error': 'Invalid file type'}), 400

#     # Save the file

#     filename = secure_filename(file.filename)
#     filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     print(filepath)
#     file.save(filepath)

#     with open(filepath, "rb") as f:
#             file_bytes = f.read()

#     chunks = extract_chunks_from_manual(file_bytes, filename)
#     store_embeddings_with_metadata(chunks)

#     return render_template('upload.html', success=True)

@app.route('/upload', methods=['POST'])
def upload_file():
    # Validate file input
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    # Validate form inputs
    category = request.form.get('category')
    brand = request.form.get('brand')
    product_id = request.form.get('product_id')

    if not category or not brand or not product_id:
        return jsonify({'error': 'Missing category, brand, or product'}), 400

    # Save uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Read file and extract chunks
    with open(filepath, "rb") as f:
        file_bytes = f.read()

    # Include metadata when extracting chunks
    chunks = extract_chunks_from_manual(
        file_bytes,
        filename=filename,
        product_id=product_id,
        category=category,
        brand=brand
    )

    # Store in Qdrant with metadata
    store_embeddings_with_metadata(chunks)

    return render_template('upload.html', success=True)

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/products.csv')
def products_csv():
    return send_file('products.csv', mimetype='text/csv')

@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Get all available product categories"""
    categories = rag.get_categories()
    return jsonify({'categories': categories})

@app.route('/api/brands/<category>', methods=['GET'])
def get_brands(category):
    """Get all brands for a specific category"""
    brands = rag.get_brands_for_category(category)
    return jsonify({'brands': brands})

@app.route('/api/products/<category>', methods=['GET'])
def get_products(category):
    """Get all products for a specific category"""
    brand = request.args.get('brand')  # Optional brand filter
    products = rag.get_products_for_category_brand(category, brand)
    return jsonify({'products': products})



@app.route('/api/chat', methods=['POST'])
def api_chat():
    try:
        data = request.get_json()
        # print("Received request",data)
        message = data.get("message")
        category = data.get("category")
        brand = data.get("brand")
        product_id = data.get("product_id")

        if not message or not category:
            return jsonify({"error": "Message and category are required"}), 400

        response = rag.generate_agentic_response(
            user_input=message,
            category=category,
            brand=brand,
            product_id=product_id
        )

        return jsonify({"response": response})
    except Exception as e:
        import traceback
        traceback.print_exc()  # 👈 will print the full error in your terminal
        return jsonify({"error": str(e)}), 500


# @app.route('/api/chat', methods=['POST'])
# def chat():
#     data = request.json

#     # Set the selected category, brand, and product in RAG system
#     rag.selected_category = data.get('category')
#     rag.selected_brand = data.get('brand')
#     rag.selected_product = data.get('product_id')

#     response = rag.generate_agentic_response(data.get('message', ''))
#     return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
