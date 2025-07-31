from flask import Flask, render_template, request, jsonify, send_file
from rag_qdrant import AgenticRAG
import os

app = Flask(__name__)
rag = AgenticRAG()

@app.route('/')
def home():
    return render_template('storefront.html')

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
def chat():
    data = request.json
    
    # Set the selected category, brand, and product in RAG system
    rag.selected_category = data.get('category')
    rag.selected_brand = data.get('brand') 
    rag.selected_product = data.get('product_id')
    
    response = rag.generate_agentic_response(data.get('message', ''))
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True, port=5000)