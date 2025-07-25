from flask import Flask, render_template, request, jsonify
from rag_qdrant import AgenticRAG

app = Flask(__name__)
rag = AgenticRAG()

@app.route('/')
def home():
    return render_template('chatbot.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    rag.device_type = data.get('device_type')
    rag.phone_type = data.get('device_subtype')
    response = rag.generate_agentic_response(data.get('message', ''))
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True, port=5000)