from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
import os

def build_index_from_documents(directory_path):
    """
    Build an index from documents in the specified directory.
    """
    # Load documents from the directory
    documents = SimpleDirectoryReader(directory_path).load_data()

    # Create an index from the documents
    index = VectorStoreIndex(documents)
    
    # Save the index to a file for later use
    index.save_to_disk('document_index.json')
    
    return index

directory_path = './documents'  # Path to directory containing documents
index = build_index_from_documents(directory_path)
print("Index built and saved to 'document_index.json'")

# Load the previously saved index
def load_index():
    return VectorStoreIndex.load_from_disk('document_index.json')

# Function to retrieve documents for the query
def retrieve_documents(query, index):
    """
    Retrieve documents related to the query from the index.
    """
    response = index.query(query)
    return response.response

def generate_answer(query, context):
    """
    Generate an answer using Ollama's LLM or another generative model.
    """
    ollama_model = Ollama(model="ollama3")

    prompt = f"Given the following context, answer the question:\nContext: {context}\nQuestion: {query}"
    
    response = ollama_model.chat(messages=[{"role": "system", "content": prompt}])
    return response['text']

def process_query(query):
    # Load index
    index = load_index()
    
    # Retrieve relevant documents
    retrieved_docs = retrieve_documents(query, index)
    
    # Generate answer using the model
    answer = generate_answer(query, retrieved_docs)
    
    return answer

from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest

app = Flask(__name__)

@app.route('/process_query', methods=['POST'])
def process_query_route():
    try:
        # Get the query from the user
        user_query = request.json.get('query')
        if not user_query:
            raise BadRequest("Query is required.")

        # Process the query
        answer = process_query(user_query)
        
        return jsonify({"answer": answer})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/build_index', methods=['POST'])
def build_index_route():
    try:
        # Get the directory path from the request
        directory_path = request.json.get('directory_path')
        if not directory_path:
            raise BadRequest("Directory path is required.")
        
        # Build the index
        build_index_from_documents(directory_path)
        
        return jsonify({"message": "Index built successfully."})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
