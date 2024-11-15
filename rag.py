import chromadb
import numpy as np
from dotenv import load_dotenv
import os
import google.generativeai as genai
import time
import streamlit as st
from pypdf import PdfReader
from docx import Document
from chromadb.utils import embedding_functions
import nltk

# Define the NLTK data path
# nltk_data_path = os.path.join(os.path.expanduser("~"), "Desktop/Internship-2024/Ericsson/app/nltk_data")
# os.makedirs(nltk_data_path, exist_ok=True)
# nltk.data.path.append(nltk_data_path)
# nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
# nltk.download('punkt')

load_dotenv()

# Initialize ChromaDB client
# @st.cache_resource
def initialize_chromadb():
    client = chromadb.PersistentClient(path=".")
    embedding_functions.SentenceTransformerEmbeddingFunction('all-MiniLM-L6-v2')
    return client

def check_collection(client):
    collections = client.list_collections()
    print(collections)
    for filename in os.listdir("./data"):
        collection_name = f"collection_{filename}"
        exists = any(col.name == collection_name for col in collections)
        if not exists:
            file_path = os.path.join("./data/", filename)
            create_collection_for_document(client, filename, file_path)

# Create a collection for each document
def create_collection_for_document(client, filename, file_path="./data"):
    chunks = get_document_text(file_path)
    collection_name = f"collection_{filename}"
    collection = client.create_collection(collection_name)
    add_chunks_to_collection(chunks, collection, filename)

# Load and chunk the document
def get_document_text(file_path):
    chunks = []
    if os.path.isfile(file_path):
        lines = []
        if file_path.lower().endswith('pdf'):
            pdf_reader = PdfReader(file_path)
            text = "".join(page.extract_text() for page in pdf_reader.pages)
            lines = text.splitlines()
        elif file_path.lower().endswith('docx'):
            doc = Document(file_path)
            text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
            lines = text.splitlines()
        else:
            with open(file_path, "r") as file:
                lines = file.readlines()
        
        chunk = ''
        for i, line in enumerate(lines):
            chunk += line
            words_in_chunk = len(chunk.split())
            if words_in_chunk > 200 or i == len(lines) - 1:
                chunks.append(chunk)
                chunk = ''
    return chunks

def tokenize(text, size=5):
    sentences = nltk.sent_tokenize(text)
    return [" ".join(sentences[i: i+size]) for i in range(0, len(sentences), size)]

# Add chunks to the ChromaDB collection
def add_chunks_to_collection(chunks, collection, filename):
    for count, chunk in enumerate(chunks):
        collection.add(
            documents=chunk,
            metadatas=[{"source": filename}],
            ids=[str(count)]
        )

if "model" not in st.session_state:
    print("instantiating model")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=gemini_api_key)
    st.session_state.model = genai.GenerativeModel('gemini-pro')
    st.session_state.chat = st.session_state.model.start_chat(history=[])

# Generate response from generative AI model
def generate_response(prompt, query, retries=2):
    answer = st.session_state.chat.send_message(prompt)
    return answer.text

# Process a query
def process_query(query, client):
    collections = client.list_collections()
    best_relevant_passage = None
    best_score = float('inf')

    for col in collections:
        collection = client.get_collection(col.name)
        relevant_passage = collection.query(query_texts=query, n_results=3)

        if relevant_passage.get('documents'):
            dist = relevant_passage.get('distances')[0]
            if dist[0] < best_score:
                best_score = dist[0]
                docs = relevant_passage['documents'][0]
                best_relevant_passage = " ".join(doc for doc in docs)
    
    if not best_relevant_passage:
        return "No relevant information found in the reference passage."

    escaped = best_relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = (
        f"You are a helpful and informative bot that answers questions using text from the reference passage included below. "
        f"Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. "
        f"However, you are talking to a non-technical audience, so be sure to break down complicated concepts and strike a friendly and conversational tone. "
        f"Your answer must only include relevant text given to you. Do not use any outside resources in your response. "
        f"QUESTION: '{query}' PASSAGE: '{escaped}' ANSWER: "
    )
    response = generate_response(prompt, query)
    return response