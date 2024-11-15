import streamlit as st
from rag_logic import process_query, initialize_chromadb, create_collection_for_document, check_collection
from pathlib import Path

@st.cache_resource
def get_chromadb():
    return initialize_chromadb()

def main():
    st.title("RAG")

    # Initialize ChromaDB
    client = get_chromadb()

    if not client:
        st.error("Error initializing ChromaDB connection.")
        return  # Exit if there's an error

    # Check and set up collections
    check_collection(client)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    prompt = st.chat_input("Say something")

    # Process user prompt if entered
    if prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        response = process_query(prompt, client)
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        with st.chat_message("assistant"):
            st.markdown(response)

if __name__ == "__main__":
    main()