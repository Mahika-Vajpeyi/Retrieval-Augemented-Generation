# Retrieval-Augemented-Generation

### Goal
It is widely known that Large Language Models (LLMs) are not good at answering questions from resources they have not been trained on such as internal company documents. One solution to address this information gap is to build a Retrieval Augmented Generation (RAG) system that enhances LLM capabilities with non-public resources. RAG combines the strengths of both retrieval- and generative-based artificial intelligence models by optimizing the output of a large language model to reference an authoritative knowledge base outside of its training data sources before generating a response. Since a RAG system fetches facts from external sources, its results are also more reliable and accurate. 

### Approach
###### Retrieval 
Large Language models rely on text embeddings as these embeddings retain context and relationships between words. Accordingly, the first step was to select an efficient embedding model and a vector database for efficient storage and retrieval of embeddings during search. 

###### Generation 
Since LLMs specialize in generating human-like text, the second step involved extracting information relevant to a query from the vector database and leveraging an LLM to make the final response presented to the user coherent and concise.

Note: While LlamaIndex and LangChain are frameworks to streamline RAG development, I implemented the system with as few external tools to better understand how a RAG system works and tackle some of the challenges involved myself. 
 
#### Implementation
###### Retrieval 
ChromaDB is an open-source vector database widely used for AI tasks which I used for my system as well. Since a single vector can’t adequately capture specific content within a document, I needed to chunk documents and then embed those chunks. To select the most suitable chunking method and embedding model, I implemented fixed-size chunking with and without overlap between consecutive chunks and line chunking (that is, chunks with complete sentences and a specified minimum number of words in each chunk). I then evaluated each of these three chunking methods for both the default embedding model in ChromaDB (all-MiniLM-L6-v2) and the msmarco-distilbert-tas-b model for chunks of size 100, 200, 300, and 400.  

I recorded the number of relevant chunks retrieved for two text files of varying length - the Wikipedia page for the Government of India and the novel A Tale of Two Cities made available by Project Gutenberg. I also later added functionality to parse PDF and .docx files to expand the system’s scope. Since both embedding models were comparable, I used all-MiniLM-L6-v2, the default included with ChromaDB. Also, larger chunk sizes consistently returned more relevant results possibly because larger chunks maintain context better. Therefore, I picked line chunking with a chunk size of 200.

To retrieve the most relevant chunks across all documents in the database, I retrieved the three chunks most relevant to my query from each document (stored as a collection) in the database, compared the distances calculated by the all-MiniLM-L6-v2 model and picked the set of chunks with the least distance. 

###### Generation
Since the Gemini API supports a large free tier, I used Gemini to process the three most relevant chunks retrieved and make the final response coherent and concise. I also used the following system prompt to set the context for the conversation and also maintained chat history to make the exchange between the user and assistant more conversational. 

You are a helpful and informative bot that answers questions using text from the reference passage included below. Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. However, you are talking to a non-technical audience, so be sure to break down complicated concepts and strike a friendly and conversational tone.

Your answer must only include relevant text given to you. Do not use any outside resources in your response. If the passage is irrelevant to the answer, you may ignore it. Answer only on the basis of the passage provided.

QUESTION: '{query}'
PASSAGE: '{relevant_text}'
ANSWER:

Where query refers to the user query and relevant_text refers to the most relevant chunks retrieved from the database.

I also designed an interface to accept a user query, fetch results and present them back to the user. Since Streamlit is purely in Python, requires no front-end development experience and is intuitive to use, I used Streamlit to design the app. 
