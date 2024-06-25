# Pesticide Chatbot

The **Pesticide Chatbot** is an intelligent assistant designed to interact with medical literature documents. It leverages the following tools:

1. **Streamlit**: Streamlit is a Python library for creating interactive web applications. We use Streamlit to build the frontend interface for our chatbot.

2. **pymupdf**: The `pymupdf` library allows us to extract text from PDF files. In our chatbot, we use it to process medical literature documents.

3. **Sentence Transformers**: We employ Sentence Transformers to embed text into vectors. These embeddings help us compare and retrieve relevant information efficiently.

4. **FAISS (Facebook AI Similarity Search)**: FAISS is a powerful library for similarity search and clustering. We build an index using FAISS to quickly retrieve relevant sentences based on user queries.

5. **Cohere API**: The Cohere API acts as a Large Language Model (LLM) to generate context-aware responses. It enhances the chatbot's ability to provide accurate answers.

## How It Works

1. **Text Extraction**: The chatbot extracts text from medical literature PDFs using `pymupdf`.

2. **Embedding**: We embed the extracted sentences into vectors using Sentence Transformers.

3. **Indexing**: FAISS helps us build an efficient index for fast retrieval of relevant sentences.

4. **Query and Response**: When a user asks a question, the chatbot queries the index, retrieves relevant sentences, and generates a context-aware response using the Cohere API.

## Getting Started

1. Clone this repository to your local machine:

    ```bash
    git clone https://github.com/amaanirfan19/Glyphosate-chatbot.git
    ```

2. Install the necessary dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Set up your environment variables:
    - Create a `.env` file with the following keys:
        - `COHERE_API_KEY`: Your Cohere API key.

4. Run the chatbot using Streamlit:

    ```bash
    streamlit run app.py
    ```

5. Interact with the chatbot by typing medical queries or prompts related to specific topics.
6. 
