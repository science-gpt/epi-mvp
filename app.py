import streamlit as st
import pymupdf
from sentence_transformers import SentenceTransformer
import faiss
import nltk
from dotenv import load_dotenv
import cohere
import os

load_dotenv()
co = cohere.Client(os.environ["COHERE_API_KEY"])


# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def split_text_into_sentences(text):
    sentences = nltk.sent_tokenize(text)
    return sentences

# Function to embed text using sentence-transformers
def embed_text(text, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    sentences = split_text_into_sentences(text)
    embeddings = model.encode(sentences)
    return sentences, embeddings

# Function to build FAISS index
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# Function to query FAISS index
def query_faiss_index(index, query_text, model, sentences, top_k=5):
    query_embedding = model.encode([query_text])
    _, indices = index.search(query_embedding, top_k)
    results = [(sentences[i]) for i in indices[0]]
    return results


# Main function to demonstrate the process
def retrieval(pdf_path, queries):
    # Step 1: Extract text from PDF
    text = extract_text_from_pdf(pdf_path)

    # Step 2: Embed text into vectors
    sentences, embeddings = embed_text(text)

    # Step 3: Build FAISS index
    index = build_faiss_index(embeddings)

    # Step 4: Query the index
    model = SentenceTransformer('all-MiniLM-L6-v2')
    docs = []
    for query in queries:
        results = query_faiss_index(index, query, model, sentences)
        for doc in results:
            docs.append({"snippet": doc})

    # Print the results
    return docs

def generate_search_queries(query):
    queries = co.chat(
            message=query,
            search_queries_only=True
        )
    return [query.text for query in queries.search_queries]

def respond(query, pdf_path):
    search_queries = generate_search_queries(query)
    docs = retrieval(pdf_path, search_queries)
    response = co.chat(
            model="command-r",
            message=query,
            documents=docs
        )
    return response


def main():
    st.title("Pesticide Epidemiology Question Answering Bot")

    question = st.text_input("Ask a question")

    if st.button("Ask"):
        if question:
            response = respond(question, "Glyphosate.pdf")
            st.write(response.text)
        else:
            st.write("Please enter a question")

if __name__ == "__main__":
    main()