import streamlit as st
import pymupdf
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import nltk
from dotenv import load_dotenv
from langchain import hub
import os
from langchain_openai import AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.globals import set_debug

set_debug(True)


load_dotenv()

llm = AzureChatOpenAI(
    azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
)


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
def embed_text(text, model_name="all-MiniLM-L6-v2"):
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    sentences = split_text_into_sentences(text)
    sentence_docs = []
    for sentence in sentences:
        sentence_docs.append(Document(page_content=sentence))
    return sentence_docs, embedding_model


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def main():
    st.title("Pesticide Epidemiology Question Answering Bot")

    text = extract_text_from_pdf("A - Tables 11 and 21 (less complex).pdf")

    sentence_docs, embedding_model = embed_text(text)
    print("Number of chunks: ", len(sentence_docs))
    print("Number of chunks: ", embedding_model)
    db = FAISS.from_documents(sentence_docs, embedding_model)
    print("FAISS DB INITIALISED")
    retriever = db.as_retriever()

    prompt = hub.pull("rlm/rag-prompt")

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    question = st.text_input("Ask a question")

    if st.button("Ask"):
        if question:
            print("CHAIN INVOKED")
            response = rag_chain.invoke(question)

            st.write(response)
        else:
            st.write("Please enter a question")


if __name__ == "__main__":
    main()
