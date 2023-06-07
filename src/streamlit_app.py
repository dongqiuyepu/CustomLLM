import streamlit as st

from langchain_demo import document_qa_chain
from data_loader import load_documents_from_file_upload


def main_func():
    st.title("Customized LLM App")

    document = st.file_uploader("Upload your .txt document for QA", type=["txt"])
    if document:
        with st.spinner("Indexing your document..."):
            load_documents_from_file_upload(document)

    question = st.text_input("Enter your question here...")

    if question != "":
        with st.spinner("Answering your question. Please hold..."):
            qa = document_qa_chain()
            ans = qa.run(question)
            st.write(ans)

main_func()