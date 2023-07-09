import streamlit as st

from langchain_demo import document_qa_chain, sql_chain, sql_chain_with_plot
from data_loader import load_documents_from_file_upload


@st.cache_data
def upload_file_contents(document):
    load_documents_from_file_upload(document)


def Streamlit_DocQA():
    st.title("Customized LLM App For Document Q&A")

    document = st.file_uploader("Upload your .txt document for QA", type=["txt"])
    if document:
        with st.spinner("Indexing your document..."):
            upload_file_contents(document)

    question = st.text_input("Enter your question here...")

    if question != "":
        with st.spinner("Answering your question. Please hold..."):
            qa = document_qa_chain()
            ans = qa.run(question)
            st.write(ans)


def Streamlit_SqlChain():
    st.title("Customized LLM App For Database Querying")

    question = st.text_input("Enter your question here...")

    if question != "":
        with st.spinner("Answering your question. Please hold..."):
            ans = sql_chain(question)
            st.write(ans)


def Streamlit_SqlChain_Plot():
    st.title("Customized LLM App For Generating Plots from DB")

    question = st.text_input("Enter your request here...")

    if question != "":
        with st.spinner("Generating plots. Please hold..."):
            sql_chain_with_plot(question)


# Streamlit_DocQA()
# Streamlit_SqlChain()
Streamlit_SqlChain_Plot()