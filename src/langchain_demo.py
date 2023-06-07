
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

import os

from retrievers.milvus import MilvusRetriever


# os.environ["OPENAI_API_KEY"] = getpass.getpass('OpenAI API Key:')
os.environ["OPENAI_API_KEY"] = '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'

milvus_host = '127.0.0.1'
milvus_port = '19530'

collection_name = 'llm_demo'


def document_qa_chain():

	embedding_func = OpenAIEmbeddings()
	milvus_retriever = MilvusRetriever(
		embedding_function=embedding_func,
		collection_name=collection_name,
		connection_args={"host": milvus_host, "port": milvus_port},
	)

	return RetrievalQA.from_chain_type(llm=OpenAI(temprature=0.0), chain_type="stuff", retriever=milvus_retriever)
	
	


