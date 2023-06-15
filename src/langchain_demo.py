
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

import os

from retrievers.milvus import MilvusRetriever


# os.environ["OPENAI_API_KEY"] = getpass.getpass('OpenAI API Key:')
os.environ["OPENAI_API_KEY"] = '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'

milvus_host = '127.0.0.1'
milvus_port = '19530'

milvus_collection_name = 'llm_demo'


def document_qa_chain():

	embedding_func = OpenAIEmbeddings()
	milvus_retriever = MilvusRetriever(
		embedding_function=embedding_func,
		collection_name=milvus_collection_name,
		connection_args={"host": milvus_host, "port": milvus_port},
	)

	return RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=milvus_retriever)
	
	
def sql_chain():
	from langchain import SQLDatabase, SQLDatabaseChain
	from langchain.chains import SQLDatabaseSequentialChain

	db = SQLDatabase.from_uri("mysql+pymysql://root:1234@localhost:3309/marketing", sample_rows_in_table_info=2)
	
	llm = OpenAI(temperature=0, verbose=True)

	db_chain = SQLDatabaseSequentialChain.from_llm(llm, db, verbose=True, use_query_checker=True, return_direct=True)
	res = db_chain("Which are the 5 happiest countries?")
	print(res)


sql_chain()

