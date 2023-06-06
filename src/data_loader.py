# Load data into vector store

import os
import getpass

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Milvus
from langchain.document_loaders import TextLoader


os.environ['OPENAI_API_KEY'] = getpass.getpass('OpenAI API Key:')

document_path = '../data/state_of_the_union.txt'
splitter_chunk_size = 1000
chunk_overlap = 0

milvus_host = '127.0.0.1'
milvus_port = '19530'
milvus_collection = 'llm_demo'


def load_documents():

	# Load document
	loader = TextLoader(document_path)
	documents = loader.load()

	# Split document into chunks
	text_splitter = CharacterTextSplitter(chunk_size=splitter_chunk_size, chunk_overlap=chunk_overlap)
	docs = text_splitter.split_documents(documents)

	# Embed document chunks
	# TODO embedding to be replaced with local embedder for internal use
	embeddings = OpenAIEmbeddings()

	vector_db = Milvus.from_documents(
		docs,
		embeddings,
		collection_name=milvus_collection,
		connection_args={"host": milvus_host, "port": milvus_port},
	)


def query():
	embedding_func = OpenAIEmbeddings()
	vector_db = Milvus(
		embedding_function=embedding_func,
		collection_name=milvus_collection,
		connection_args={"host": milvus_host, "port": milvus_port},
	)

	query = "What did the president say about Ketanji Brown Jackson"
	docs = vector_db.similarity_search(query)
	print(docs[0].page_content)





if __name__ == '__main__':
	# load_documents()
	query()