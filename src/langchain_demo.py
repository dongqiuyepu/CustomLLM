# from chatGLM_langchain import ChatGLM
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

import os

os.environ["OPENAI_API_KEY"] = getpass.getpass('OpenAI API Key:')
os.environ["SERPAPI_API_KEY"] = getpass.getpass('SERPAPI API Key:')

def agent_demo():
	from langchain.agents import load_tools
	from langchain.agents import initialize_agent
	from langchain.agents import AgentType	
	

	llm = ChatGLM()

	tools = load_tools(["serpapi", "llm-math"], llm=llm)

	# Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
	agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

	# Now let's test it out!
	agent.run("What was the high temperature in SF yesterday in Fahrenheit? What is that number raised to the .023 power?")


def index_demo():
	print("Start loading OpenAI model")
	# llm = ChatOpenAI()
	print("Finish loading OpenAI model")
	# load documents
	loader = TextLoader('./state_of_the_union.txt', encoding='utf8')
	documents = loader.load()

	# split 
	print("Start splitting text")
	text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
	texts = text_splitter.split_documents(documents)
	print("Finish loading OpenAI model")

	# embedding
	print("Start embedding")
	embeddings = OpenAIEmbeddings()
	db = Chroma.from_documents(texts, embeddings)
	print("Done embedding")

	retriever = db.as_retriever()

	qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever)

	query = "What did the president say about Ketanji Brown Jackson"
	print("Start querying")
	print(qa.run(query))
	print("Done querying")


def streamlit_demo():
	import streamlit as st
	st.title("Example app for GPT")
	data_load_state = st.text('Loading data...')
	data_load_state.text('Loading data...done!')
	if st.checkbox('Show raw data'):
		st.subheader('Raw data')

index_demo()
