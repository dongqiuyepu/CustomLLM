import os

from dotenv import load_dotenv
from langchain import LLMChain, OpenAI, SQLDatabase, SQLDatabaseChain

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
db = SQLDatabase.from_uri(DATABASE_URL)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

# res = llm.predict("Hi")


sql_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)


from langchain.callbacks import get_openai_callback
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

# query = "How many employees are also customers?"

# with get_openai_callback() as cb:
#     result = sql_chain.run(query)
#     print(result)
#     print(cb)


loader = TextLoader("data/state_of_the_union.txt", encoding="utf-8")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()

from langchain.vectorstores import Chroma, Milvus

vectordb = Milvus.from_documents(
    texts,
    embeddings,
    connection_args={"host": "192.168.2.201", "port": "19530"},
)

doc_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(), chain_type="stuff", retriever=vectordb.as_retriever()
)
doc_chain.verbose = True

# query = "What did the president say about Ketanji Brown Jackson?"

# with get_openai_callback() as cb:
#     result = doc_chain.run(query)
#     print(result)
#     print(cb)


from langchain.agents import AgentType, Tool, initialize_agent, load_tools

tools = []

tools.append(Tool(name="sql chain", func=sql_chain.run, description="sql chain"))
tools.append(Tool(name="doc chain", func=doc_chain.run, description="doc chain"))

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# query = "What did the president say about Ketanji Brown Jackson?"
# query = "Which are the 5 happiest countries?"

# with get_openai_callback() as cb:
#     result = agent.run(query)
#     print(result)
#     print(cb)

import gradio as gr
from loguru import logger


async def make_completion(query):
    with get_openai_callback() as cb:
        res = agent.run(query)
        print(res)
        print(cb)

    return str(res)


async def predict(input, history):
    """
    Predict the response of the chatbot and complete a running list of chat history.
    """
    history.append(input)
    response = await make_completion(history[-1])
    history.append(response)
    messages = [(history[i], history[i + 1]) for i in range(0, len(history) - 1, 2)]
    return messages, history


css = """
.contain {margin-top: 80px;}
.title > div {font-size: 24px !important}
"""

with gr.Blocks(css=css) as demo:
    logger.info("Starting Demo...")
    chatbot = gr.Chatbot(label="Chatbot", elem_classes="title")
    state = gr.State([])
    with gr.Row():
        txt = gr.Textbox(
            show_label=False, placeholder="Enter text and press enter"
        ).style(container=False)
    txt.submit(predict, [txt, state], [chatbot, state])

demo.launch(server_port=18080, share=True, server_name="0.0.0.0")
