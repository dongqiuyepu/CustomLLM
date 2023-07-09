from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

import pandas as pd
from pandasai import PandasAI

import os

from retrievers.milvus import MilvusRetriever
from custom_chains.SQLDBChain import SQLDBChain
from langchain.prompts.prompt import PromptTemplate

_mysql_prompt = """You are a data analyst. Given an input question, perform the following task:
1. create a syntactically correct MySQL query to run
2. then look at the results of the query and generate a json string with column names as key

Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in backticks (`) to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table. The final answer should only include json string with no additional text.

Use the following format:

Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Json string with column names key here

Only use the following tables:
{table_info}

Let's look at an example first:

Question: Plot the histogram of countries showing for each the happiness index, using different colors for each bar.
SQLQuery: SELECT country, happiness_index FROM country_gdp
SQLResult: [('United States', 4.2), ('United Kingdom', 4.3), ('France', 4.4), ('Germany', 5.4), ('Italy', 1.2), ('Spain', 4.3), ('Canada', 2.3), ('Australia', 3.4), ('Japan', 4.7), ('China', 1.2)]
Answer: {{
	"country": ["United States", "United Kingdom", "France", "Germany", "Italy", "Spain", "Canada", "Australia", "Japan", "China"],
    "happiness_index": [4.2, 4.3, 4.4, 5.4, 1.2, 4.3, 2.3, 3.4, 4.7, 1.2]
}}

Now let's start the task:

Question: {input}
"""

CUSTOM_PROMPT = PromptTemplate(
    input_variables=["input", "table_info"],
    template=_mysql_prompt,
)


# os.environ["OPENAI_API_KEY"] = getpass.getpass('OpenAI API Key:')
os.environ["OPENAI_API_KEY"] = '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'

milvus_host = '127.0.0.1'
milvus_port = '19530'

milvus_collection_name = 'llm_demo'

chart_save_path = '/Users/dongqiuyepu/Desktop/charts/'


def document_qa_chain():

	embedding_func = OpenAIEmbeddings()
	milvus_retriever = MilvusRetriever(
		embedding_function=embedding_func,
		collection_name=milvus_collection_name,
		connection_args={"host": milvus_host, "port": milvus_port},
	)

	return RetrievalQA.from_chain_type(llm=OpenAI(model_name="gpt-3.5-turbo", temperature=0.7, verbose=True), chain_type="stuff", retriever=milvus_retriever)
	
	
def sql_chain(question):

	###################
	# Get data from DB
	###################

	from langchain import SQLDatabase, SQLDatabaseChain
	from langchain.chains import SQLDatabaseSequentialChain

	db = SQLDatabase.from_uri("mysql+pymysql://root:1234@localhost:3309/marketing", sample_rows_in_table_info=2)
	
	llm_for_sqlchain = OpenAI(temperature=0, verbose=True)

	db_chain = SQLDatabaseChain.from_llm(llm_for_sqlchain, db, verbose=True, use_query_checker=False, return_direct=False)

	res = db_chain(question)['result']
	# res = db_chain(question)

	return res


def sql_chain_with_plot(question):
	###################
	# Get data from DB
	###################
	from langchain import SQLDatabase, SQLDatabaseChain

	db = SQLDatabase.from_uri("mysql+pymysql://root:1234@localhost:3309/marketing", sample_rows_in_table_info=2)
	
	llm_for_sqlchain = OpenAI(model_name="gpt-3.5-turbo", temperature=0, verbose=True)

	db_chain = SQLDBChain.from_llm(llm_for_sqlchain, db, prompt=CUSTOM_PROMPT, verbose=True, use_query_checker=False, return_direct=False)

	res = db_chain(question)['result']

	#########################################
	# Use data to answer question: pandasAI
	#########################################
	from pandasai.llm.openai import OpenAI as OA_pandas
	
	df = pd.DataFrame(dict(eval(res)))
	llm = OA_pandas(api_token=os.environ["OPENAI_API_KEY"])

	pandas_ai = PandasAI(llm=llm, save_charts=True)
	pandas_ai(df, prompt=question)


# sql_chain_with_plot("Plot the histogram of companies showing for average salary, using different colors for each bar.")
# sql_chain("Give me all software engineers in table format")
# sql_chain_with_plot("Generate data that can be used for following request: Plot the histogram of countries showing for each the gdp, using different colors for each bar. The answer should be in json format that include all column names.")

