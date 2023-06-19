from langchain import OpenAI
from langchain.prompts.prompt import PromptTemplate


def llm_load():
    llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0, verbose=True)
    return llm


def prompt_load():
    _DEFAULT_TEMPLATE = """Given an input question, first create a syntactically correct {dialect} query to run,
    then return the results of the query in JSON format with the following keys: country,gdp,happiness_index.
    Use the following format:

    Question: "Question here"
    SQLQuery: "SQL Query to run"
    SQLResult: "Result of the SQLQuery"

    Only use the following tables:
    {table_info}

    Question: {input}"""
    prompt = PromptTemplate(
    input_variables=["input", "table_info", "dialect"], template=_DEFAULT_TEMPLATE
    )
    return prompt
