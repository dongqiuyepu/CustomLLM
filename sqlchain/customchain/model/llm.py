from langchain import OpenAI
from langchain.prompts.prompt import PromptTemplate


def llm_load():
    llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0, verbose=True)
    return llm


def prompt_load():
    _DEFAULT_TEMPLATE = """Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
    Use the following format:

    Question: "Question here"
    SQLQuery: "SQL Query to run"
    SQLResult: "Result of the SQLQuery"
    Answer: "Output a json object that contains the following keys: country,gdp,happiness_index."

    Only use the following tables:
    {table_info}

    Question: {input}"""
    prompt = PromptTemplate(
    input_variables=["input", "table_info", "dialect"], template=_DEFAULT_TEMPLATE
    )
    return prompt
