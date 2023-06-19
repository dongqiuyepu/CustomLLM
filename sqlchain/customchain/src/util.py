import os
from getpass import getpass
from langchain import SQLDatabase


def openai_api_key_input():
    os.environ["OPENAI_API_KEY"] = getpass(prompt='Entry your API Key:')

def db_connector():
    db = SQLDatabase.from_uri("sqlite:///../db/sqlite/data.db")
    return db


def json_parser(str):
    s = str.replace('\n', '').replace('{', '[{', 1)[::-1].replace('}', '}]'[::-1], 1)[::-1]
    return s
