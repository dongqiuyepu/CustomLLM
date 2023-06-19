import pandasai
from pandasai.llm.openai import OpenAI


def llm_p_load():
    llm_p = pandasai.llm.openai.OpenAI(model_name="gpt-3.5-turbo", temperature=0, verbose=True)
    return llm_p