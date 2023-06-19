import pandas as pd
import pandasai


def json_load(json):
    df = pd.read_json(json)
    return df
def pandasai_instance_load(llm_p):
    pandas_ai = pandasai.PandasAI(llm_p)
    return pandas_ai

def pandasai_run(df,pandasai):
    res = pandasai.run(
        df,
        "Plot the histogram of countries showing for each the gpd, using different colors for each bar"
    )
    return res

