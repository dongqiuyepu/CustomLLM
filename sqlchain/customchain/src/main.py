import util
from customchain.model import llm
from customchain.model import llm_p
from customchain.chain import langchain
from customchain.pandasai import plot


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # entry your openai api key
    util.openai_api_key_input()

    # connect to sqlite db
    db = util.db_connector()

    # load model gpt3.5 for langchain
    llm_o = llm.llm_load()

    # give customized prompt for gpt to perform instructions
    prompt = llm.prompt_load()

    # generate sqlchain
    sqlchain = langchain.sql_chain_load(llm_o,db,prompt)

    # run sqlchain
    answer = langchain.sql_chain_run(sqlchain)

    # add [] and remove \n to convert output in JSON format
    #parsed_json = util.json_parser(answer)

    # load model gpt3.5 for pandasai
    llm_p_o = llm_p.llm_p_load()

    # convert json to dataframe
    df = plot.json_load(answer)

    # generate pandasai instance
    apndasai = plot.pandasai_instance_load(llm_p_o)

    # Plot the histogram of countries showing for each the gpd, using different colors for each bar
    res = plot.pandasai_run(df,apndasai)
    print(res)





