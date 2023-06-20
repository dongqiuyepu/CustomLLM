from langchain import SQLDatabaseChain


def sql_chain_load(llm, db, prompt):
    sqlchain = SQLDatabaseChain.from_llm(llm, db, prompt, verbose=True)
    return sqlchain


def sql_chain_run(sqlchain):
    answer = sqlchain.run("list all in goodcountryindex table")
    return answer
