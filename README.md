# CustomLLM

## Run this demo:

- Start milvus database (make sure you have docker desktop installed to run docker container)
`sudo docker-compose up -d`

- Load document into milvus and index
Run `data_loader.py` with `load_documents` function

- Start streamlit app
`streamlit run streamlit_app.py`

        *To install streamlit, just run `pip install streamlit`.* You will be promped for openAI API key, please enter your key to proceed.

- It will open a browser where you can enter question and get an answer