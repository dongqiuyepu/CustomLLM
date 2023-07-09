import gradio as gr

from langchain_demo import document_qa_chain
from data_loader import load_documents_from_file_upload


def document_qa(input):
    qa = document_qa_chain()
    ans = qa.run(input)

    return ans


def upload_file(files):
    file_paths = [file.name for file in files]
    return file_paths


with gr.Blocks() as demo:
    # gr.Markdown("Start typing below and then click **Run** to see the output.")

    # file_output = gr.Video(label="File Upload", file_types=[".txt"], scale=1)
    # upload_button = gr.UploadButton("Click to Submit", file_types=["text"], file_count="multiple")
    # upload_button.upload(upload_file, upload_button, file_output)
    gr.interface(
        fn=upload_file,
        inputs=[
            gr.File(label="File Upload", file_types=[".txt"], scale=1)
        ]
    )
    gr.Interface(fn=document_qa, inputs="text", outputs="text")
# demo = gr.Interface(fn=document_qa, inputs="text", outputs="text")

demo.launch()