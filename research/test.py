from langchain_core.prompts import ChatPromptTemplate
from src.helper import download_hugging_face_embeddings
from src.prompt import prompt
from pinecone import Pinecone
from flask import Flask, render_template, jsonify, request
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import os

app = Flask(__name__)

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

embeddings = download_hugging_face_embeddings()

#Initializing the Pinecone
pc = Pinecone(api_key='YOUR_API_KEY')

index_name="medical-bot"

#Loading the index
docsearch=pc.Index(index_name)

## Prompt Template

prompt=prompt

# ollama LLAma2 LLm 
llm=Ollama(model="llama2")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=chain.invoke({"question": input})
    print("Response : ", result)
    return str(result)



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)