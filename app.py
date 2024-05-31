from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from src.helper import download_hugging_face_embeddings, load_pdf, text_split, upsert_to_index, create_pinecone_index
from src.prompt import prompt
from dotenv import load_dotenv
from flask import Flask, render_template, jsonify, request
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import os

# loading the environment
load_dotenv()

# creating the Flask app for Chatbot
app = Flask(__name__)

# setting the pinecone API Key
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

# downloading an open-source hugging face embedding model
embeddings = download_hugging_face_embeddings() 

# pinecone index (already created)
PINECONE_INDEX_NAME="askwolpert"


# if you have not already created an index and created a vectorstore 
# of your conetextual documents, then uncomment the following lines. Please refer
# to helper.py and read the documentation for accurate arguments adjustment 


# PINECONE_INDEX_NAME = create_pinecone_index(index_name= "YOUR_INDEX_NAME", kwargs)
# extracted_data = load_pdf("data/")
# docs = text_split(extracted_data)
# vectorstore = upsert_to_index(docs, embeddings, PINECONE_INDEX_NAME)

# creating a retriever for our RAG application
vectorstore = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, 
                                  embedding=embeddings,
                                  pinecone_api_key=PINECONE_API_KEY)
retriever=vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":5})

## Prompt Template
prompt=prompt

# ollama LLAma2 LLm 
llm=Ollama(model="llama2")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

question_answer_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=retrieval_chain.invoke({"input":input})
    answer = result.get('answer', 'No answer provided')
    return str(answer)



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)