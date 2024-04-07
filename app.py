from flask import Flask, render_template, jsonify, request
#from src.helper import download_hugging_face_embeddings

#from langchain_pinecone import PineconeVectorStore
import pinecone
#from langchain_community.vectorstores import Pinecone
from langchain_community.vectorstores import Pinecone as  PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
#from src.prompt import *
import os
import sentence_transformers
import pdb


#from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
#from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_community.embeddings import HuggingFaceEmbeddings

from dotenv import load_dotenv
load_dotenv()

from src.helper import download_hugging_face_embeddings
from src.prompt import *

app = Flask(__name__)

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')
print(PINECONE_API_KEY)
print(PINECONE_API_ENV)

#Embedding method for the word to vector embedding
embed_model = "sentence-transformers/all-MiniLM-L6-v2"
embedding   = download_hugging_face_embeddings(embed_model)

#Initialize Pincone vector database client
index_name = "med-app"
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY,
                    environment=PINECONE_API_ENV)
#index = pc.Index(index_name)
#Loading the index
docsearch = PineconeVectorStore.from_existing_index(index_name, embedding)

#Use the prompting template package from LangChain
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt":PROMPT}

#Loading the LLM model
llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})

qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)


'''user_input="what is cancer? What are the most common cancer types found in men?"
result=qa({"query": user_input})
print("Response : ")
print(result["result"])'''

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)