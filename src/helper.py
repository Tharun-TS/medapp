from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

#Method to parse all the documents in the root directory
#Each page is read into DirectoryLoader object with each element representing a page
def pdf_loader(pdf_dir):
    loader = DirectoryLoader(pdf_dir, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents  

#Each page is then split into chunks of max size 500
#Also each chunk overlaps each other by 20 tokens
def text_splitter(exrtacted_doc):
    splitter_obj = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    split_doc = splitter_obj.split_documents(exrtacted_doc)
    return split_doc

#Embeding model to conver the word to embeddings
def download_hugging_face_embeddings(embed_model):
    embeddings = HuggingFaceEmbeddings(model_name=embed_model)
    return embeddings
