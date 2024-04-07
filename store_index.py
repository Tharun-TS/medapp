import os
import pinecone
from langchain.vectorstores import Pinecone as  PineconeVectorStore
from dotenv import load_dotenv

load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

#Initialize the PineCone database vector
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY,
                    environment=PINECONE_API_ENV)
for name in pc.list_indexes().names():
    print(name)
index_name = "med-app"
index = pc.Index(index_name)

def create_index(index_name=None, dimensions=384, metrics='dotproduct', use_serverless=False):
    #from pinecone import Pinecone, ServerlessSpec, PodSpec Do not USE!
    import time

    #use_serverless = False
    # configure client
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

    if use_serverless:
        spec = pinecone.ServerlessSpec(cloud='aws', region='us-west-2')
    else:
        # if not using a starter index, you should specify a pod_type too
        spec = pinecone.PodSpec(environment="gcp-starter")

    # check for and delete index if already exists
    #index_name = index_name
    if index_name in pc.list_indexes().names():
        pc.delete_index(index_name)

    # create a new index
    pc.create_index(
        index_name,
        dimension=dimensions,  # dimensionality of text-embedding-ada-002
        metric=metrics,
        spec=spec
    )

    # wait for index to be initialized
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

    # configure client
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    #index_name = "med-app"
    index = pc.Index(index_name)
    index.describe_index_stats()