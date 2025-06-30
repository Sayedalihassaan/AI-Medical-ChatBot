from System.helper import extracted_data , text_split , embedding
from pinecone import Pinecone , ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os



load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


extracted_data = extracted_data(data="Datasets/")
text_chunks = text_split(extracted_data=extracted_data)
embeddings = embedding()


pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medicalbot"
pc.create_index(name=index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws",region="us-east-1"))

docsearch = PineconeVectorStore.from_documents(text_chunks,
                                               index_name=index_name,
                                               embedding=embeddings)
