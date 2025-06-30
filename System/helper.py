from langchain_community.document_loaders import PyPDFLoader , DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from dotenv import load_dotenv
import os


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")



def data_extraction(path_data) :

    if not os.path.exists(path=path_data) :
        raise FileNotFoundError(f"The directory {path_data} does not exist.")
    
    loader = DirectoryLoader(path=path_data , 
                             glob="*.pdf" ,
                               loader_cls=PyPDFLoader ,
                                show_progress=True )
    docs = loader.load()
    print(f"Number of documents loaded: {len(docs)}")
    if not docs :
        print(f"No PDF files found in {path_data}")

    return docs




def text_split(extracted_data) :
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 2000 , chunk_overlap = 200)
    chunks = text_splitter.split_documents(extracted_data)
    return chunks





def embedding() :
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001" ,
                                              google_api_key=GOOGLE_API_KEY)
    
    return embedding


