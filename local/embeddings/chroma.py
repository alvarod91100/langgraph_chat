from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import logging
from dotenv import load_dotenv
import pypdf
import os
load_dotenv()

embedding_model = HuggingFaceEmbeddings(
    model_name = os.getenv("EMBEDDING_MODEL"), 
    cache_folder = os.getenv("EMBEDDING_MODEL_CACHE")
)

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embedding_model,
    persist_directory=os.getenv("VECTOR_DATABASE_PATH"), 
)