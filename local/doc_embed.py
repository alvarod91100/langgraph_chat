from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from embeddings.chroma import (
    vector_store
)
import uuid
import logging
from dotenv import load_dotenv
import pypdf
import os
load_dotenv()

def get_files(path:str):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

def load_file(path:str):
    loader = PyPDFLoader(path)
    chunks = loader.load_and_split()
    return chunks

def extract_pages(file_dir_path:str):
    files = get_files(file_dir_path)
    chunked_pages = []
    for file in files:
        file_path = os.path.join(file_dir_path, file)
        chunked_pages.extend(load_file(file_path))
        print(f"Loaded file {file}")
    return chunked_pages

# ----

chunked_pages = extract_pages(os.getenv("RAW_DOCS_PATH"))
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
documents = text_splitter.split_documents(chunked_pages)

uuids = [str(uuid.uuid4()) for _ in range(len(documents))]
uploaded_ids = vector_store.add_documents(documents=documents, ids=uuids)
print(f"Uploaded {len(uploaded_ids)} documents to the vector store\n {uploaded_ids}")