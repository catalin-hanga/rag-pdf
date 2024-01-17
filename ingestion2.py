import pickle

with open("docs.txt", "rb") as fp:
    documents_loaded = pickle.load(fp)

print("Loaded", len(documents_loaded), "pages in total")

#-----------------------------------------------------------------------------

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""])
documents = text_splitter.split_documents(documents = documents_loaded)

print(f"Splitted into {len(documents)} chunks")

#-----------------------------------------------------------------------------

import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

os.environ['OPENAI_API_TYPE'] = "azure"
os.environ['OPENAI_API_KEY']  = "eac36d2da3a4436aa801debaa529517b"
os.environ['OPENAI_API_BASE'] = "https://openai-ailab-002.openai.azure.com/"
os.environ['OPENAI_API_VERSION'] = "2023-05-15"

print(f"Going to insert {len(documents)} chunks into FAISS")
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

vectorstore = FAISS.from_documents(documents, embeddings) 
vectorstore.save_local("./faiss-index")
print("****** Added to FAISS vectorstore vectors")
