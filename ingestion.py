import os
import glob
import time
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import MergedDataLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document


embeddings = AzureOpenAIEmbeddings(
        azure_deployment = "text-embedding-ada-002"
    )

doc =  Document(page_content="text", metadata={"source": "local"}) # create an empty document
vectorstore_final = FAISS.from_documents([doc], embeddings)

pdf_path = '/home/ec2-user/VSA_ReportDB_for_AI/**'

for filename in tqdm(glob.iglob(pdf_path, recursive=True)):
    
    if os.path.isfile(filename) and ('.' in filename):    
        extension = filename.split('.')[-1].lower()
        
        if extension == 'pdf':

            loader = PyPDFLoader(file_path=filename)
            document = loader.load()

            # increment page number
            for page in document:
                page.metadata['page'] += 1

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""])
            chunks = text_splitter.split_documents(documents = document)

            for i in range(len(chunks)):
                vectorstore_temp = FAISS.from_documents([chunks[i]], embeddings)
                vectorstore_final.merge_from(vectorstore_temp)
#                time.sleep(0.01)

vectorstore_final.delete([vectorstore_final.index_to_docstore_id[0]]) # delete the initial empty document
vectorstore_final.save_local("./faiss-index-new")
print("****** Added to FAISS vectorstore vectors")
