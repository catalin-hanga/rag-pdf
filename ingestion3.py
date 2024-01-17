import os, glob, time
from tqdm import tqdm
from langchain.document_loaders import PyPDFLoader, MergedDataLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

os.environ['OPENAI_API_TYPE'] = "azure"
os.environ['OPENAI_API_KEY']  = "eac36d2da3a4436aa801debaa529517b"
os.environ['OPENAI_API_BASE'] = "https://openai-ailab-002.openai.azure.com/"
os.environ['OPENAI_API_VERSION'] = "2023-05-15"

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
doc =  Document(page_content="text", metadata={"source": "local"})
vectorstore_final = FAISS.from_documents([doc], embeddings)

pdf_path = '/home/ec2-user/VSA_ReportDB_for_AI/**'

for filename in tqdm(glob.iglob(pdf_path, recursive=True)):
    
    if os.path.isfile(filename) and ('.' in filename):    
        extension = filename.split('.')[-1].lower()
        
        if extension == 'pdf':
            loader = PyPDFLoader(file_path=filename)

            document = loader.load()

            for page in document:
                page.metadata['page'] += 1

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""])
            chunks = text_splitter.split_documents(documents = document)

            for i in range(len(chunks)):
                vectorstore_temp = FAISS.from_documents([chunks[i]], embeddings)
                vectorstore_final.merge_from(vectorstore_temp)
                time.sleep(0.1)


#print(vectorstore_final.docstore._dict)
vectorstore_final.save_local("./faiss-index-new")
print("****** Added to FAISS vectorstore vectors")
