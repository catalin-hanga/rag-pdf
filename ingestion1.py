import pickle
import os, glob
from tqdm import tqdm
from langchain.document_loaders import PyPDFLoader, MergedDataLoader

loaders_list = []
pdf_path = '/home/ec2-user/FEM-23-002/**'

for filename in tqdm(glob.iglob(pdf_path, recursive=True)):
    
    if os.path.isfile(filename) and ('.' in filename):
        extension = filename.split('.')[-1].lower()
        
        if extension == 'pdf':
            loader = PyPDFLoader(file_path=filename)
            loaders_list.append(loader)

loader_all = MergedDataLoader(loaders=loaders_list)
docs_all = loader_all.load()
print("Loaded", len(loaders_list), "pdf documents, with", len(docs_all), "pages in total")

# fix page number
for p in docs_all:
    p.metadata['page'] += 1

with open("docs.txt", "wb") as fp:
    pickle.dump(docs_all, fp)
