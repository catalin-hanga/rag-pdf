import os

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_KEY"] = "eac36d2da3a4436aa801debaa529517b"
os.environ["OPENAI_API_BASE"] = "https://openai-ailab-002.openai.azure.com/" 
os.environ["OPENAI_API_VERSION"] = "2023-05-15" 

from langchain.chat_models import AzureChatOpenAI
#from langchain.schema import HumanMessage

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.vectorstores import FAISS 

from typing import Any, Dict, List, Tuple


def run_llm(query: str, chat_history: List[Tuple[str, Any]] = []) -> Any: 
    embeddings = OpenAIEmbeddings() 
    
    docsearch = FAISS.load_local(
                    folder_path="faiss-index",
                    embeddings=embeddings
                    )
    
    chat = AzureChatOpenAI(deployment_name="gpt4",
                           #deployment_name = "gpt-35-turbo", 
                           verbose=True, 
                           temperature=0)

    qa = ConversationalRetrievalChain.from_llm(
                     llm=chat,
                     chain_type="stuff",
                     retriever=docsearch.as_retriever(), 
                     return_source_documents=True,
                     )
    
    return qa( {"question": query, "chat_history": chat_history})
    

#if __name__ == "__main__":
#    print(run_llm(query="In the case of the Cursor 9, can you give a summary of the Static FEA study on Pulley ?"))
