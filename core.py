from dotenv import load_dotenv
load_dotenv()

from langchain_openai import AzureChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS 

from typing import Any, Dict, List, Tuple


def run_llm(query: str, chat_history: List[Tuple[str, Any]] = []) -> Any: 
    
    embeddings = AzureOpenAIEmbeddings(
                azure_deployment = "text-embedding-ada-002"
            )
    
    docsearch = FAISS.load_local(
                    folder_path="faiss-index-new",
                    embeddings=embeddings
                    )
    
    chat = AzureChatOpenAI(
                           azure_deployment="gpt4",
                           #azure_deployment = "gpt-35-turbo",
                           verbose=True, 
                           temperature=0
    )

    qa = ConversationalRetrievalChain.from_llm(
                     llm=chat,
                     chain_type="stuff",
                     retriever=docsearch.as_retriever(), 
                     return_source_documents=True,
                     )
    
    return qa.invoke( {"question": query, "chat_history": chat_history})
    

#if __name__ == "__main__":
#    print(run_llm(query="In the case of the Cursor 9, can you give a summary of the Static FEA study on Pulley ?"))
