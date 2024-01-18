from dotenv import load_dotenv
load_dotenv()

from langchain_openai import AzureChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS 

from typing import Any, Dict, List, Tuple


def run_llm(question: str, chat_history: List[Tuple[str, Any]] = [], deployment_name: str = "gpt4") -> Any: 
    
    embeddings = AzureOpenAIEmbeddings(
                azure_deployment = "text-embedding-ada-002"
    )
    
    docsearch = FAISS.load_local(
                    folder_path = "faiss-index-1000-100",
                    embeddings = embeddings
    )
    
    chat = AzureChatOpenAI(
                           azure_deployment = deployment_name,
                           verbose = True, 
                           temperature = 0
    )

    qa = ConversationalRetrievalChain.from_llm(
                     llm = chat,
                     chain_type = "stuff",
                     return_source_documents = True,
                     response_if_no_docs_found = "Unfortunately, I am unable to find any information on this topic",
                     
                     retriever = docsearch.as_retriever(),
    )
    
    return qa.invoke( {"question": question, "chat_history": chat_history})
    

#if __name__ == "__main__":
#    print(run_llm(query="In the case of the Cursor 9, can you give a summary of the Static FEA study on Pulley ?"))
