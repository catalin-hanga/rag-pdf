from dotenv import load_dotenv
load_dotenv()

from langchain_openai import AzureChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

from typing import Any, Dict, List, Tuple


def run_llm(
        question: str,
        chat_history: List[Tuple[str, Any]] = [],
        deployment_name: str = 'gpt4',
        search_type: str = 'similarity',
        k: int = 3,
        score_threshold: float = 0.5,
) -> Any:
    
    embeddings = AzureOpenAIEmbeddings(azure_deployment='text-embedding-ada-002')

    docsearch = FAISS.load_local(
        folder_path='faiss-index-1000-100',
        embeddings=embeddings
    )

    chat = AzureChatOpenAI(
        azure_deployment=deployment_name,
        verbose=True,
        temperature=0
    )

    template = (
                        "Combine the chat history and follow up question into a standalone question." 
                        "Chat History: {chat_history}"
                        "Follow up question: {question}"
    )
    
    condense_question_prompt = PromptTemplate.from_template(template)

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat,
        chain_type='stuff',
        return_source_documents=True,
        response_if_no_docs_found='Unfortunately, I am unable to find any information available on this topic',
        #condense_question_prompt=condense_question_prompt,       
        retriever=docsearch.as_retriever(search_type = search_type, search_kwargs={"k": k, "score_threshold": score_threshold}),
    )

#    q = condense_question_prompt.format(question=question, chat_history=chat_history)
#    print(q)
#    print(chat.invoke(q))

    return qa.invoke({'question': question, 'chat_history': chat_history})


# if __name__ == "__main__":
#    print(run_llm(query="In the case of the Cursor 9, can you give a summary of the Static FEA study on Pulley ?"))
