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
        param: float = 0.5,
) -> Any:
    
    embeddings = AzureOpenAIEmbeddings(azure_deployment = 'text-embedding-ada-002')

    docsearch = FAISS.load_local(
        folder_path = 'faiss-index-1000-100',
        embeddings = embeddings
    )
    
    llm = AzureChatOpenAI(
        azure_deployment = deployment_name,
        verbose = True,
        temperature = 0
    )

#    https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/chains/conversational_retrieval/prompts.py
#    template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

#    Chat History:
#    {chat_history}
#    Follow Up Input: {question}
#    Standalone question:"""
#    condense_question_prompt = PromptTemplate.from_template(template)

    if search_type != "mmr":
        search_kwargs = {"k": k, "score_threshold": param}
    else:
        search_kwargs = {"k": k, "lambda_mult": param}

    qa = ConversationalRetrievalChain.from_llm(
        llm = llm,
        chain_type = 'stuff',
        return_source_documents = True,
        response_if_no_docs_found = 'Unfortunately, I am unable to find any information available on this topic',
 #       condense_question_prompt=condense_question_prompt,
        retriever = docsearch.as_retriever(search_type = search_type, search_kwargs = search_kwargs),
    )

#    cqp = condense_question_prompt.format(question = question, chat_history = chat_history)
#    condensed_question = llm.invoke(cqp)
#    print(condensed_question)

    return qa.invoke({'question': question, 'chat_history': chat_history})


# if __name__ == "__main__":
#    print(run_llm(query="In the case of the Cursor 9, can you give a summary of the Static FEA study on Pulley ?"))
