from dotenv import load_dotenv
load_dotenv()

from langchain_openai import AzureChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.docstore.document import Document

from typing import Any, Dict, List, Tuple

    
def run_llm(
        question: str,
        chat_history: List[Tuple[str, Any]] = [],
        deployment_name: str = 'gpt4',
        search_type: str = 'similarity',
        k: int = 3,
        score_threshold: float = 0.5,
) -> Dict:

    class MyVectorStoreRetriever(VectorStoreRetriever):
        def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:

            if search_type == "similarity":
                docs_and_similarities = (self.vectorstore.similarity_search_with_score(query, **self.search_kwargs))
            else:   #   search_type == 'similarity_score_threshold'
                docs_and_similarities = (self.vectorstore.similarity_search_with_relevance_scores(query, **self.search_kwargs))
        
            # Make the score part of the document metadata
            for (doc, similarity) in docs_and_similarities:
                doc.metadata["score"] = similarity

            docs = [doc for (doc, _) in docs_and_similarities]
            
            return docs

        
    embeddings = AzureOpenAIEmbeddings(azure_deployment = 'text-embedding-ada-002')

    vectorstore = FAISS.load_local(
        folder_path = 'faiss-index-1000-100',
        embeddings = embeddings
    )
    
    llm = AzureChatOpenAI(
        azure_deployment = deployment_name,
        verbose = True,
        temperature = 0,
    )

#    template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

#    Chat History:
#    {chat_history}
#    Follow Up Input: {question}
#    Standalone question:"""
#    condense_question_prompt = PromptTemplate.from_template(template)

    qa = ConversationalRetrievalChain.from_llm(
        llm = llm,
        chain_type = 'stuff',
        return_source_documents = True,
#        response_if_no_docs_found = 'Unfortunately, I am unable to find any information available on this topic',
#        condense_question_prompt=condense_question_prompt,        
#        retriever = vectorstore.as_retriever(search_type = search_type, search_kwargs = {"k": k, "score_threshold": score_threshold} ),
        retriever = MyVectorStoreRetriever(
               vectorstore = vectorstore,
#               search_type = search_type, # not necessary
               search_kwargs = {"k": k, "score_threshold": score_threshold},
            )
    )

#    cqp = condense_question_prompt.format(question = question, chat_history = chat_history)
#    condensed_question = llm.invoke(cqp)

    return qa.invoke({'question': question, 'chat_history': chat_history})


# if __name__ == "__main__":
#    print(run_llm(query="In the case of the Cursor 9, can you give a summary of the Static FEA study on Pulley ?"))
