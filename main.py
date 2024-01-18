from typing import Set

from core import run_llm

import streamlit as st
from streamlit_chat import message

import datetime
import base64

from langchain_community.callbacks import get_openai_callback

def show_pdf(file_path, nr_page):
    with st.expander(file_path.replace("/home/ec2-user/", "") + " - page " + str(nr_page)):
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}#page={nr_page}" width="650" height="500" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)


def create_sources(source_urls):
    if not source_urls:
        return ""
    
    sources_list = list(source_urls) # source_urls is a set

    st.write("**Sources**:")

    for (file_path, nr_page) in sources_list:
        show_pdf(file_path, nr_page)
        

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


st.title("**Chat**_:red[FPT]_")

message = st.chat_message("ai",  avatar="🤖")
message.write("Ciao! How may AI help you?")


with st.sidebar:
    
    gpt_version = st.selectbox(
        label='Model version',
        options=('GPT-3.5', 'GPT-4'),
        index=1,
    )
    if gpt_version == 'GPT-4':
        deployment_name = 'gpt4'
        st.write('Tokens limit: 8,192')
    else:
        deployment_name = 'gpt-35-turbo'
        st.write('Tokens limit: 4,096')

    st.markdown('---')
        
prompt = st.chat_input(placeholder="Please enter your prompt here ...")
if prompt:
    with st.spinner("Generating response ..."):
        with get_openai_callback() as cb:

            generated_response = run_llm(question=prompt, chat_history=st.session_state["chat_history"], deployment_name=deployment_name)
            sources = set([(doc.metadata["source"], doc.metadata["page"]) for doc in generated_response["source_documents"]])
                
            formated_response = (generated_response['answer'] , sources)
                
            st.session_state["chat_history"].append((prompt, generated_response["answer"]))
            st.session_state["user_prompt_history"].append(prompt)
            st.session_state["chat_answers_history"].append(formated_response)

            with st.sidebar:
                st.text(cb)


if st.session_state["chat_answers_history"]:
    
    for (formated_response, user_query) in zip(st.session_state["chat_answers_history"], st.session_state["user_prompt_history"]):
        
        message = st.chat_message("user",  avatar="👨‍🔬")
        message.write(user_query)
        
        message = st.chat_message("ai",  avatar="🤖")
        message.write(formated_response[0])
        create_sources(formated_response[1])
