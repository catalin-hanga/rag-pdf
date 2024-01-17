from typing import Set

from core import run_llm

import streamlit as st
from streamlit_chat import message
import datetime
import base64

def greetings():
    now = datetime.datetime.now().time()
    if now < datetime.time(12,0,0):
        return "Good morning,"
    elif now < datetime.time(18,0,0):
        return "Good afternoon,"
    else:
        return "Good evening,"

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

    st.write("Sources:")

    for (file_path, nr_page) in sources_list:
#        show_pdf(file_path, nr_page)
        show_pdf(file_path.replace('\\\\nas_arbon\\sensitive\\ArtificialIntelligence', '/home/ec2-user').replace('\\', '/'), nr_page)
        

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


st.title("**Chat**_:red[FPT]_")

prompt = st.chat_input(placeholder="Please enter your prompt here...")


if prompt:
    with st.spinner("Generating response..."):
        generated_response = run_llm(query=prompt, chat_history=st.session_state["chat_history"])
        sources = set([(doc.metadata["source"], doc.metadata["page"]) for doc in generated_response["source_documents"]])
        
        formated_response = (generated_response['answer'] , sources)
        
        st.session_state["chat_history"].append((prompt, generated_response["answer"]))
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formated_response)

message = st.chat_message("ai",  avatar="🤖")
message.write("Ciao! How may AI help you?")


if st.session_state["chat_answers_history"]:
    
    for (formated_response, user_query) in zip(st.session_state["chat_answers_history"], st.session_state["user_prompt_history"]):
        
        message = st.chat_message("user",  avatar="👨‍🔬")
        message.write(user_query)
        
        message = st.chat_message("ai",  avatar="🤖")
        message.write(formated_response[0])
        create_sources(formated_response[1])
