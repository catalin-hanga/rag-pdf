from core import run_llm   

import streamlit as st
from streamlit_chat import message

from langchain_community.callbacks import get_openai_callback
from typing import Set
from base64 import b64encode


def show_pdf(i, file_path, page, score):
    
    # reference = '(' + str(i) + ') ' + file_path.replace('/home/ec2-user', '') + ' - page ' + str(page)
    reference = file_path.replace('/home/ec2-user', '') + ' - page ' + str(page)
    if show_scores:
        reference = reference + ' (score: ' + str(round(score, 4)) + ')'
    
    with st.expander(reference):
        with open(file_path, 'rb') as f:
            base64_pdf = b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}#page={page}" width={650} height={500} type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)


def create_sources(sources_list):
    if not sources_list:
        return ''
    
    st.write("**Sources**:")

    for (i, (file_path, page, score)) in enumerate(sources_list):
        show_pdf((i+1), file_path, page, score)


if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

st.set_page_config(page_title="ChatFPT", page_icon=":car:")
st.title("**Chat**_:red[FPT]_")

message = st.chat_message("ai", avatar="🤖")
message.write("Ciao! How may AI help you?")


with st.sidebar:
    
    search_type = st.selectbox(
        label="Search type",
        options=("similarity", "similarity score threshold"),
        index=0,
        help="todo"
    )

    if search_type == "similarity score threshold":
        search_type = "similarity_score_threshold"

    col1, col2 = st.columns(2)

    with col1:
        k = st.number_input(label="k", min_value=1, value=3, help="todo")

    with col2:
        score_threshold = st.number_input(label="score threshold", min_value=0.0, max_value=1.0, value=0.5, format='%.4f', help="todo")

    show_scores = st.toggle(label = "show scores", value=True, help="todo")
        
    st.markdown("---")

    deployment_name = st.selectbox(
        label="Model version",
        options=("GPT-3.5", "GPT-4"),
        index=1,
        help="todo"
    )
    if deployment_name == "GPT-4":
        deployment_name = "gpt4"
        st.write("Tokens limit: 8192")
    else:
        deployment_name = "gpt-35-turbo"
        st.write("Tokens limit: 4096")


question = st.chat_input(placeholder="Please enter your prompt here ...")
if question:
    with st.spinner("Wait for it ... :hourglass_flowing_sand:"):
        with get_openai_callback() as cb:
            
            generated_response = run_llm(
                question = question,
                chat_history = st.session_state["chat_history"],
                deployment_name = deployment_name,
                search_type = search_type,
                k = k,
                score_threshold = score_threshold,
            )
            
            sources = list( 
                [
                    (doc.metadata["source"], doc.metadata["page"], doc.metadata["score"]) for doc in generated_response["source_documents"]
                ]
             ) # should not contain any duplicates

            formated_response = (generated_response["answer"], sources)

            st.session_state["chat_history"].append((question, generated_response["answer"]))
            st.session_state["user_prompt_history"].append(question)
            st.session_state["chat_answers_history"].append(formated_response)

            with st.sidebar:
                st.text(cb)


if st.session_state["chat_answers_history"]:
    for formated_response, user_query in zip(
        st.session_state["chat_answers_history"],
        st.session_state["user_prompt_history"],
    ):
        message = st.chat_message("user", avatar="👨‍🔬")
        message.write(user_query)

        message = st.chat_message("ai", avatar="🤖")
        message.write(formated_response[0])
        create_sources(formated_response[1])

#st.balloons()
#st.snow()
