import time
import random 

from core import run_llm

from langchain_community.callbacks import get_openai_callback
import streamlit as st
from typing import Set
from base64 import b64encode


def show_pdf(i, file_path, page, score):
    
    # reference = '(' + str(i) + ') ' + file_path.replace('/home/ec2-user', '') + ' - page ' + str(page)
    reference = file_path[file_path.find('VSA_ReportDB_for_AI'):] + ' - page ' + str(page)
    if show_scores:
        reference = reference + ' (score: ' + str(round(score, 4)) + ')'
    
    with st.expander(reference):
        with open(file_path, 'rb') as f:
            base64_pdf = b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}#page={page}" width={650} height={500} type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)


def display_sources(sources_list):
    if not sources_list:
        return ''
    
    st.write("**search results**:")

    for (i, (file_path, page, score)) in enumerate(sources_list):
        show_pdf((i+1), file_path, page, score)


# Initialize chat history

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []
    
if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []
    
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


st.set_page_config(page_title="ChatFPT", page_icon=":car:")
st.title(body="**Chat**_:red[FPT]_ :speech_balloon:")

#--------------------------------------------------------------

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

#--------------------------------------------------------------

with st.chat_message(name="ai", avatar="🤖"):
    st.markdown("Ciao! How may AI help you?") # not appended to the chat history

if question := st.chat_input(placeholder="Please enter your prompt here ...") :
    if len(st.session_state["chat_history"]) == 0 or question != st.session_state["chat_history"][-1][0]:
        st.session_state["user_prompt_history"].append(question)

# Display previous question-answer pairs, if they exists
if st.session_state["chat_answers_history"]:
    for (question, formated_response) in zip(
        st.session_state["user_prompt_history"],
        st.session_state["chat_answers_history"],
    ):       
        with st.chat_message(name="user", avatar="👨‍🔬"):
            st.markdown(question)

        with st.chat_message(name="ai", avatar="🤖"):
            st.markdown(formated_response[0])
            display_sources(formated_response[1])

wait_message = ["Wait for it ... ", "Just a moment, please ... ", "Thinking ... ", "Generating answer ... ", 
                "Working on it ... ", "Any moment now ... ", "Please wait ... ", "Searching database ... "]
wait_emoji = [":hourglass_flowing_sand:", ":timer_clock:", ":stopwatch:", ":alarm_clock:", ":mantelpiece_clock:"]

# display and answer the latest user question
if st.session_state["user_prompt_history"]:

    question = st.session_state["user_prompt_history"][-1]
    if len(st.session_state["chat_history"]) == 0 or question != st.session_state["chat_history"][-1][0]:

        with st.chat_message(name="user", avatar="👨‍🔬"):
            st.markdown(question)

        with st.chat_message(name="ai", avatar="🤖"):
            with st.spinner(text = random.choice(wait_message) + random.choice(wait_emoji)):
            
                with get_openai_callback() as cb:
                    generated_response = run_llm(
                        question = question,
                        chat_history = st.session_state["chat_history"],
                        deployment_name = deployment_name,
                        search_type = search_type,
                        k  = k,
                        score_threshold = score_threshold,
                    )
                    with st.sidebar:
                        st.text(cb)
            
#               for x in generated_response["source_documents"]:
#                    print(x.page_content, '\n')

                # should not contain any duplicates 
                metadata = list([(doc.metadata["source"], doc.metadata["page"], doc.metadata["score"]) for doc in generated_response["source_documents"]]) 
                formated_response = (generated_response["answer"], metadata)

                placeholder = st.empty()
                full_response = ''
                for item in formated_response[0]:
                    full_response += item
                    time.sleep(0.01)
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)

#               st.markdown(formated_response[0])
                display_sources(formated_response[1])

        st.session_state["chat_history"].append((question, generated_response["answer"]))
        st.session_state["chat_answers_history"].append(formated_response)

#st.balloons()
#st.snow()