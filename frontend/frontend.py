import requests
import json
import os
import streamlit as st
import time

BACKEND_URL = os.getenv("BACKEND_URL") or "http://localhost:8000"

def chat(user_input, data, session_id=None):
    url = BACKEND_URL + "/chat"

    if session_id is None:
        payload = json.dumps({"user_input": user_input, "data_source": data})
    else:
        payload = json.dumps(
            {"user_input": user_input, "data_source": data, "session_id": session_id}
        )

    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    if response.status_code == 200:
        return response.json()["response"]["answer"], response.json()["session_id"]

def upload_file(data_file):
    url = BACKEND_URL + "/uploadFile"
    files = {
        "data_file": (data_file.name, data_file, data_file.type)
    }
    
    headers = {"accept": "application/json"}
    response = requests.post(url, headers=headers, files=files)

    if response.status_code == 200:
        return response.json()["file_path"]

# Set page configuration for the Streamlit app
st.set_page_config(page_title="Document Chat", page_icon="📕", layout="wide")

# Initialize chat history and session variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sessionid" not in st.session_state:
    st.session_state.sessionid = None
if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False
if "s3_upload_url" not in st.session_state:
    st.session_state.s3_upload_url = None

# Allow user to upload a file (PDF or DOCX)
data_file = st.file_uploader(
    label="Input file", accept_multiple_files=False, type=["pdf", "docx"]
)
st.divider()

# Process the uploaded file if available and not already uploaded
if data_file and not st.session_state.file_uploaded:
    print("data file is checking again")
    # Directly upload the file to the specified API endpoint
    st.session_state.s3_upload_url = upload_file(data_file)
    st.session_state.file_uploaded = True
    print("uploading again from frontend")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("You can ask any question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if st.session_state.sessionid is None:
            assistant_response, session_id = chat(prompt, data=st.session_state.s3_upload_url)
            st.session_state.sessionid = session_id
        else:
            assistant_response, session_id = chat(prompt, session_id=st.session_state.sessionid, data=st.session_state.s3_upload_url)

        message_placeholder = st.empty()
        full_response = ""

        for chunk in assistant_response.split():
            full_response += chunk + " "
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})