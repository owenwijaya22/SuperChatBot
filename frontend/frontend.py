import requests
import json

import os

BACKEND_URL = os.getenv("BACKEND_URL") or "http://localhost:8000"

def chat(user_input, data, session_id=None):
    """
    Sends a user input to a chat API and returns the response.

    Args:
        user_input (str): The user's input.
        data (str): The data source.
        session_id (str, optional): Session identifier. Defaults to None.

    Returns:
        tuple: A tuple containing the response answer and the updated session_id.
    """
    # API endpoint for chat
    url = BACKEND_URL+"/chat"

    # Print inputs for debugging
    print("user ", user_input)
    print("data", data)
    print("session_id", session_id)

    # Prepare payload for the API request
    if session_id is None:
        payload = json.dumps({"user_input": user_input, "data_source": data})
    else:
        payload = json.dumps(
            {"user_input": user_input, "data_source": data, "session_id": session_id}
        )

    # Set headers for the API request
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
    }

    # Make a POST request to the chat API
    response = requests.request("POST", url, headers=headers, data=payload)
    print(response)
    # Print the API response for debugging
    print(response.json())

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Return the response answer and updated session_id
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


import streamlit as st
import time
import os

# Set page configuration for the Streamlit app
st.set_page_config(page_title="Document Chat", page_icon="📕", layout="wide")

# Initialize chat history and session variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sessionid" not in st.session_state:
    st.session_state.sessionid = None

# Allow user to upload a file (PDF or DOCX)
data_file = st.file_uploader(
    label="Input file", accept_multiple_files=False, type=["pdf", "docx"]
)
st.divider()

# Process the uploaded file if available
if data_file:
    # Directly upload the file to the specified API endpoint
    s3_upload_url = upload_file(data_file)

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
                assistant_response, session_id = chat(prompt, data=s3_upload_url)
                st.session_state.sessionid = session_id
            else:
                assistant_response, session_id = chat(prompt, session_id=st.session_state.sessionid, data=s3_upload_url)

            message_placeholder = st.empty()
            full_response = ""

            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.03)
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})