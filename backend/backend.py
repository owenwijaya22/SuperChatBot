import os
import sys
import gc
import uuid
import traceback
from typing import List
from io import BytesIO

import boto3
import docx
import pymongo
import awswrangler as wr
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, status, HTTPException, Request
import time
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document as LangchainDocument
from langchain_community.callbacks.manager import get_openai_callback

# from langchain.vectorstores.redis import Redis as RedisVectorStore

# redis_url = "redis://redis:6379"


if "OPENAI_API_BASE" in os.environ:
    del os.environ["OPENAI_API_BASE"]
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
S3_KEY = os.getenv("S3_KEY")
S3_SECRET = os.getenv("S3_SECRET")
S3_BUCKET = os.getenv("S3_BUCKET")
S3_REGION = os.getenv("S3_REGION")
S3_PATH = os.getenv("S3_PATH")
MONGO_URL = os.getenv("MONGO_URL")

# Initialize Azure services
embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-ada-002",
    azure_endpoint=AZURE_ENDPOINT,
    api_key=OPENAI_API_KEY,
    openai_api_version="2023-07-01-preview",
)

llm = AzureChatOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    openai_api_version="2023-07-01-preview",
    deployment_name="GPT4",
    openai_api_key=OPENAI_API_KEY,
    openai_api_type="azure",
    model_name="text-embedding-ada-002",
    temperature=0,
)

# Initialize AWS S3 to read file
s3 = boto3.client(
    "s3",
    aws_access_key_id=S3_KEY,
    aws_secret_access_key=S3_SECRET,
    region_name=S3_REGION,
)

# Initialize aws s3 session for uploading file
aws_s3 = boto3.Session(
    aws_access_key_id=S3_KEY,
    aws_secret_access_key=S3_SECRET,
    region_name=S3_REGION,
)

# Initialize MongoDB
try:
    client = pymongo.MongoClient(MONGO_URL, uuidRepresentation="standard")
    db = client["chat_with_doc"]
    conversationcol = db["chat-history"]
    conversationcol.create_index([("session_id")], unique=True)
except:
    print(traceback.format_exc())
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print(exc_type, fname, exc_tb.tb_lineno)


# Pydantic models
class ChatMessageSent(BaseModel):
    session_id: str = None
    user_input: str
    data_source: str


# Helper functions
def read_file_from_s3(file_name: str) -> str:
    """Read and extract text content from S3 files"""
    file_content = s3.get_object(Bucket=S3_BUCKET, Key=f"{S3_PATH}{file_name}")[
        "Body"
    ].read()

    if file_name.endswith(".pdf"):
        pdf_reader = PdfReader(BytesIO(file_content))
        return " ".join(page.extract_text() for page in pdf_reader.pages)

    elif file_name.endswith(".docx"):
        docx_reader = docx.Document(BytesIO(file_content))
        return "\n".join(paragraph.text for paragraph in docx_reader.paragraphs)

    raise ValueError("Unsupported file format. Please use PDF or DOCX files.")


def get_response(
    file_name: str,
    session_id: str,
    query: str,
    model: str = "text-embedding-ada-002",
    temperature: float = 0,
):
    file_name = file_name.split("/")[-1]
    text_content = read_file_from_s3(file_name)

    data = [LangchainDocument(page_content=text_content)]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, separators=["\n", " ", ""]
    )
    all_splits = text_splitter.split_documents(data)
    vectorstore = FAISS.from_documents(all_splits, embeddings)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectorstore.as_retriever(),
        condense_question_prompt=PromptTemplate.from_template(
            "You are a professional document analyzer. Please answer the following question based on the document content. "
            "Be direct and precise. If the information is not in the document, clearly state that. "
            "Remember, you are Bob, the analyzer, not the user asking the question.\n\nQuestion: {question}"
        ),
    )

    with get_openai_callback() as cb:
        answer = qa_chain.invoke(
            {
                "question": query,
                "chat_history": load_memory_to_pass(session_id=session_id),
            }
        )
        answer["total_tokens_used"] = cb.total_tokens

    gc.collect()
    return answer


def load_memory_to_pass(session_id: str):
    data = conversationcol.find_one({"session_id": session_id})
    history = []
    if data:
        data = data["conversation"]
        for x in range(0, len(data), 2):
            history.extend([(data[x], data[x + 1])])
    return history


def get_session() -> str:
    return str(uuid.uuid4())


def add_session_history(session_id: str, new_values: List):
    document = conversationcol.find_one({"session_id": session_id})
    if document:
        conversation = document["conversation"]
        conversation.extend(new_values)
        conversationcol.update_one(
            {"session_id": session_id}, {"$set": {"conversation": conversation}}
        )
    else:
        conversationcol.insert_one(
            {"session_id": session_id, "conversation": new_values}
        )


app = FastAPI()

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    # Log the timing
    print(f"Time taken: {process_time:.3f} seconds")
    # Add timing to response headers
    response.headers["X-Process-Time"] = str(process_time)
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API endpoints
@app.post("/chat")
async def create_chat_message(chats: ChatMessageSent):
    try:
        session_id = chats.session_id or get_session()
        payload = ChatMessageSent(
            session_id=session_id,
            user_input=chats.user_input,
            data_source=chats.data_source,
        ).model_dump()

        response = get_response(
            file_name=payload.get("data_source"),
            session_id=payload.get("session_id"),
            query=payload.get("user_input"),
        )

        add_session_history(
            session_id=session_id,
            new_values=[payload.get("user_input"), response["answer"]],
        )

        return JSONResponse(
            content={
                "response": response,
                "session_id": str(session_id),
            }
        )

    except Exception:
        print(traceback.format_exc())
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        raise HTTPException(status_code=status.HTTP_204_NO_CONTENT, detail="error")


@app.post("/uploadFile")
async def uploadtos3(data_file: UploadFile):
    try:
        content = await data_file.read()
        
        # Upload directly to S3 using boto3 client
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=f"{S3_PATH}{data_file.filename.split('/')[-1]}",
            Body=content
        )

        response = {
            "filename": data_file.filename.split("/")[-1],
            "file_path": f"s3://{S3_BUCKET}/{S3_PATH}{data_file.filename.split('/')[-1]}",
        }
        return JSONResponse(content=response)

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Item not found")


import uvicorn

if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
