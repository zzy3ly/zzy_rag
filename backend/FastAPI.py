"""
how to start this file
pip install -r requirements.txt

uvicorn FastAPI:app --port 8000 --reload

"""

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel
from typing import List
import time
import os
import uuid

import json


# import ModularChatbot from wherever it is defined
from ChatBot import ModularChatbot
from LLM import LLMInteraction
from Database import Database
from Ingestion import IngestionManager

iso_LLM = LLMInteraction()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    # allow_origins=["http://localhost:3000"],  # List your frontend's origins here
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# initialize Database here
db = Database()

# initialize ModularChatbot here
chatbot = ModularChatbot(database=db)
_backend_dir = os.path.dirname(os.path.abspath(__file__))
ingestion = IngestionManager(
    chatbot.embedding_handler,
    database_path=os.path.join(_backend_dir, "ingestion_data", "lifecycle.db"),
    blob_directory=os.path.join(_backend_dir, "ingestion_data", "blobs"),
)


@app.on_event("shutdown")
def stop_ingestion_workers():
    ingestion.shutdown()

class ChatRequest(BaseModel):
    user_id: str
    message: str
    conversation_id: str

@app.post("/chat/")
# 大改
async def chat(request: ChatRequest):
    print(request.conversation_id)
    def response_stream():
        found_documents = False  # 默认没有找到相关文档
        for response, similar_documents in chatbot.handle_query(request.message, request.user_id, request.conversation_id):
                if similar_documents:  # 如果有检索到文档
                    found_documents = True
                    yield response
        
        if not found_documents:
            yield "未检索到相关文档，无法回答您的问题。"
    return StreamingResponse(response_stream())

class ModelUpdateRequest(BaseModel):
    name_of_model: str

@app.post("/update_model/")
async def update_model(request: ModelUpdateRequest):
    chatbot.llm_interaction.switch_model(request.name_of_model)
    return {"message": f"Model updated to {request.name_of_model}"}

@app.get("/get_paid_status/")
async def get_paid_status():
    status = chatbot.llm_interaction.get_paid_status()
    return {"status": status}


# Endpoint to add user
class User(BaseModel):
    user_id: str

@app.post("/add_user/")
async def add_user(user: User):
    user_id = user.user_id
    user_added = db.add_user(user_id)
    if user_added:
        return {"message": f"User {user_id} added successfully"}
    else:
        return {"message": f"User {user_id} already exists in the database"}


# Endpoint to add conversation
class Conversation(BaseModel):
    collection_id: str
    user_id: str
    conversation_id: str
    
class ConversationId(BaseModel):
    conversation_id: str

@app.post("/add_conversation/")
async def add_conversation(conversation: Conversation):
    conversation_id = db.add_conversation(collection_id=conversation.collection_id, user_id=conversation.user_id, conversation_id=conversation.conversation_id)
    return {"conversation_id": conversation_id}

@app.post("/delete_conversation/")
async def delete_conversation(conversation_id: ConversationId):
    db.remove_conversation(conversation_id.conversation_id)
    return {"message": f"Conversation deleted: {conversation_id.conversation_id}"}

@app.get("/get_conversations_by_user/")
async def get_conversations_by_user(user_id: str):
    conversations = db.get_conversations_by_user(user_id)
    return {"conversations": conversations}

@app.get("/get_conversations_by_collection/")
async def get_conversations_by_collection(collection_id: str):
    conversations = db.get_conversations_by_collection(collection_id)
    return {"conversations": conversations}

@app.get("/get_empty_conversation_by_collection/")
async def get_empty_conversation_by_collection(collection_id: str):
    conversations = db.get_empty_conversation_by_collection(collection_id)
    return {"conversations": conversations}

@app.get("/get_conversation_messages/")
async def get_conversation_messages(conversation_id: str):
    conversation_messages = db.get_conversation_messages(conversation_id)
    return {"conversation_messages": conversation_messages}

@app.get("/get_conversation_exists/")
async def get_conversation_exists(conversation_id: str, user_id: str):
    conversation_exists = db.conversation_exists(conversation_id, user_id)
    return {"message": conversation_exists}

class MessageRequest(BaseModel):
    conversation_id: str
    text: str
    is_user: bool
    is_complete: bool

@app.post("/add_message/")
async def add_message(request: MessageRequest):
    # add a message to a conversation
    message_id = db.add_message(conversation_id=request.conversation_id, text=request.text, 
                    is_user=request.is_user, is_complete=request.is_complete)
    return {"message_id": message_id}
    
class Collection(BaseModel):
    user_id: str
    name: str
    collection_id: str
    
@app.get("/get_collection_exists/")
async def get_collection_exists(collection_id: str, user_id: str):
    collection_exists = db.collection_exists(collection_id=collection_id, user_id=user_id)
    return {"message": collection_exists}
    
@app.post("/add_collection/")
async def add_collection(collection: Collection):
    collection_id = db.add_collection(user_id=collection.user_id, name=collection.name, collection_id=collection.collection_id)
    if collection_id is None:
        collection_id = "exists"
    return {"collection_id": collection_id}

class CollectionId(BaseModel):
    collection_id: str

@app.post("/delete_collection/")
async def delete_collection(collection_id: CollectionId):
    print(f"Starting deletion of collection with ID: {collection_id.collection_id}")
    # retrieve all document IDs associated with the collection
    documents = db.get_documents_by_collection(collection_id.collection_id)
    print(f"Retrieved documents: {documents}")
    # attempt to remove documents from the embedding handler
    doc_ids = [doc.document_id for doc in documents]
    print(f"Document IDs to remove: {doc_ids}")
    try:
        chatbot.embedding_handler.remove_documents(doc_ids)
        print("Documents successfully removed from embedding handler.")
    except Exception as e:
        print(f"Failed to delete documents: {str(e)}")
        return {"status": "error", "message": f"Failed to delete documents: {str(e)}"}
    # remove the collection itself
    db.remove_collection(collection_id.collection_id)
    print(f"Collection {collection_id.collection_id} removed from database.")
    return {"message": f"Collection deleted: {collection_id.collection_id}"}

@app.get("/get_collections_by_user/")
async def get_collections_by_user(user_id: str):
    collections = db.get_collections_by_user(user_id)
    return {"collections": collections}

@app.get("/get_documents_by_collection/")
async def get_documents_by_collection(collection_id: str):
    documents = db.get_documents_by_collection(collection_id)
    return {"documents": documents}

class DocumentId(BaseModel):
    document_id: str
    
@app.post("/delete_document/")
async def delete_document(document_id: DocumentId):
    try:
        db.remove_document(document_id.document_id)
        chatbot.embedding_handler.remove_documents([document_id.document_id])
        return {"status": "success", "message": f"Document deleted: {document_id.document_id}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

### FILE/URL PROCESSING ###

@app.post("/process_file/", status_code=202)
async def process_file(collection_id: str = Form(...), file: UploadFile = File(...),
                       document_key: str = Form(None)):
    """Persist the upload and return before parsing or embedding starts."""
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    if not db.get_collection(collection_id):
        raise HTTPException(status_code=404, detail="Collection not found")
    filename = file.filename or "upload.txt"
    file_extension = os.path.splitext(filename)[1].lower()
    if file_extension == '.zip':
        contents = await file.read()
        zip_contents = chatbot.embedding_handler.read_zip_contents(contents, filename)
        if zip_contents[0] == "error":
            raise HTTPException(status_code=400, detail=zip_contents[1])
        jobs = []
        for file_name, file_content in zip_contents[1]:
            if file_name == "error":
                raise HTTPException(status_code=400, detail=file_content)
            submitted = ingestion.submit(
                collection_id, file_name, file_content.encode("utf-8"),
                document_key=file_name)
            db.add_document(submitted["document_id"], collection_id,
                            file_name, "Managed by ingestion lifecycle")
            jobs.append(submitted)
        return {"status": "accepted", "jobs": jobs}
    try:
        await file.seek(0)
        job = await run_in_threadpool(
            ingestion.submit_stream, collection_id, filename, file.file,
            document_key or filename)
        db.add_document(job["document_id"], collection_id, filename,
                        "Managed by ingestion lifecycle")
        return {"status": "accepted", **job}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/ingestion_jobs/{job_id}")
async def ingestion_status(job_id: str):
    result = ingestion.status(job_id)
    if not result:
        raise HTTPException(status_code=404, detail="Ingestion job not found")
    return result


@app.post("/ingestion_jobs/{job_id}/retry", status_code=202)
async def retry_ingestion(job_id: str):
    try:
        return ingestion.retry(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Ingestion job not found")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


class VersionRollback(BaseModel):
    version_id: str


@app.get("/documents/{document_id}/versions")
async def document_versions(document_id: str):
    return {"document_id": document_id,
            "versions": ingestion.store.history(document_id)}


@app.post("/documents/{document_id}/rollback")
async def rollback_document(document_id: str, request: VersionRollback):
    try:
        return ingestion.rollback(document_id, request.version_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Document version not found")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

@app.post("/process_url/")
async def process_url(collection_id: str = Form(...), url: str = Form(...)):
    document_id = str(uuid.uuid4())
    if not url:
        raise HTTPException(status_code=400, detail="No URL provided")
    try:
        print(f"Processing URL for collection ID: {collection_id}")
        # process the URL using the embedding handler
        processing_result = chatbot.embedding_handler.fully_process_url(url, document_id)
        if processing_result[0] == "success":
            # add the document to the database if processing was successful
            db.add_document(document_id, collection_id, url, "URL content processed")
            return {"status": "success", "message": "URL processed successfully", "document_id": document_id}
        else:
            return {"status": "error", "message": f"Error processing URL: {processing_result[1]}", "document_id": document_id}
    except Exception as e:
        return {"status": "error", "message": f"Error processing URL: {str(e)}", "document_id": document_id}
