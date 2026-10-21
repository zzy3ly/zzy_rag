from Embeddings import EmbeddingHandler
from LLM import LLMInteraction
import logging


_CITATION_MARKERS = (
    "\n\n---\n\n参考文档：",
    "\n\n---\n\n### Document Citations:",
)


def _without_citation_appendix(text):
    end = len(text)
    for marker in _CITATION_MARKERS:
        position = text.find(marker)
        if position >= 0:
            end = min(end, position)
    return text[:end].rstrip()

class ModularChatbot:
    def __init__(self, database, context_text=None):
        # initialize the chatbot with a database and optional context text
        self.database = database
        self.llm_interaction = LLMInteraction()
        self.embedding_handler = EmbeddingHandler()
        
        # set up initial conversation history with a system message
        self.conversation_history = [
            {
                "role": "system",
                "content": "You are a helpful knowledge retrieval agent. You are provided documents that you can use to answer user queries. You should answer the user queries only based on the provided documents, and inform the user if you either were not provided documents, or the documents do not seem to answer their question.",
            }
        ]

    def handle_query(self, query, userid, conversation_id):
        user_query = query
        full_response = ""
        context = None
        # retrieve the raw conversation history from the database
        conversation_history = self.database.get_conversation_raw(conversation_id)
        
        # retrieve related documents for the conversation
        related_docs = self.database.get_documents_by_conversation(conversation_id)
        print(f"Related docs: {related_docs}")
        doc_ids = [doc.document_id for doc in related_docs]
        
        # initialize conversation history if it does not exist
        if conversation_history is None:
            conversation_history = [
                {
                    "role": "system",
                    # 修改1
                    "content": "You are a knowledgeable AI assistant. If provided with documents, use them as a source to answer questions. If no documents are available, use your general knowledge to respond."
                    # "content": "You are a helpful knowledge retrieval agent. You are provided documents that you can use to answer user queries. You should answer the user queries only based on the provided documents, and inform the user if you either were not provided documents, or the documents do not seem to answer their question.",
                }
            ]
        else:
            for message in conversation_history:
                if message.get("role") == "assistant":
                    message["content"] = _without_citation_appendix(
                        message.get("content", ""))
        if conversation_history and conversation_history[0].get("role") == "system":
            conversation_history[0]["content"] = (
                "You are a knowledgeable assistant. Answer in the same language "
                "as the user's latest question. If the question contains Chinese, "
                "you must answer in Simplified Chinese and must not translate the "
                "answer into English. When documents are provided, base the answer "
                "on them and do not invent unsupported procedures."
            )
        # retrieve documents similar to the user's query using document IDs as filters
        metadata_filters = {"document_id": {"$in": doc_ids}}
        retrieval_status, retrieval_error, similar_documents = self.embedding_handler.retrieve_documents(query, metadata_filters=metadata_filters)

        #修改！
        found_documents = bool(similar_documents)

        # construct context from the most similar documents
        context = "<$$ THIS SIGNIFIES THE BEGINNING OF RELEVANT CONTEXTUAL DOCUMENTS $$>\n\n"
        if retrieval_status == "success" and similar_documents:
            for index, doc_info in enumerate(similar_documents, start=1):
                document_text = doc_info["text"]
                document_filename = doc_info["metadata"]["filename"]  # extracting filename from metadata
                context += f"<$$ Document Index: {index} $$>\n<$$ Filename: {document_filename} $$>\n<$$ Content: {document_text} $$>\n<$$ Document Separator $$>\n"
        elif retrieval_status == "error":
            context += "THERE WAS A SYSTEM ERROR RETRIEVING DOCUMENTS, INFORM THE USER\n\n"
            print(f"Error retrieving documents: {retrieval_error}\n\n")
        else:
            context += "NO RELEVANT DOCUMENTS FOUND, INFORM THE USER\n\n"
            print("No documents found\n\n")
        context += "<$$ THIS SIGNIFIES THE END OF ATTACHED DATA AND MARKS THE BEGINNING OF USER QUERY $$>\n\n"

        # append the user's query to the conversation history
        if context:
            query = context + query

        conversation_history.append({"role": "user", "content": query})

        # generate responses from the LLM based on the query and context
        for response in self.llm_interaction.generate_response(query, context, conversation=conversation_history):
            full_response += response
            #修改
           # yield response, similar_documents
            yield response, found_documents 


        # append the assistant's response to the conversation history
        conversation_history.append({
            "role": "assistant",
            "content": _without_citation_appendix(full_response),
        })

        # update the raw conversation history in the database
        self.database.update_conversation_raw(conversation_id, conversation_history)
        
        # check if the conversation title is "New chat" and update if necessary
        if self.database.get_conversation_title(conversation_id) == "New chat":
            try:
                new_title = self.llm_interaction.generate_title(conversation_history)
            except Exception:
                logging.getLogger(__name__).exception(
                    "Title generation failed; using the user query as fallback")
                new_title = user_query.strip().replace("\n", " ")[:40] or "New chat"
            # update the conversation title in the database
            self.database.update_conversation_title(conversation_id, new_title)
