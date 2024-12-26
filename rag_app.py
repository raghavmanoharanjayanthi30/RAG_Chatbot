
import yaml
with open('chatgpt_api_credentials.yml', 'r') as file:
    api_creds = yaml.safe_load(file)
    
import os
os.environ['OPENAI_API_KEY'] = api_creds['openai_key']

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from operator import itemgetter
import streamlit as st
import tempfile
import os
import pandas as pd
import logging
from langchain.chains import RetrievalQA


st.set_page_config(page_title = 'RAG ChatBot')
st.title('RAG ChatBot')
@st.cache_resource(ttl="1h")

def configure_retriever(uploaded_files):
    """
    Configures the retriever for the RAG chatbot.

    Args:
        uploaded_files: List of uploaded PDF files.

    Returns:
        Retriever object for retrieving relevant documents, or None on error.
    """
    
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())

            try:
                loader = PyMuPDFLoader(temp_filepath)
                docs.extend(loader.load())
            except Exception as e:
                logging.error(f"Error loading PDF file: {file.name} - {e}")
                st.error(f"Error loading PDF file: {file.name}. Please check the file.")
    
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        doc_chunks = text_splitter.split_documents(docs)

        embeddings_model = OpenAIEmbeddings()
        
    try:
        #st.write('before-vectordb')
        persist_directory = '/Users/raghavraahul/Downloads'
        vectordb = Chroma.from_documents(doc_chunks, embeddings_model, persist_directory = persist_directory)
        #st.write('vectordb')

        retriever = vectordb.as_retriever()
        #st.write('retriever')
        return retriever

    except Exception as e:
        logging.error(f"Unexpected error configuring retriever: {e}")
        st.error("An error occurred while configuring the retriever. Please try again.")
        return None

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text = ""):
        self.container = container
        self.text = initial_text
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)
    
uploaded_files = st.sidebar.file_uploader(
    label = 'Upload pdf files for RAG',
    type = ['pdf'],
    accept_multiple_files = True
)

if not uploaded_files:
  st.info("Please upload PDF documents to continue.")
  st.stop()

retriever = configure_retriever(uploaded_files)

chatgpt = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0,
                     streaming=True)
    
# Create a prompt template for QA RAG System
qa_template = """
              Use only the following pieces of context to answer the question at the end.
              If you don't know the answer, just say that you don't know,
              don't try to make up an answer. Keep the answer as concise as possible.

              {context}

              Question: {question}
              """
qa_prompt = ChatPromptTemplate.from_template(qa_template)

# This function formats retrieved documents before sending to LLM
def format_docs(docs):
  return "\n\n".join([d.page_content for d in docs])

# Create a QA RAG System Chain
qa_rag_chain = (
  {
    "context": itemgetter("question") # based on the user question get context docs
      |
    retriever
      |
    format_docs,
    "question": itemgetter("question") # user question
  }
    |
  qa_prompt # prompt with above user question and context
    |
  chatgpt # above prompt is sent to the LLM for response
)

streamlit_msg_history = StreamlitChatMessageHistory(key="langchain_messages")

# Shows the first message when app starts
#if len(streamlit_msg_history.messages) == 0:
#  streamlit_msg_history.add_ai_message("Please ask your question?")

# Render current messages from StreamlitChatMessageHistory
for msg in streamlit_msg_history.messages:
  st.chat_message(msg.type).write(msg.content)

# Callback handler which does some post-processing on the LLM response
# Used to post the top 3 document sources used by the LLM in RAG response
class PostMessageHandler(BaseCallbackHandler):
  def __init__(self, msg: st.write):
    BaseCallbackHandler.__init__(self)
    self.msg = msg
    self.sources = []

  def on_retriever_end(self, documents, *, run_id, parent_run_id, **kwargs):
    source_ids = []
    for d in documents: # retrieved documents from retriever based on user query
      metadata = {
        "source": d.metadata["source"],
        "page": d.metadata["page"],
        "content": d.page_content[:200]
      }
      idx = (metadata["source"], metadata["page"])
      if idx not in source_ids: # store unique source documents
        source_ids.append(idx)
        self.sources.append(metadata)

  def on_llm_end(self, response, *, run_id, parent_run_id, **kwargs):
    if len(self.sources):
      st.markdown("__Sources:__ "+"\n")
      st.dataframe(data=pd.DataFrame(self.sources[:3]),
                    width=1000) # Top 3 sources

if "messages" not in st.session_state:
    st.session_state.messages = []

# If user inputs a new prompt, display it and show the response
if user_prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    
    #with st.chat_message("user"):
    #    st.markdown(user_prompt)
            
    st.chat_message("human").write(user_prompt)
    
    try:
        sources_container = st.write("")
        stream_handler = StreamHandler(st.empty())
        pm_handler = PostMessageHandler(sources_container)
        config = {"callbacks": [stream_handler, pm_handler]}
        response = qa_rag_chain.invoke({"question": user_prompt},config)
                
        st.session_state.messages.append({"role": "assistant", "content": response.content})
        #overall_history = overall_history + f"LLM Assistant: {response.content}\n"
                        
        with st.chat_message("assistant"):
            st.markdown(response.content)
    except Exception as e:
        print(f"Error: {e}")
        st.write(e)
        st.write("An error occurred while processing your request.")

    # Display the updated conversation history after each user input
    for msg in streamlit_msg_history.messages:
        st.chat_message(msg.type).write(msg.content)
        
