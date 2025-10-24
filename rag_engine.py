import os, tempfile
from pathlib import Path

from langchain.chains import ConversationalRetrievalChain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from pinecone import Pinecone

import streamlit as st

# === PATHS ===
TMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp')
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath('data', 'vector_store')

# Create directories if they don't exist
TMP_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="RAG")
st.title("Retrieval Augmented Generation Engine")

# === INITIALIZE SESSION STATE ===
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# === DOCUMENT HANDLING ===
def load_documents():
    loader = DirectoryLoader(TMP_DIR.as_posix(), glob='**/*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def split_documents(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    return texts

def embeddings_on_local_vectordb(texts):
    embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.openai_api_key)
    vectordb = Chroma.from_documents(
        texts,
        embedding=embeddings,
        persist_directory=LOCAL_VECTOR_STORE_DIR.as_posix()
    )
    vectordb.persist()
    retriever = vectordb.as_retriever(search_kwargs={'k': 7})
    return retriever

def embeddings_on_pinecone(texts):
    pc = Pinecone(api_key=st.session_state.pinecone_api_key)
    
    # Get or create index
    index = pc.Index(st.session_state.pinecone_index)
    
    embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.openai_api_key)
    
    vectordb = PineconeVectorStore.from_documents(
        documents=texts,
        embedding=embeddings,
        index_name=st.session_state.pinecone_index
    )
    
    retriever = vectordb.as_retriever()
    return retriever

# === QUERY HANDLING ===
def query_llm(retriever, query):
    if retriever is None:
        st.warning("⚠️ Please upload and process documents first.")
        return ""
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(
            openai_api_key=st.session_state.openai_api_key,
            model="gpt-3.5-turbo",
            temperature=0
        ),
        retriever=retriever,
        return_source_documents=True,
    )
    result = qa_chain({'question': query, 'chat_history': st.session_state.messages})
    answer = result['answer']
    st.session_state.messages.append((query, answer))
    return answer

# === UI INPUTS ===
def input_fields():
    with st.sidebar:
        if "openai_api_key" in st.secrets:
            st.session_state.openai_api_key = st.secrets.openai_api_key
        else:
            st.session_state.openai_api_key = st.text_input("OpenAI API key", type="password")

        if "pinecone_api_key" in st.secrets:
            st.session_state.pinecone_api_key = st.secrets.pinecone_api_key
        else:
            st.session_state.pinecone_api_key = st.text_input("Pinecone API key", type="password")

        if "pinecone_env" in st.secrets:
            st.session_state.pinecone_env = st.secrets.pinecone_env
        else:
            st.session_state.pinecone_env = st.text_input("Pinecone environment")

        if "pinecone_index" in st.secrets:
            st.session_state.pinecone_index = st.secrets.pinecone_index
        else:
            st.session_state.pinecone_index = st.text_input("Pinecone index name")

    st.session_state.pinecone_db = st.toggle('Use Pinecone Vector DB')
    st.session_state.source_docs = st.file_uploader(
        label="Upload Documents", type="pdf", accept_multiple_files=True
    )

# === DOCUMENT PROCESSING ===
def process_documents():
    if not st.session_state.openai_api_key or not st.session_state.source_docs:
        st.warning("Please upload documents and provide API keys.")
        return
    
    try:
        # Save uploaded files to temp directory
        for source_doc in st.session_state.source_docs:
            with tempfile.NamedTemporaryFile(delete=False, dir=TMP_DIR.as_posix(), suffix='.pdf') as tmp_file:
                tmp_file.write(source_doc.read())

        # Load documents
        documents = load_documents()
        
        # Clean up temp files
        for _file in TMP_DIR.iterdir():
            if _file.suffix == '.pdf':
                _file.unlink()

        # Split documents
        texts = split_documents(documents)
        
        # Create embeddings and retriever
        if st.session_state.pinecone_db:
            if not st.session_state.pinecone_api_key or not st.session_state.pinecone_index:
                st.warning("Please provide Pinecone API key and index name.")
                return
            st.session_state.retriever = embeddings_on_pinecone(texts)
        else:
            st.session_state.retriever = embeddings_on_local_vectordb(texts)
        
        st.success("✅ Documents processed successfully!")
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
        import traceback
        st.error(traceback.format_exc())

# === MAIN FUNCTION ===
def boot():
    input_fields()
    st.button("Submit Documents", on_click=process_documents)

    # Display chat history
    for message in st.session_state.messages:
        st.chat_message('human').write(message[0])
        st.chat_message('ai').write(message[1])    

    # Handle new user input
    if query := st.chat_input():
        st.chat_message("human").write(query)
        response = query_llm(st.session_state.retriever, query)
        st.chat_message("ai").write(response)

if __name__ == '__main__':
    boot()