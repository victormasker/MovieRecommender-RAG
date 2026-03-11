import time
import os
from datetime import datetime
import streamlit as st
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from utils import local_llm, huggingface_instruct_embedding

DB_PATH = '../vectorstore_faiss'
LOG_DIR = '../logs'
LOG_FILE = os.path.join(LOG_DIR, 'rag_queries.log')

st.set_page_config(layout='wide', page_title="Movie Recommender RAG")

st.title('Movie Recommender RAG with LLaMA3')

prompt = ChatPromptTemplate.from_template(
    """
    The user is looking for movie recommendations. Based on the movie reviews provided below, recommend films that match what the user is looking for.

    The user is looking for: {input}

    <context>
    {context}
    </context>

    Based on these reviews, recommend the most relevant films and explain why they match the user's preferences.
    """
)


def load_vector_store():
    """Load the pre-built FAISS vector store from disk."""
    if 'vectors' not in st.session_state:
        if not os.path.exists(DB_PATH):
            st.error(
                'Vector store not found. Run `python build_vectorstore.py` first to build it.'
            )
            st.stop()
        with st.spinner('Loading vector store...'):
            embeddings = huggingface_instruct_embedding()
            st.session_state.vectors = FAISS.load_local(
                DB_PATH, embeddings, allow_dangerous_deserialization=True
            )
        st.success('Vector store loaded.')


def log_query(user_question, context_docs, full_prompt, answer, response_time):
    """Append a query log entry to the log file."""
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    chunks_text = ''
    for i, doc in enumerate(context_docs, 1):
        chunks_text += f'\n--- Chunk {i} ---\n{doc.page_content}\n'

    entry = (
        f'{"=" * 80}\n'
        f'TIMESTAMP: {timestamp}\n'
        f'RESPONSE TIME: {response_time:.2f}s\n'
        f'{"=" * 80}\n\n'
        f'USER QUESTION:\n{user_question}\n\n'
        f'RETRIEVED CHUNKS ({len(context_docs)}):{chunks_text}\n'
        f'FULL PROMPT SENT TO LLM:\n{full_prompt}\n\n'
        f'LLM ANSWER:\n{answer}\n\n'
    )

    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(entry)


# Load vector store on startup
load_vector_store()

user_input = st.text_input('Enter your question from documents')

if user_input:
    document_chain = create_stuff_documents_chain(local_llm(), prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()

    response = retrieval_chain.invoke({'input': user_input})
    response_time = time.process_time() - start

    st.write(response['answer'])
    st.write(f'Response time: {response_time:.2f} secs')

    # Build the full prompt text for logging
    context_text = '\n\n'.join(doc.page_content for doc in response['context'])
    full_prompt = prompt.format(context=context_text, input=user_input)

    log_query(user_input, response['context'], full_prompt, response['answer'], response_time)

    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
