# import all necessary libraries
import time
import shutil
import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_objectbox.vectorstores import ObjectBox
from langchain_core.prompts import ChatPromptTemplate
from utils import local_llm, huggingface_instruct_embedding

st.set_page_config(layout='wide', page_title="Objectbox and Langchain")

st.title('Objectbox VectorstoreDB with LLaMA3')

prompt = ChatPromptTemplate.from_template(
    """
    
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    </context>
    Questions: {input}

    """
)

# function for vector embedding and Objectbox VectorstoreDB
def vector_embedding():

    if 'vectors' not in st.session_state:
        # Clean up old DB to avoid schema conflicts
        db_path = '../objectbox'
        if os.path.exists(db_path):
            shutil.rmtree(db_path)

        with st.spinner('Loading and embedding documents...'):
            st.session_state.embeddings = huggingface_instruct_embedding()
            st.session_state.loader = CSVLoader('../data/IMDB Dataset.csv')
            st.session_state.docs = st.session_state.loader.load()
            # Clean HTML tags from review text
            for doc in st.session_state.docs:
                doc.page_content = doc.page_content.replace('<br /><br />', '\n').replace('<br />', '\n')
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:1000])
            st.session_state.vectors = ObjectBox.from_documents(st.session_state.final_documents, st.session_state.embeddings, embedding_dimensions=768, db_directory=db_path)

        st.write(f'Loaded {len(st.session_state.docs)} docs, split into {len(st.session_state.final_documents)} chunks')


if st.button('Embedd Documents'):
    vector_embedding()
    st.write('ObjectBox Database is ready. You can now enter your question')

user_input = st.text_input('Enter your question from documents')


if user_input:
    if 'vectors' not in st.session_state:
        st.error('Please embed documents first by clicking the "Embedd Documents" button.')
        st.stop()
    document_chain = create_stuff_documents_chain(local_llm(), prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()

    response = retrieval_chain.invoke({'input': user_input})
    st.write(response['answer'])
    st.write(f'response time: {(time.process_time() - start):.2f} secs')


     # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")