# app.py - Final Corrected Version

import streamlit as st
import os
from dotenv import load_dotenv
from langdetect import detect

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ibm import WatsonxLLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from pinecone import Pinecone

# --- LOAD CREDENTIALS ---
# This section remains the same
load_dotenv()
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# --- CORE RAG FUNCTIONS ---

@st.cache_resource
def initialize_rag_pipeline():
    """
    Initializes and returns the core components: the LLM and the Vector Store.
    """
    # 1. Initialize the LLM
    llm = WatsonxLLM(
        model_id="ibm/granite-3-8b-instruct",
        url="https://au-syd.ml.cloud.ibm.com", 
        apikey=WATSONX_API_KEY,
        params={"max_new_tokens": 1024}
    )

    # 2. Initialize embeddings model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # 3. Load documents and build the vector store
    try:
        loader = PyPDFDirectoryLoader("knowledge_base/")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        
        # Initialize Pinecone and add documents
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
        vectorstore = PineconeVectorStore(index, embeddings, "text")
        vectorstore.add_documents(docs)

    except Exception as e:
        st.error(f"Failed to build knowledge base: {e}")
        return None, None
    
    # Return both the llm and vectorstore so they can be used outside this function
    return llm, vectorstore

# --- STREAMLIT USER INTERFACE ---

st.set_page_config(page_title="Street Vendor Digitalization Agent", page_icon="🤖")
st.title("Street Vendor Digitalization Agent 🤖")
st.write(
    "Welcome! I am here to help local hawkers and micro-entrepreneurs. "
    "Ask your question in English or Hindi."
)

# Initialize the pipeline
llm, vectorstore = initialize_rag_pipeline()

if llm and vectorstore:
    st.success("AI assistant is ready.")

    user_question = st.text_input("Ask your question here:", placeholder="e.g., How do I register my business?")

    if user_question:
        with st.spinner("Generating answer..."):
            detected_language = detect(user_question)
            
            prompt_template = f"""
            You are an assistant for local street vendors. Your primary goal is to answer the user's question based ONLY on the provided context.
            Follow these rules strictly:
            1. Formulate your final answer exclusively in the following language: **{detected_language}**.
            2. Use only the information given in the 'Context' below. Do not use any other knowledge.
            3. If the context does not contain the answer, state that the information is not available in that language.
            4. Do not add any extra questions, conversations, or information beyond the direct answer.

            Context: {{context}}
            Question: {{question}}
            Answer:
            """
            
            QA_PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            
            retriever = vectorstore.as_retriever()
            
            qa_chain = RetrievalQA.from_chain_type(
                llm,
                retriever=retriever,
                chain_type_kwargs={"prompt": QA_PROMPT},
                return_source_documents=True
            )
            
            result = qa_chain.invoke({"query": user_question})

            st.subheader("Answer:")
            st.write(result["result"])

            with st.expander("Show Sources"):
                st.write("The answer was generated based on the following documents:")
                for doc in result["source_documents"]:
                    st.write(f"- {doc.metadata.get('source', 'Unknown source')}")
else:
    st.error("There was a problem initializing the AI assistant. Please check configurations.")