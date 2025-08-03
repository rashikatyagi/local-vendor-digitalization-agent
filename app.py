# app.py - Final Working Version

import streamlit as st
import os
from dotenv import load_dotenv
from langdetect import detect

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ibm import WatsonxLLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from pinecone import Pinecone

# --- LOAD CREDENTIALS ---
load_dotenv()
# Load all four credentials now
WATSONX_API_KEY = st.secrets.get("WATSONX_API_KEY", os.getenv("WATSONX_API_KEY"))
WATSONX_PROJECT_ID = st.secrets.get("WATSONX_PROJECT_ID", os.getenv("WATSONX_PROJECT_ID"))
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", os.getenv("PINECONE_API_KEY"))
PINECONE_INDEX_NAME = st.secrets.get("PINECONE_INDEX_NAME", os.getenv("PINECONE_INDEX_NAME"))
WATSONX_URL = st.secrets.get("WATSONX_URL", os.getenv("WATSONX_URL"))

# --- CORE RAG FUNCTIONS ---
@st.cache_resource
def initialize_components():
    # ...
    # 1. Initialize the LLM
    llm = WatsonxLLM(
        model_id="ibm/granite-13b-instruct-v2",
        url=WATSONX_URL,  # <-- CHANGE THIS LINE
        project_id=WATSONX_PROJECT_ID,
        apikey=WATSONX_API_KEY,
        params={"max_new_tokens": 1024}
    )

    # 2. Initialize embeddings model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # 3. Initialize Pinecone vector store and retriever
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
        vectorstore = PineconeVectorStore(index, embeddings, "text")
        retriever = vectorstore.as_retriever()
    except Exception as e:
        st.error(f"Failed to connect to Pinecone: {e}")
        return None, None
    
    return llm, retriever

# --- STREAMLIT USER INTERFACE ---

st.set_page_config(page_title="Street Vendor Digitalization Agent", page_icon="🤖")
st.title("Street Vendor Digitalization Agent 🤖")
st.write("Welcome! Ask a specific question or describe your business for a digitalization plan.")

# Initialize the pipeline components
llm, retriever = initialize_components()

if llm and retriever:
    st.success("AI assistant is ready.")

    user_question = st.text_input("Ask your question here:", placeholder="e.g., I sell mangoes in Ghaziabad")

    if user_question:
        with st.spinner("Thinking..."):
            # Step 1: Manually retrieve relevant documents
            retrieved_docs = retriever.get_relevant_documents(user_question)
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            # Detect language for the response
            detected_language = detect(user_question)
            
            # Step 2: Create a dynamic prompt and an LLMChain
            prompt_template = f"""
            You are the "Street Vendor Digitalization Agent." Your goal is to provide a complete, actionable plan for the user based on their statement and the provided Context.
            
            **Rules:**
            1. Your final answer must be written exclusively in the following language: **{detected_language}**.
            2. Use the information from the Context to create a detailed, structured response.
            3. If the user asks a specific question, answer it directly. If they make a statement about their business, create a full digitalization plan with headings.
            4. If the Context doesn't have the answer, state that the information is not available.

            **Context:**
            {context}

            **User's Statement/Question:**
            "{user_question}"

            **Your Detailed Answer:**
            """
            
            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "user_question"])
            chain = LLMChain(llm=llm, prompt=prompt)

            # Step 3: Invoke the chain with the retrieved context
            result = chain.invoke({
                "context": context,
                "user_question": user_question
            })

            # Display the answer and sources
            st.subheader("Your Digitalization Plan:")
            st.markdown(result["text"]) # The output from LLMChain is in the "text" key

            with st.expander("Show Sources Used"):
                for doc in retrieved_docs:
                    st.write(f"- {doc.metadata.get('source', 'Unknown source')}")
else:
    st.error("There was a problem initializing the AI assistant. Please check configurations.")