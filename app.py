# app.py - Final English-Only Version with Improved Retrieval

import streamlit as st
import os
from dotenv import load_dotenv

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
WATSONX_API_KEY = st.secrets.get("WATSONX_API_KEY", os.getenv("WATSONX_API_KEY"))
WATSONX_PROJECT_ID = st.secrets.get("WATSONX_PROJECT_ID", os.getenv("WATSONX_PROJECT_ID"))
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", os.getenv("PINECONE_API_KEY"))
PINECONE_INDEX_NAME = st.secrets.get("PINECONE_INDEX_NAME", os.getenv("PINECONE_INDEX_NAME"))
WATSONX_URL = st.secrets.get("WATSONX_URL", os.getenv("WATSONX_URL"))

# --- CORE RAG FUNCTIONS ---

@st.cache_resource
def initialize_components():
    """
    Initializes and returns the core components: LLM and the document retriever.
    """
    llm = WatsonxLLM(
        model_id="ibm/granite-13b-instruct-v2",
        url=WATSONX_URL,
        project_id=WATSONX_PROJECT_ID,
        apikey=WATSONX_API_KEY,
        params={
            "decoding_method": "sample",
            "max_new_tokens": 1024,
            "temperature": 0.7,
            "repetition_penalty": 1.2
        }
    )

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    try:
        loader = PyPDFDirectoryLoader("knowledge_base/")
        documents = loader.load()
        if not documents:
            st.error("No documents found in the 'knowledge_base' folder.")
            return None, None
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
        vectorstore = PineconeVectorStore(index, embeddings, "text")
        vectorstore.add_documents(docs)
        
        # This is the updated line to retrieve more documents
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    except Exception as e:
        st.error(f"Failed to build knowledge base: {e}")
        return None, None
    
    return llm, retriever

# --- STREAMLIT USER INTERFACE ---

st.set_page_config(page_title="Street Vendor Digitalization Agent", page_icon="🤖")
st.title("Street Vendor Digitalization Agent 🤖")
st.write("Welcome! Ask a specific question or describe your business for a digitalization plan.")

llm, retriever = initialize_components()

if llm and retriever:
    st.success("AI assistant is ready.")
    user_question = st.text_input("Ask your question here:", placeholder="e.g., I sell mangoes in Ghaziabad")

    if user_question:
        with st.spinner("Thinking..."):
            retrieved_docs = retriever.get_relevant_documents(user_question)
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            prompt_template = """
            You are a helpful AI assistant for street vendors in India called the "Street Vendor Digitalization Agent."
            Your goal is to provide a complete, actionable plan based on the user's statement and the provided Context.

            **FOLLOW THESE RULES:**
            1. Use the provided "Context" to create a detailed, structured response in markdown.
            2. If the user asks a specific question, answer it directly. If they make a statement about their business, create a full digitalization plan with headings.
            3. If the Context does not have the answer, state that the information is not available.

            **Context:**
            {context}

            **User's Statement/Question:**
            "{user_question}"

            **Your Detailed Answer:**
            """
            
            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "user_question"])
            chain = LLMChain(llm=llm, prompt=prompt)

            result = chain.invoke({"context": context, "user_question": user_question})

            st.subheader("Your Digitalization Plan:")
            st.markdown(result["text"])

            with st.expander("Show Sources Used"):
                for doc in retrieved_docs:
                    st.write(f"- {doc.metadata.get('source', 'Unknown source')}")
else:
    st.error("There was a problem initializing the AI assistant.")