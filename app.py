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
@st.cache_resource
def initialize_rag_pipeline():
    """
    Initializes and returns the core components: the LLM and the Vector Store.
    """
    # --- DEBUGGING -- Add these lines ---
    print("--- Initializing WatsonxLLM with the following credentials ---")
    print(f"URL: {os.getenv('WATSONX_API_URL', 'Not Found')}") # Use a default value for printing
    print(f"API Key is present: {bool(os.getenv('WATSONX_API_KEY'))}")
    print(f"Project ID: {os.getenv('WATSONX_PROJECT_ID')}")
    # --- END DEBUGGING ---

    # 1. Initialize the LLM
    llm = WatsonxLLM(
        model_id="ibm/granite-3-8b-instruct",
        url=os.getenv("WATSONX_API_URL"), # Make sure this key exists in your secrets
        project_id=os.getenv("WATSONX_PROJECT_ID"),
        apikey=os.getenv("WATSONX_API_KEY"),
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

# --- STREAMLIT USER INTERFACE ---

st.set_page_config(page_title="Street Vendor Digitalization Agent", page_icon="🤖")
st.title("Street Vendor Digitalization Agent 🤖")
st.write(
    "Welcome! I am here to help local hawkers and micro-entrepreneurs. "
    "Ask your question in English or Hindi."
)

# Initialize the RAG pipeline components
llm, vectorstore = initialize_rag_pipeline()

if llm and vectorstore:
    st.success("AI assistant is ready.")

    user_question = st.text_input("Ask your question here:", placeholder="e.g., How do I register my business?")

    if user_question:
        with st.spinner("Generating answer..."):
            # 1. Detect the language of the user's question
            detected_language = detect(user_question)
            
            # 2. Create the final, most robust prompt template
            prompt_template = f"""
            Follow these rules precisely to answer the user's question based on the provided Context.
            1. Your final answer must be written exclusively in the following language: **{detected_language}**.
            2. Use only the information from the Context. Do not use outside knowledge.
            3. Answer only the user's question and then stop. Do not add any extra questions, conversation, or text.
            4. If the answer is not in the Context, simply state that the information is not available in that language.

            Context: {{context}}
            Question: {{question}}
            Answer:
            """
            
            QA_PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            
            # 3. Get the retriever from your vector store
            retriever = vectorstore.as_retriever()
            
            # 4. Create the full RAG chain with the dynamic prompt
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type_kwargs={"prompt": QA_PROMPT},
                return_source_documents=True
            )
            
            # 5. Invoke the chain to get the result
            result = qa_chain.invoke({"query": user_question})

            # Display the answer and sources
            st.subheader("Answer:")
            st.write(result["result"])

            with st.expander("Show Sources"):
                st.write("The answer was generated based on the following documents:")
                for doc in result["source_documents"]:
                    st.write(f"- {doc.metadata.get('source', 'Unknown source')}")
else:
    st.error("There was a problem initializing the AI assistant. Please check configurations.")