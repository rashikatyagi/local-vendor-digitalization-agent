# app.py - Final Version

import streamlit as st
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ibm import WatsonxLLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from pinecone import Pinecone

# # --- LOAD CREDENTIALS ---
# # Load credentials from .env file for local development
# load_dotenv()
# WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
# WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# # For cloud deployment, use Streamlit's secrets management
# if not WATSONX_API_KEY:
#     WATSONX_API_KEY = st.secrets["WATSONX_API_KEY"]
#     WATSONX_PROJECT_ID = st.secrets["WATSONX_PROJECT_ID"]
#     PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
#     PINECONE_INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]

# --- LOAD CREDENTIALS ---
load_dotenv()
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# --- DEBUGGING -- Add these lines ---
print(f"WATSONX_API_KEY: {WATSONX_API_KEY}")
print(f"WATSONX_PROJECT_ID: {WATSONX_PROJECT_ID}")
print(f"PINECONE_API_KEY: {PINECONE_API_KEY}")
print(f"PINECONE_INDEX_NAME: {PINECONE_INDEX_NAME}")
# --- END DEBUGGING ---



# --- CORE RAG FUNCTIONS ---

# Use Streamlit's caching to store the retriever for performance
@st.cache_resource
def initialize_rag_pipeline():
    """
    Initializes the entire RAG pipeline: loads docs, builds the vector store,
    and creates the retrieval chain.
    """
    # 1. Load documents
    # 1. Load documents
    loader = PyPDFDirectoryLoader("knowledge_base/")
    documents = loader.load()

    # 2. Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # 3. Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 4. Initialize Pinecone and create the vector store
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    vectorstore = PineconeVectorStore(index, embeddings, "text")
    vectorstore.add_documents(docs) # This will add documents if they don't exist

    # 5. Initialize the LLM
    llm = WatsonxLLM(
        model_id="ibm/granite-3-8b-instruct",
        url="https://au-syd.ml.cloud.ibm.com", # Ensure this matches your project region
        project_id=WATSONX_PROJECT_ID,
        apikey=WATSONX_API_KEY,
        params={"max_new_tokens": 1024}
    )

    # 6. Create the prompt template
    prompt_template = """
    Use the provided context to answer the user's question accurately and concisely.
    Your response should directly answer the question and nothing more. Do not add extra questions or information.
    If the context does not contain the answer, state that the information is not available.

    Context: {context}
    Question: {question}
    Answer:
    """
    QA_PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # 7. Create the RetrievalQA chain
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True
    )
    
    return qa_chain

# --- STREAMLIT USER INTERFACE ---

st.set_page_config(page_title="Street Vendor Digitalization Agent", page_icon="🤖")
st.title("Street Vendor Digitalization Agent 🤖")
st.write(
    "Welcome! I am here to help local hawkers and micro-entrepreneurs get online. "
    "Ask me how to set up UPI, get registered, or get tips on growing your business digitally."
)

try:
    # Initialize the RAG chain
    chain = initialize_rag_pipeline()
    st.success("AI assistant is ready.")

    # Get user input
    user_question = st.text_input("Ask your question here:", placeholder="e.g., How do I register my business?")

    if user_question:
        with st.spinner("Generating answer..."):
            # Invoke the chain with the user's question
            result = chain.invoke({"query": user_question})

            # Display the answer
            st.subheader("Answer:")
            st.write(result["result"])

            # Display the sources
            with st.expander("Show Sources"):
                st.write("The answer was generated based on the following documents:")
                for doc in result["source_documents"]:
                    st.write(f"- {doc.metadata.get('source', 'Unknown source')}")

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.error("There was a problem initializing the AI assistant. Please check the configurations.")