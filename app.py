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

    # In your main Streamlit UI section, after the user_question input...

    if user_question:
        with st.spinner("Generating answer..."):
            # Detect the language of the user's input
            detected_language = detect(user_question)

            # --- NEW PERSONA-BASED PROMPT ---
            # This prompt gives the AI a role and a structured plan to follow
            prompt_template = f"""
            You are the "Street Vendor Digitalization Agent," a helpful AI assistant for street vendors in India. Your goal is to provide a complete, actionable plan to help them grow their business digitally.

            **User's Business:** "{user_question}"
            **Detected Language for Response:** {detected_language}

            Based on the user's business description and the context provided below, generate a comprehensive business profile and action plan for them. Structure your response with the following sections, and answer in the detected language:

            ### Business Profile Suggestion
            - Create a catchy, one-line business name and description.

            ### Digital Payments (UPI)
            - Provide a step-by-step guide on how to get a UPI QR code.

            ### Online Visibility (Local SEO)
            - Give tips on how to list their business on Google Maps and other local platforms.

            ### Customer Engagement
            - Suggest simple tips for using WhatsApp or social media to connect with customers.
            
            ### Relevant Government Schemes
            - Mention any relevant schemes from the context, like PM SVANidhi, and explain how they can help.

            Use the following context to fill in the details for your plan. If some information is not in the context, provide general, helpful advice for that section.

            Context: {{context}}
            Question: How can the user describing their business as "{user_question}" be helped with digitalization?
            Answer:
            """

            QA_PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

            retriever = vectorstore.as_retriever()

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type_kwargs={"prompt": QA_PROMPT},
            )

            # We re-frame the user's statement as a question for the RAG chain
            result = qa_chain.invoke({
                "query": f"Create a digitalization plan for a user who says: '{user_question}'"
            })
            # --- END OF NEW LOGIC ---

            st.subheader("Your Digitalization Plan:")
            st.markdown(result["result"]) # Use st.markdown to render headings properly

            with st.expander("Show Sources"):
                st.write("The answer was generated based on the following documents:")
                for doc in result["source_documents"]:
                    st.write(f"- {doc.metadata.get('source', 'Unknown source')}")
else:
    st.error("There was a problem initializing the AI assistant. Please check configurations.")