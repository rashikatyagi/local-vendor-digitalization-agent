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
WATSONX_API_KEY = st.secrets.get("WATSONX_API_KEY", os.getenv("WATSONX_API_KEY"))
WATSONX_PROJECT_ID = st.secrets.get("WATSONX_PROJECT_ID", os.getenv("WATSONX_PROJECT_ID"))
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", os.getenv("PINECONE_API_KEY"))
WATSONX_URL = st.secrets.get("WATSONX_URL", os.getenv("WATSONX_URL"))
# We will select the index name dynamically
PINECONE_INDEX_NAME_EN = "street-vendor-agent"
PINECONE_INDEX_NAME_HI = "street-vendor-agent-hi"

# --- CORE RAG FUNCTIONS ---

@st.cache_resource
def get_llm():
    """Initializes and returns the LLM."""
    return WatsonxLLM(
        model_id="ibm/granite-13b-instruct-v2",
        url=WATSONX_URL,
        project_id=WATSONX_PROJECT_ID,
        apikey=WATSONX_API_KEY,
        params={"decoding_method": "sample", "max_new_tokens": 1024, "temperature": 0.7, "repetition_penalty": 1.2}
    )

@st.cache_resource
def get_retriever(language="en"):
    """Initializes and returns a retriever for a specific language."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    if language == "hi":
        folder_path = "knowledge_base/hi/"
        index_name = PINECONE_INDEX_NAME_HI
    else: # Default to English
        folder_path = "knowledge_base/en/"
        index_name = PINECONE_INDEX_NAME_EN

    try:
        # Load and process documents
        loader = PyPDFDirectoryLoader(folder_path) # Simplified to PDF for now
        documents = loader.load()
        if not documents: return None # Return None if no documents are found
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        
        # Initialize Pinecone and add documents
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(index_name)
        vectorstore = PineconeVectorStore(index, embeddings, "text")
        vectorstore.add_documents(docs) # This will create embeddings and upload them
        
        return vectorstore.as_retriever()
    except Exception as e:
        st.error(f"Failed to build knowledge base for language '{language}': {e}")
        return None

# --- STREAMLIT USER INTERFACE ---

st.set_page_config(page_title="Street Vendor Digitalization Agent", page_icon="🤖")
st.title("Street Vendor Digitalization Agent 🤖")
st.write("Welcome! Ask a specific question or describe your business for a digitalization plan.")

llm = get_llm()

if llm:
    st.success("AI assistant is ready.")
    user_question = st.text_input("Ask your question here:", placeholder="e.g., I sell mangoes in Ghaziabad")

    if user_question:
        with st.spinner("Thinking..."):
            detected_language = detect(user_question)
            
            retriever = get_retriever(language=detected_language)
            
            if retriever:
                retrieved_docs = retriever.get_relevant_documents(user_question)
                context = "\n\n".join([doc.page_content for doc in retrieved_docs])

                # Simple, language-specific prompts
                if detected_language == 'hi':
                    prompt_template_str = """
                    दिए गए "कॉन्टेक्स्ट" का उपयोग करके "प्रश्न" का उत्तर दें। उत्तर केवल हिंदी में होना चाहिए।

                    कॉन्टेक्स्ट:
                    {context}

                    प्रश्न:
                    "{user_question}"

                    उत्तर:
                    """
                else: # English prompt
                    prompt_template_str = """
                    Use the "Context" to answer the "Question". The answer must be in English.

                    Context:
                    {context}

                    Question:
                    "{user_question}"

                    Answer:
                    """
                
                prompt = PromptTemplate(template=prompt_template_str, input_variables=["context", "user_question"])
                chain = LLMChain(llm=llm, prompt=prompt)

                result = chain.invoke({"context": context, "user_question": user_question})

                st.subheader("Your Digitalization Plan:")
                st.markdown(result["text"])

                with st.expander("Show Sources Used"):
                    for doc in retrieved_docs:
                        st.write(f"- {doc.metadata.get('source', 'Unknown source')}")
            else:
                st.error(f"Could not prepare the knowledge base for the detected language: '{detected_language}'. Please ensure documents exist in the correct folder.")
else:
    st.error("There was a problem initializing the AI assistant.")
