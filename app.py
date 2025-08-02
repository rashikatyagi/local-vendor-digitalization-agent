import streamlit as st
import os
from dotenv import load_dotenv

# --- SETUP ---
# Load credentials from the .env file
load_dotenv()

# Get the credentials from environment variables
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_HOST = os.getenv("PINECONE_HOST")

# --- UI (User Interface) ---
# Set the page title and icon
st.set_page_config(page_title="Street Vendor Digitalization Agent", page_icon="🤖")

# Display the header
st.header("Street Vendor Digitalization Agent 🤖")
st.write("This tool helps you create a digital profile for your business. Just tell me what you sell and where!")

# A simple check to confirm that the API keys are loaded
if WATSONX_API_KEY and PINECONE_API_KEY and WATSONX_PROJECT_ID and PINECONE_HOST:
    st.success("API keys loaded successfully! Ready to proceed.")
else:
    st.error("One or more API keys are missing. Please check your .env file.")