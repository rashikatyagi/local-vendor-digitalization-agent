# 🧺 Street Vendor Digitalization Agent

An AI-powered conversational agent that generates **actionable digitalization plans** for street vendors and micro-entrepreneurs in India, leveraging a **Retrieval-Augmented Generation (RAG)** architecture with **IBM's Granite LLM**.

---

## 🔴 Live Demo

👉 [Click here to access the live Streamlit app][https://your-streamlit-url.com](https://local-vendor-digitalization-agent-wvibtnfdhg6kfy4wsmjrwd.streamlit.app] 

---

## 🎯 Problem Statement

India's vibrant community of street vendors — the backbone of its informal economy — often faces **exclusion from the digital revolution** due to:

- Technological complexity  
- Low awareness of government schemes  
- Language barriers  

This project bridges that **digital divide** by offering an **accessible AI tool** that guides vendors toward digitalization — step by step, in plain language.

---

## ✨ Key Features

- **🗣 Conversational Interface**: Simply say "I sell mangoes in Ghaziabad" or ask any question.  
- **📄 Structured Plan Generation**: Get a full roadmap covering business profiles, marketing, digital payments, etc.  
- **🏛 Scheme Awareness**: Highlights schemes like **PM SVANidhi**, pulled from a custom knowledge base.  
- **🔍 Factual Answers**: Uses **RAG** to ground answers in trusted documents — minimizing hallucinations.

---

## 🧠 Core Architecture & Novel Effects

At its core is a **RAG pipeline**, enhanced with a **persona-driven prompting strategy**.

### 🔗 Architecture Flow

1. **User Input → Embedding**  
2. **Pinecone → Relevant Context Retrieval**  
3. **Persona-Driven Prompt Creation**  
4. **IBM Granite LLM → Structured Response Generation**

### 💡 What’s Novel?

- **Persona-Based Prompting**: The model is prompted to act as a _"Digitalization Agent"_.
- If a vendor says: `"I sell mangoes"`, the model turns that into a **task**:  
  _“Generate a full digitalization plan for this user.”_  
- This makes it proactive — not just a passive Q&A bot.

---

## 🛠️ Technology Stack

| Component             | Tool/Platform                                     |
|-----------------------|--------------------------------------------------|
| Cloud Platform        | IBM Cloud (Watsonx.ai)                           |
| LLM                   | IBM Granite (`granite-13b-instruct-v2`)          |
| Vector DB             | Pinecone                                          |
| AI Framework          | Python, LangChain                                 |
| Embeddings            | Hugging Face (`all-MiniLM-L6-v2`)                |
| UI & Deployment       | Streamlit, Streamlit Community Cloud, GitHub     |

---

## ⚙️ Setup and Installation

### 1. Prerequisites

- [x] Git  
- [x] Python 3.11+  
- [x] Conda or any virtual environment tool

### 2. Clone the Repo

```bash
git clone https://github.com/your-username/local-vendor-digitalization-agent.git
cd local-vendor-digitalization-agent
```

### 3. Create and Activate Virtual Environment

```bash
conda create --name vendoragent_env python=3.11 -y
conda activate vendoragent_env
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```
### 5. Add Environment Variables
Create a .env file in the root folder and add:

```bash
WATSONX_API_KEY="your_ibm_cloud_api_key_here"
WATSONX_PROJECT_ID="your_watsonx_project_id_here"
WATSONX_URL="your_ibm_watsonx_region_url_here"
PINECONE_API_KEY="your_pinecone_api_key_here"
PINECONE_INDEX_NAME="your_pinecone_index_name_here"
```

### 6. Run the App

```bash
streamlit run app.py
```

## 🚀 Future Scope
- 🎙️ Voice Integration: Speech-to-text & text-to-speech support
- 💬 WhatsApp Chatbot: Conversational access via WhatsApp
- 🗺️ Hyper-local Content: Expand to city-specific tips & schemes
- 🎨 Visual Asset Generator: QR code posters, basic logos, etc.
