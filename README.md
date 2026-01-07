# ABC Grocery AI Assistant (RAG)

A production-style **Retrieval-Augmented Generation (RAG)** chatbot built as a portfolio project.

This application demonstrates how to design a **grounded, hallucination-safe AI assistant** using a vector database, a modern LLM pipeline, and a clean, realistic web interface.

---

## ✨ Key Features

- Retrieval-Augmented Generation (RAG)
- Strict context grounding (no hallucinations)
- Vector similarity search using ChromaDB
- Session-based conversational memory
- Floating chatbot UI with typing indicator
- Responsive design (desktop & mobile)

---

## 🧠 System Architecture

User (Browser)
↓
Flask Web App
↓
LangChain RAG Pipeline
↓
Vector Retriever (ChromaDB)
↓
OpenAI LLM
↓
Response


The assistant is designed to **only answer using retrieved context**.  
If the required information is not found, a predefined fallback response is returned.

---

## 🛠 Technology Stack

- **Backend:** Python, Flask, Gunicorn  
- **LLM Framework:** LangChain  
- **Vector Store:** ChromaDB  
- **LLM Provider:** OpenAI API  
- **Frontend:** HTML, CSS, JavaScript  

---

## 🧠 Chat Memory & History

This project includes **session-based conversational memory** using LangChain message history.

- Memory is maintained **per browser session**
- The assistant can handle follow-up questions naturally within the same session

**Note:**  
Chat UI history is **not persisted across page refresh or server restart**.  
This is an intentional design decision for this portfolio project.

Planned future options include:
- Client-side persistence using browser storage (localStorage)
- Server-side persistence using a database or Redis
- Long-term semantic memory using a vector database (e.g., Pinecone)

---

## ▶️ Running the Project Locally

### 1. Install dependencies
Run the following command in your terminal:

pip install -r requirements.txt

### 2. Create a .env file
In the project root directory, create a file named .env and add the following variables:

OPENAI_API_KEY=your_openai_api_key
FLASK_SECRET_KEY=your_secret_key

### 3. Run the project in development mode
Start the application using:

python app.py

### 4. Run the project in production-style mode
To run the app with a production-ready server, use:

gunicorn app:app
