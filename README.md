# 📚 RAG AI Bot: PDF QA + MCQ Generator (Local Embeddings + Groq LLM)

An interactive Streamlit app that performs Retrieval-Augmented Generation (RAG) over academic PDFs. It allows users to upload multiple research papers, ask questions about their contents, and generate multiple-choice questions (MCQs) — all powered by **local embeddings** and **Groq's lightning-fast LLMs**.

---

## 🚀 Features

- 📄 Upload and analyze **multiple PDF documents**
- 🧠 Perform **semantic search** with **local sentence embeddings**
- 🔍 Ask custom questions based on document content
- ✏️ Generate **MCQs** from the papers using Groq's **LLaMA 3** model
- ⚡ Fully offline for embeddings, with blazing-fast cloud-based generation via Groq

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit** – UI
- **PyMuPDF (fitz)** – PDF parsing
- **LangChain** – Text splitting
- **Sentence Transformers** – Local embeddings (`all-MiniLM-L6-v2`)
- **FAISS** – Vector index
- **Groq API** – LLMs (e.g., `llama3-70b-8192`)
- **OpenAI SDK (>=1.0)** – For Groq API compatibility

---

## 📦 Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/vishalbanwari26/rag-pdf-bot.git
cd rag-pdf-bot
```

### 2. Setup requirements
pip install -r requirements.txt

### 3. Setup API Keys in .env
GROQ_API_KEY=your_groq_key_here

### 4. Run the app
streamlit run app.py


