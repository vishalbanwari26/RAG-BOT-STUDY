import streamlit as st
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
from dotenv import load_dotenv
import os
import time
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import random

# --- Load environment variables ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Groq Client ---
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY
)

# --- Load local embedding model ---
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Initial Setup ---
st.set_page_config(page_title="RAG PDF QA Bot", layout="wide")

st.title("📚 RAG AI Bot: PDF Question Answering & MCQ Generator")

# --- PDF Upload ---
uploaded_files = st.file_uploader("Upload one or more PDF files", type=["pdf"], accept_multiple_files=True)

# --- Text Extraction ---
def extract_text(pdf_file):
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        full_text = "\n".join([page.get_text() for page in doc if page.get_text().strip()])
        return full_text
    except Exception as e:
        st.error("Error extracting text from PDF.")
        st.text(str(e))
        return ""

# --- Embedding using local model ---
def embed_texts(texts):
    try:
        return np.array(embed_model.encode(texts))
    except Exception as e:
        st.error("Embedding error")
        st.text(str(e))
        st.stop()

# --- Text generation using Groq API (LLaMA 3) ---
def generate_text(prompt, max_tokens=300, temperature=0.7):
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error("Groq API error")
        st.text(str(e))
        return ""


# --- Store chunks & embeddings ---
all_chunks = []
chunk_text_lookup = []

if uploaded_files:
    st.success(f"{len(uploaded_files)} PDF(s) uploaded. Processing...")

    for file in uploaded_files:
        full_text = extract_text(file)
        st.write(f"Total words in extracted text: {len(full_text.split())}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = text_splitter.split_text(full_text)
        all_chunks.extend(chunks)
        chunk_text_lookup.extend(chunks)

    st.info(f"Total chunks created: {len(all_chunks)}")

    # Embed and index
    st.write("Embedding and indexing text locally...")
    embeddings = embed_texts(all_chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    st.success("Ready for questions and MCQ generation!")

    # --- Question Answering ---
    st.header("🔎 Ask a Question")
    user_question = st.text_input("Enter your question:")

    if user_question:
        question_emb = embed_texts([user_question])
        D, I = index.search(np.array(question_emb), k=5)

        MAX_WORDS = 2250
        selected_chunks = []
        word_count = 0
        for i in I[0]:
            chunk = chunk_text_lookup[i]
            chunk_words = len(chunk.split())
            if word_count + chunk_words > MAX_WORDS:
                break
            selected_chunks.append(chunk)
            word_count += chunk_words

        context = "\n\n".join(selected_chunks)

        prompt = f"""You are an AI assistant tasked with answering questions about a collection of research papers.

Below is the combined content extracted from {len(uploaded_files)} document(s). Use only this information to answer the user's question clearly and accurately.

Content:
{context}

Question: {user_question}"""

        answer = generate_text(prompt, max_tokens=300)
        st.markdown("**Answer:**")
        st.write(answer)

    # --- MCQ Generation ---
    st.header("📝 Generate MCQs")
    generate = st.button("Generate MCQs")

    if generate:
        MAX_WORDS = 2250
        selected_chunks = []
        word_count = 0
        shuffled_chunks = chunk_text_lookup.copy()
        random.shuffle(shuffled_chunks)
        for chunk in chunk_text_lookup:
            chunk_words = len(chunk.split())
            if word_count + chunk_words > MAX_WORDS:
                break
            selected_chunks.append(chunk)
            word_count += chunk_words


        context_for_mcq = "\n\n".join(selected_chunks)

        mcq_prompt = f"""
You are an AI tutor. Based on the following text from {len(uploaded_files)} research paper(s), generate 3 multiple choice questions with 4 options each and provide the correct answer.

Text:
{context_for_mcq}

Format:
Q1. Question?
a) Option A
b) Option B
c) Option C
d) Option D
Answer: b) Option B
"""

        mcqs = generate_text(mcq_prompt, max_tokens=500, temperature=1.0)
        st.markdown("**MCQs Generated:**")
        st.text(mcqs)
