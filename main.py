import os
import re
import tempfile
import io

import chromadb
import ollama
import streamlit as st
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from streamlit.runtime.uploaded_file_manager import UploadedFile

# ---------------------------- PROMPT FOR MCQ GENERATION ----------------------------
system_prompt = """
You are an AI assistant that generates multiple choice questions (MCQs) from a given academic or informational text.

Instructions:
1. Analyze the provided context and extract key information.
2. Create 15 well-structured MCQs based solely on the content.
3. Each MCQ should include:
   - A clear question
   - Four options (A–D)
   - One correct answer indicated

Format the output as:

Question 1:
What is the main purpose of X?
A. Option one
B. Option two
C. Option three
D. Option four
Answer: B

Question 2:
...
"""

# ---------------------------- UTILITIES ----------------------------
def parse_mcqs(raw_mcq_text):
    questions = re.findall(r"(Question \d+:\n.*?Answer: [A-D])", raw_mcq_text, re.DOTALL)
    parsed_mcqs = []
    for q_block in questions:
        lines = q_block.strip().split("\n")
        question_title = lines[0].strip()
        question_text = lines[1].strip()
        options = lines[2:6]
        answer_line = lines[-1].strip()
        correct_answer = answer_line.split(": ")[1]
        parsed_mcqs.append({
            "title": question_title,
            "question": question_text,
            "options": options,
            "answer": correct_answer
        })
    return parsed_mcqs

def process_document(uploaded_file: UploadedFile) -> list[Document]:
    with tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    loader = PyMuPDFLoader(temp_file_path)
    docs = loader.load()
    os.unlink(temp_file_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    return text_splitter.split_documents(docs)

def get_vector_collection() -> chromadb.Collection:
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest",
    )
    chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")
    return chroma_client.get_or_create_collection(
        name="rag_app",
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"},
    )

def add_to_vector_collection(all_splits: list[Document], file_name: str):
    collection = get_vector_collection()
    documents, metadatas, ids = [], [], []
    for idx, split in enumerate(all_splits):
        documents.append(split.page_content)
        metadatas.append(split.metadata)
        ids.append(f"{file_name}_{idx}")
    collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
    st.toast("✅ Document embedded and stored in vector DB!", icon="📦")

def query_collection(prompt: str, n_results: int = 10):
    collection = get_vector_collection()
    results = collection.query(query_texts=[prompt], n_results=n_results)
    return results

def call_llm(context: str, prompt: str):
    response = ollama.chat(
        model="llama3.2:3b",
        stream=True,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {context}, Question: {prompt}"},
        ],
    )
    for chunk in response:
        if not chunk["done"]:
            yield chunk["message"]["content"]
        else:
            break

def re_rank_cross_encoders(documents: list[str]) -> tuple[str, list[int]]:
    relevant_text = ""
    relevant_text_ids = []
    encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    ranks = encoder_model.rank("generate MCQs", documents, top_k=3)
    for rank in ranks:
        relevant_text += documents[rank["corpus_id"]]
        relevant_text_ids.append(rank["corpus_id"])
    return relevant_text, relevant_text_ids

# ---------------------------- STREAMLIT UI ----------------------------
st.set_page_config(page_title="🧠 Smart MCQ Generator", layout="wide")
st.title("📘 Intelligent MCQ Generator from PDF")
st.caption("Upload your PDF and generate interactive multiple choice questions using AI")

if "mcqs" not in st.session_state:
    st.session_state.mcqs = []
    st.session_state.response_text = ""

with st.sidebar:
    st.subheader("📂 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    process = st.button("⚡ Process PDF")
    if uploaded_file and process:
        file_key = uploaded_file.name.translate(str.maketrans({"-": "_", ".": "_", " ": "_"}))
        all_splits = process_document(uploaded_file)
        add_to_vector_collection(all_splits, file_key)

st.divider()
st.subheader("🎓 Step 2: Generate & Review Questions")
generate = st.button("🧠 Generate MCQs")

if generate:
    results = query_collection("generate MCQs")
    context = results.get("documents")[0]
    relevant_text, _ = re_rank_cross_encoders(context)

    with st.spinner("Generating MCQs using AI..."):
        st.session_state.response_text = ""
        for chunk in call_llm(context=relevant_text, prompt="Generate 15 multiple choice questions"):
            st.session_state.response_text += chunk

    st.session_state.mcqs = parse_mcqs(st.session_state.response_text)
    st.success("✅ MCQs generated successfully!")

if st.session_state.mcqs:
    st.markdown("### 📖 Review Your Questions")
    score = 0
    for i, mcq in enumerate(st.session_state.mcqs):
        with st.expander(f"{mcq['title']}: {mcq['question']}"):
            user_choice = st.radio("Choose the correct answer:", mcq["options"], key=f"q{i}")
            if st.button("Check Answer", key=f"check{i}"):
                if user_choice:
                    if user_choice.startswith(mcq["answer"]):
                        st.success("✅ Correct!")
                        score += 1
                    else:
                        st.error(f"❌ Incorrect. Correct answer is: {mcq['answer']}")
                else:
                    st.warning("⚠️ Please select an option before checking.")

    st.markdown(f"### 🏁 Final Score: {score} / {len(st.session_state.mcqs)}")
    st.download_button("💾 Download All MCQs", data=st.session_state.response_text, file_name="mcqs.txt")

    with st.expander("📂 See Retrieved Context Chunks"):
        st.write(st.session_state.response_text)
