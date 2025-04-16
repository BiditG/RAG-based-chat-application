# 🧠 Intelligent MCQ Generator from PDF

This application leverages **Retrieval-Augmented Generation (RAG)** and a locally hosted **Large Language Model (LLM)** to automatically generate high-quality, multiple choice questions (MCQs) from uploaded PDF documents. Built with [Streamlit](https://streamlit.io/), it offers an interactive interface for reviewing and validating MCQs derived from the content of any academic or informational text.

---

## 🚀 Features

- **PDF Upload**: Upload academic or professional documents in PDF format.
- **Document Processing**: Automatically chunk and embed documents using vector representations.
- **Semantic Retrieval**: Retrieve the most relevant sections using semantic search.
- **MCQ Generation**: Generate 15 structured MCQs using a locally hosted LLM.
- **Interactive UI**: Answer, check correctness, and review feedback for each question.
- **Final Scoring**: Calculate and display total correct responses.
- **Export**: Download generated MCQs as a `.txt` file for offline use.

---

## 🧰 Tech Stack

| Component           | Role                                                        |
|---------------------|-------------------------------------------------------------|
| **Streamlit**        | Frontend interface for user interaction                     |
| **Ollama**           | Local LLM runner (using `llama3.2:3b` model)                |
| **ChromaDB**         | Local vector database for embedding & retrieval             |
| **LangChain**        | Document loading and chunking                              |
| **PyMuPDF**          | PDF text extraction                                        |
| **Sentence Transformers** | CrossEncoder for re-ranking document relevance       |

---

## 📦 Installation & Setup

1. Clone the Repository
git clone https://github.com/yourusername/mcq-generator-app.git
cd mcq-generator-app

3. Set Up Python Environment
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate


3. Install Required Dependencies
streamlit
chromadb
ollama
langchain
langchain-community
pymupdf
sentence-transformers

4. Install and Run Ollama Locally
Make sure Ollama is installed and running on your machine.
ollama serve
ollama pull llama3.2:3b
ollama pull nomic-embed-text
5. Launch the Application
bash
Copy
Edit
streamlit run main.py
Once the app is running, access it at http://localhost:8501
