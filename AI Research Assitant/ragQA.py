# ragQA.py — fixed, drop-in replacement
# Save this exact file as F:\AI Research Assitant\ragQA.py

import streamlit as st
import os
import time
from dotenv import load_dotenv

# Load environment file
load_dotenv()

# Helpful runtime print so you can confirm the file being executed
print("Running ragQA.py (fixed)")

# --- Safe imports with clear error messages ---
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception as e:
    raise ImportError("Install langchain-text-splitters: pip install langchain-text-splitters") from e

# Try multiple paths for create_stuff_documents_chain to be compatible across versions
try:
    from langchain.chains.combine_documents.base import create_stuff_documents_chain
except Exception:
    try:
        from langchain.chains.combine_documents import create_stuff_documents_chain
    except Exception:
        raise ImportError(
            "Could not import create_stuff_documents_chain from langchain. "
            "If you want to run the original old-style code, install a compatible langchain: "
            "pip install langchain==0.1.20 langchain-community==0.0.38 "
            "Otherwise use a fixed script or update imports."
        )

try:
    from langchain.chains.retrieval import create_retrieval_chain
except Exception:
    try:
        from langchain.chains import create_retrieval_chain
    except Exception:
        raise ImportError("Could not import create_retrieval_chain from langchain.")

# Chat prompt import (works across newer/older layouts)
try:
    from langchain_core.prompts import ChatPromptTemplate
except Exception:
    try:
        from langchain.prompts import ChatPromptTemplate
    except Exception:
        raise ImportError("Could not import ChatPromptTemplate from langchain_core or langchain.prompts")

# Community vectorstore / loader
try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import PyPDFDirectoryLoader
except Exception:
    raise ImportError("Missing langchain-community. Install with: pip install langchain-community")

# Groq & HuggingFace adapters
try:
    from langchain_groq import ChatGroq
except Exception:
    raise ImportError("Missing langchain-groq. Install with: pip install langchain-groq")

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    raise ImportError("Missing langchain-huggingface. Install with: pip install langchain-huggingface")

# Load keys from .env
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY', '')
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN', '')

groq_api_key = os.getenv('GROQ_API_KEY')
hf_token = os.getenv('HF_TOKEN')

if not groq_api_key:
    st.warning("GROQ_API_KEY not found in environment. Make sure .env contains GROQ_API_KEY=<your_key>")

if not hf_token:
    st.info("HF_TOKEN not found. Some embeddings may require it.")

# Instantiate embeddings and LLM
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# choose the appropriate model name you have access to
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

prompt = ChatPromptTemplate.from_template(
    """
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Question:{input}
"""
)

st.title("RAG Document Q&A With Groq And Llama3 (fixed ragQA.py)")

user_prompt = st.text_input("Enter your query from the research paper")

def create_vector_embedding(pdf_dir: str = "research_papers"):
    if "vectors" in st.session_state:
        st.info("Vector DB already exists in session.")
        return
    if not os.path.isdir(pdf_dir):
        st.error(f"PDF directory '{pdf_dir}' not found. Create it and place PDFs inside.")
        return
    with st.spinner("Loading PDFs and creating embeddings (this may take a while)..."):
        loader = PyPDFDirectoryLoader(pdf_dir)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_docs = splitter.split_documents(docs[:50])
        st.session_state.vectors = FAISS.from_documents(final_docs, embeddings)
    st.success("Vector DB ready")

if st.button("Document Embedding"):
    create_vector_embedding()

if user_prompt:
    if "vectors" not in st.session_state:
        st.warning("Vector DB not found. Click 'Document Embedding' first to build vectors from PDFs.")
    else:
        try:
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            start = time.process_time()
            response = retrieval_chain.invoke({'input': user_prompt})
            elapsed = time.process_time() - start

            answer = response.get('answer', None)
            if answer:
                st.write(answer)
            else:
                st.write(response)

            st.write(f"Response time: {elapsed:.2f}s")

            with st.expander("Document similarity Search"):
                context_docs = response.get('context') or []
                for i, doc in enumerate(context_docs):
                    st.write(doc.page_content)
                    st.write('------------------------')
        except Exception as e:
            st.error(f"An error occurred while running retrieval chain: {e}")
            st.exception(e)

st.caption("If you get import errors related to langchain, install compatible versions or run the fixed script provided.")
