"""
Fixed version of ragquesANSproject.py
- Updated imports to be compatible with recent LangChain layout while keeping original logic.
- Includes helpful error messages and a fallback suggestion to install matching package versions.

INSTRUCTIONS:
1. Save this file as ragquesANSproject_fixed.py in your project folder.
2. Recommended packages (run in your venv):
   pip install streamlit python-dotenv langchain==0.1.20 langchain-community langchain-groq langchain-huggingface langchain-text-splitters faiss-cpu pypdf
   (If you prefer newer langchain, remove the version pin and install langchain-text-splitters etc. but APIs may differ.)
3. Ensure your .env exists and contains GROQ_API_KEY and HF_TOKEN.
4. Run: streamlit run ragquesANSproject_fixed.py

"""

import streamlit as st
import os
import time
from dotenv import load_dotenv

# Try to import the correct text splitter and chain helpers. Provide clear errors if imports fail.
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception as e:
    raise ImportError(
        "Missing package 'langchain-text-splitters'.\nInstall it with: pip install langchain-text-splitters"
    ) from e

# Try to import chain helpers in a way that works across several LangChain releases
try:
    # preferred for some newer versions
    from langchain.chains.combine_documents.base import create_stuff_documents_chain
except Exception:
    try:
        from langchain.chains.combine_documents import create_stuff_documents_chain
    except Exception:
        raise ImportError(
            "Could not import create_stuff_documents_chain from langchain.\n"
            "Either install a compatible langchain version (e.g. pip install langchain==0.1.20) or adjust imports."
        )

try:
    from langchain.chains.retrieval import create_retrieval_chain
except Exception:
    # some versions expose a different module path
    try:
        from langchain.chains import create_retrieval_chain
    except Exception:
        raise ImportError(
            "Could not import create_retrieval_chain from langchain.\n"
            "Consider installing langchain==0.1.20 or check langchain docs for the correct import path."
        )

# Other imports (community / core / adapters)
try:
    from langchain_core.prompts import ChatPromptTemplate
except Exception:
    # fallback to langchain.prompts for some versions
    try:
        from langchain.prompts import ChatPromptTemplate
    except Exception as e:
        raise ImportError("Could not import ChatPromptTemplate from langchain_core or langchain.prompts") from e

try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import PyPDFDirectoryLoader
except Exception as e:
    raise ImportError(
        "Missing langchain-community package or its components.\nInstall with: pip install langchain-community"
    ) from e

# LLM & embeddings adapters
try:
    from langchain_groq import ChatGroq
except Exception:
    raise ImportError("Missing langchain-groq. Install with: pip install langchain-groq")

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    raise ImportError("Missing langchain-huggingface. Install with: pip install langchain-huggingface")

# Load environment
load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY', '')
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN', '')

groq_api_key = os.getenv('GROQ_API_KEY')
if not groq_api_key:
    st.warning("GROQ_API_KEY not found in environment. Make sure .env contains GROQ_API_KEY=<your_key>")

hf_token = os.getenv('HF_TOKEN')
if not hf_token:
    st.info("HF_TOKEN not found. HuggingFace embeddings may fail if not provided (some models require it).")

# instantiate embeddings and LLM
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Prompt template
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

# Streamlit UI
st.title("RAG Document Q&A With Groq And Llama3 (fixed)")

user_prompt = st.text_input("Enter your query from the research paper")

# Helper: create vector DB from PDFs
def create_vector_embedding(pdf_dir: str = "research_papers"):
    if "vectors" not in st.session_state:
        if not os.path.isdir(pdf_dir):
            st.error(f"PDF directory '{pdf_dir}' not found. Create it and place PDFs inside.")
            return
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.loader = PyPDFDirectoryLoader(pdf_dir)
        with st.spinner("Loading PDFs and creating embeddings (this may take a while)..."):
            st.session_state.docs = st.session_state.loader.load()
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        st.success("Vector DB ready")

if st.button("Document Embedding"):
    create_vector_embedding()

if user_prompt:
    if "vectors" not in st.session_state:
        st.warning("Vector DB not found. Click 'Document Embedding' first to build vectors from PDFs.")
    else:
        try:
            # Build chains and run retrieval
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            start = time.process_time()
            response = retrieval_chain.invoke({'input': user_prompt})
            elapsed = time.process_time() - start

            st.write(response.get('answer', response))
            st.write(f"Response time: {elapsed:.2f}s")

            with st.expander("Document similarity Search"):
                # response['context'] may be present depending on chain implementation
                context_docs = response.get('context') or []
                for i, doc in enumerate(context_docs):
                    st.write(doc.page_content)
                    st.write('------------------------')
        except Exception as e:
            st.error(f"An error occurred while running retrieval chain: {e}")
            st.exception(e)

# Footer note
st.caption("If you get import errors related to langchain, install the pinned versions listed at the top of this file or ask me to provide a requirements.txt with tested versions.")
