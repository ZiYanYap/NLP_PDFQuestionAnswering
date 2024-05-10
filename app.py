import streamlit as st
from pdfminer.high_level import extract_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import re
# from constants import *


EMBEDDING_MODEL_NAMES = {
    "MPNet": "sentence-transformers/all-mpnet-base-v2",
    "GTE": "Alibaba-NLP/gte-base-en-v1.5"
}

QA_MODEL_NAMES = {
    "deepset/roberta-base-squad2": "RoBERTa Base SQuAD 2",
    "timpal0l/mdeberta-v3-base-squad2": "mDeBERTa V3 Base SQuAD 2",
    "distilbert/distilbert-base-cased-distilled-squad": "DistilBERT Base Cased Distilled SQuAD",
    "deepset/bert-large-uncased-whole-word-masking-squad2": "BERT Large Uncased Whole Word Masking SQuAD 2",
    "mrm8488/longformer-base-4096-finetuned-squadv2": "Longformer Base 4096 Finetuned SQuAD V2"
}


def initialize_session_state():
    """Initialize or update the session state variables."""
    if 'uploaded' not in st.session_state:
        st.session_state.uploaded = False
    if "messages" not in st.session_state:
        st.session_state.messages = []


@st.cache_resource
def load_embedding_models():
    """Load and cache the embedding models."""
    return {model_key: SentenceTransformer(model_name, trust_remote_code=True)
            for model_key, model_name in EMBEDDING_MODEL_NAMES.items()}


@st.cache_resource
def load_qa_models():
    """Load and cache QA models."""
    return {model_key: pipeline("question-answering", model=model_key)
            for model_key in QA_MODEL_NAMES.keys()}


def toggle_uploader():
    """Toggle the state of the uploader button and clear messages."""
    st.session_state.uploaded = not st.session_state.uploaded
    st.session_state.messages.clear()


def extract_text_from_pdf(uploaded_file):
    """Extract text from the uploaded PDF."""
    raw_text = extract_text(uploaded_file)
    cleaned_text = re.sub(r'\n\s*\n', '\n', raw_text.strip())
    cleaned_text = re.sub(r' +', ' ', cleaned_text)
    return cleaned_text


# def removeNonASCII(text):
#     """Encode the given text to ASCII and decode back to string, ignoring non-ASCII characters."""
#     # Encode to ASCII, ignoring errors
#     encoded_text = text.encode(encoding="ascii", errors="ignore")
#     # Decode back to string
#     decoded_text = encoded_text.decode('ascii')
#     return decoded_text


def chunk_text(text):
    """Chunk text."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text=text)
    # chunks = [removeNonASCII(chunk) for chunk in chunks]
    return chunks


def generate_response(selected_model, prompt, context):
    qa_pipeline = selected_model
    response = qa_pipeline({'question': prompt, 'context': context})['answer']
    return response


def main():
    # Initialize models when the app runs
    embedding_models = load_embedding_models()
    qa_models = load_qa_models()
    initialize_session_state()

    st.title("Resume Chat Application📃")

    with st.sidebar:
        uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"], disabled=st.session_state.uploaded, on_change=toggle_uploader)

        selected_embedding_model_name = st.radio("Select an Embedding Model:", list(EMBEDDING_MODEL_NAMES.keys()))
        # embedding_model = embedding_models[EMBEDDING_MODEL_NAMES[selected_embedding_model_name]]
        selected_model_name = st.radio("Select a QA Pipeline Model:", list(QA_MODEL_NAMES.values()))
        selected_model = qa_models[[key for key, value in QA_MODEL_NAMES.items() if value == selected_model_name][0]]

    if uploaded_file:
        resume_text = extract_text_from_pdf(uploaded_file)
        chunks = chunk_text(resume_text)
        # st.write(chunks)

        # Choose an embedding model based on your application needs
        db = FAISS.from_texts(chunks, HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAMES[selected_embedding_model_name], model_kwargs={"trust_remote_code": True}))

        # Display chat messages from history if there's any
        if st.session_state.messages:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input("Ask a question about the resume..."):
            # prompt = preprocess_query(prompt)
            most_similar_chunks = db.similarity_search_with_score(prompt, k=3)
            context = "\n\n".join([doc.page_content for doc, _ in most_similar_chunks])
            response = generate_response(selected_model, prompt, context)

            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": response})

            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                st.markdown(response)


if __name__ == "__main__":
    main()
