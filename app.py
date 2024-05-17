# from constants2 import *
import streamlit as st
from pdfminer.high_level import extract_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import re
# import os
import contractions
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from deep_translator import GoogleTranslator
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

STOPWORDS = set(stopwords.words('english'))

# Set up the Streamlit page configuration
st.set_page_config(page_icon='📑', page_title="Resume QnA Bot")

# Ensure the directory for model storage exists
# def ensure_model_directory(base_dir="saved_models/qa_models"):
#     if not os.path.exists(base_dir):
#         os.makedirs(base_dir)

# Retrieve the full path for a model
# def get_model_path(model_name, base_dir="saved_models/qa_models"):
#     return os.path.join(base_dir, model_name.replace("/", "_"))

# Save both the model and its tokenizer to the disk
# def save_model(model, tokenizer, model_path):
#     model.save_pretrained(model_path)
#     tokenizer.save_pretrained(model_path)

# Load or download the model depending on its availability
def load_or_download_model(model_name, base_dir="saved_models/qa_models"):
    # model_path = get_model_path(model_name, base_dir)
    # if not os.path.exists(model_path):
    #     print(f"Downloading and saving model: {model_name}")
    #     model = pipeline("question-answering", model=model_name)
    #     save_model(model.model, model.tokenizer, model_path)
    # else:
    #     print(f"Loading model from disk: {model_name}")
    #     model = pipeline("question-answering", model=model_path, tokenizer=model_path)

    model = pipeline("question-answering", model=model_name)

    return model

# Initialize once at start
# ensure_model_directory()

# Initialize or update session state variables
def initialize_session_state():
    if 'uploaded' not in st.session_state:
        st.session_state.uploaded = False
    if "messages" not in st.session_state:
        st.session_state.messages = []

# Toggle the state of the uploader button and clear messages
def toggle_uploader():
    st.session_state.uploaded = not st.session_state.uploaded
    reset_embedding()

# Extract and clean text from PDF
def extract_and_clean_text_from_pdf(uploaded_file):
    raw_text = extract_text(uploaded_file)
    return re.sub(r'\n\s*\n', '\n', re.sub(r'\s+', ' ', raw_text.strip()))

# Remove non-ASCII characters from text
def removeNonASCII(text):
    encoded_text = text.encode(encoding="ascii", errors="ignore")
    return encoded_text.decode('ascii')

# Chunk text using the specified text splitter
def chunk_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text=text)
    return [removeNonASCII(chunk) for chunk in chunks]

# Generate a response from the QA model based on the context and prompt
def generate_response(selected_model, prompt, context):
    try:
        qa_pipeline = selected_model
        return qa_pipeline({'question': prompt, 'context': context})['answer']
    except Exception as e:
        st.error(f"Failed to generate response: {e}")
        return "An error occurred while generating the response."

# Clear the chat messages
def clear_chat():
    st.session_state.messages.clear()
    
def reset_embedding():
    clear_chat()
    if "db" in st.session_state:
        st.session_state.pop("db")

# Processes the user query
def preprocess_query(text):
    lemmatizer = WordNetLemmatizer()

    # Normalize spaces
    text = re.sub(r'\s+', ' ', text.strip())

    # Translate text to English
    text = GoogleTranslator(source='auto', target='en').translate(text=text)
    
    # Expand contractions (e.g., "isn't" to "is not")
    text = contractions.fix(text)
    
    # Convert text to lowercase to standardize it
    text = text.lower()
    
    # Remove non-alphanumeric characters
    text = re.sub(r'[^\w\s]', '', text)

    # Part-of-Speech tagging
    pos_tagged = pos_tag(word_tokenize(text))

    # Map POS tags to wordnet tags and lemmatize verbs
    wordnet_tagged = [(word, wordnet.VERB) if tag.startswith('V') else (word, None) for word, tag in pos_tagged]
    tokens = []
    for word, tag in wordnet_tagged:
        if tag is None:
            tokens.append(word)
        else:
            tokens.append(lemmatizer.lemmatize(word, tag))
    
    # Filter out stopwords
    cleaned_tokens = [token for token in tokens if token.lower() not in STOPWORDS]

    # Reconstruct the processed text
    if len(cleaned_tokens) == 0:
        text = ' '.join(tokens)
    else:
        text = ' '.join(cleaned_tokens)

    return text

# Main function to run the Streamlit app
def main():
    initialize_session_state()
    st.title("Resume Question-Answering App📃")

    with st.sidebar:
        uploaded_file = st.file_uploader("Upload a resume (PDF)", type=["pdf"], disabled=st.session_state.uploaded, on_change=toggle_uploader)

        # Define a button to start the chat which loads models, processes text, etc.
        if st.button("Start Chat"):
            if uploaded_file is not None:
                if "db" not in st.session_state:
                    with st.spinner('Processing resume...'):
                        # Extract text from PDF
                        resume_text = extract_and_clean_text_from_pdf(uploaded_file)
                        
                        # Chunk text
                        chunks = chunk_text(resume_text)

                    with st.spinner('Loading embedding model...'):
                        # Load embedding model
                        embedding_model = HuggingFaceEmbeddings(model_name="Alibaba-NLP/gte-base-en-v1.5", model_kwargs={"trust_remote_code": True})

                    with st.spinner('Storing embeddings...'):
                        # Store embeddings
                        db = FAISS.from_texts(chunks, embedding_model)
                        st.session_state.db = db  # Store the embeddings in session state
                
                if "qa_model" not in st.session_state:
                    with st.spinner('Loading QA model...'):
                        # Load QA model
                        QA_model = pipeline("question-answering", model="deepset/bert-large-uncased-whole-word-masking-squad2")
                        st.session_state.qa_model = QA_model  # Store the QA model in session state
                
                st.success('Ready to answer questions from the resume!')
            
            else:
                st.error('Please upload a resume PDF to start the chat.')

    # Display chat history
    if st.session_state.messages:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if 'db' in st.session_state and 'qa_model' in st.session_state:
        # Accept user query input
        if prompt := st.chat_input("Ask a question about the resume..."):
            # Preprocessing on user's query
            with st.spinner("Processing your question..."):
                cleaned_prompt = preprocess_query(prompt)
            
            with st.spinner("Fetching relevant content and generating response..."):
                # Retrieve top 3 most similar chunks using FAISS
                most_similar_chunks = st.session_state.db.similarity_search_with_score(cleaned_prompt, k=3)
                # Join the 3 chunks together with 2 newline characters
                context = "\n\n".join([doc.page_content for doc, _ in most_similar_chunks])
                # Pass question and context into QA pipeline to generate response
                response = generate_response(st.session_state.qa_model, cleaned_prompt, context)

            if response.strip() == "":
                response = "I'm sorry, I don't understand your question."
            
            # Add chat to session state as history
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": response})

            # Display prompt and response
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                st.markdown(response)


if __name__ == "__main__":
    main()
