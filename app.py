import streamlit as st
from pdfminer.high_level import extract_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import re
import contractions
import nltk
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from deep_translator import GoogleTranslator
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Ensure NLTK resources are available
def download_nltk_resources():
    resources = {
        'stopwords': 'corpora',
        'punkt': 'tokenizers',
        'averaged_perceptron_tagger': 'taggers',
        'wordnet': 'corpora'
    }
    for resource, resource_type in resources.items():
        try:
            nltk.data.find(f'{resource_type}/{resource}')
        except LookupError:
            nltk.download(resource)

download_nltk_resources()

STOPWORDS = set(stopwords.words('english'))

# Set up the Streamlit page configuration
st.set_page_config(page_icon='📑', page_title="Resume QnA Bot")

# Initialize or update session state variables
def initialize_session_state():
    if 'uploaded' not in st.session_state:
        st.session_state.uploaded = False
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "qa_model" not in st.session_state:
        st.session_state.qa_model = pipeline("question-answering", model="deepset/bert-large-uncased-whole-word-masking-squad2")
    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = HuggingFaceEmbeddings(model_name="Alibaba-NLP/gte-base-en-v1.5", model_kwargs={"trust_remote_code": True})

# Toggle the state of the uploader button and clear messages
def toggle_uploader():
    st.session_state.uploaded = not st.session_state.uploaded
    reset_embedding()

# Extract and clean text from PDF
def extract_and_clean_text_from_pdf(uploaded_file):
    raw_text = extract_text(uploaded_file)
    cleaned_text = re.sub(r'\n\s*\n', '\n', re.sub(r'\s+', ' ', raw_text.strip()))
    return cleaned_text

# Remove non-ASCII characters from text
def remove_non_ascii(text):
    return text.encode(encoding="ascii", errors="ignore").decode('ascii')

# Chunk text using the specified text splitter
def chunk_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text=text)
    return [remove_non_ascii(chunk) for chunk in chunks]

# Generate a response from the QA model based on the context and prompt
def generate_response(qa_model, prompt, context):
    try:
        return qa_model({'question': prompt, 'context': context})['answer']
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

    text = re.sub(r'\s+', ' ', text.strip())  # Normalize spaces
    text = GoogleTranslator(source='auto', target='en').translate(text)  # Translate to English
    text = contractions.fix(text).lower()  # Expand contractions and convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove non-alphanumeric characters

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
            if uploaded_file:
                if "db" not in st.session_state:
                    with st.spinner('Processing resume...'):
                        # Extract, clean, and chunk text from PDF
                        resume_text = extract_and_clean_text_from_pdf(uploaded_file)
                        chunks = chunk_text(resume_text)

                    with st.spinner('Storing embeddings...'):
                        # Store embeddings
                        db = FAISS.from_texts(chunks, st.session_state.embedding_model)
                        st.session_state.db = db  # Store the embeddings in session state
                
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
