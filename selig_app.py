import streamlit as st
import os
import glob
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables (API Key)
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# Page Configuration
st.set_page_config(page_title="Selig Group Assistant", page_icon="🛡️", layout="wide")

# Custom CSS for Selig Branding
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #004a99;
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: #004a99;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Helper Functions
@st.cache_resource
def init_gemini(api_key, model_name):
    if api_key:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(model_name)
    return None

@st.cache_data
def load_wiki_content(wiki_dir="wiki"):
    wiki_files = glob.glob(os.path.join(wiki_dir, "*.md"))
    wiki_context = ""
    for wiki_file in wiki_files:
        with open(wiki_file, 'r', encoding='utf-8') as f:
            filename = os.path.basename(wiki_file)
            content = f.read()
            wiki_context += f"\n\n--- WIKI PAGE: {filename} ---\n{content}\n"
    return wiki_context

import pickle

# ... (init_gemini and load_wiki_content remain same)

@st.cache_resource
def load_pdf_index(directory="."):
    # Check if a pre-built index exists
    if os.path.exists("search_index.pkl"):
        with open("search_index.pkl", "rb") as f:
            data = pickle.load(f)
            return data["vectorizer"], data["tfidf_matrix"], data["chunks"], data["metadata"]
    
    # Fallback to building it manually (slower)
    pdf_files = glob.glob(os.path.join(directory, "*.pdf"))
    chunks = []
    metadata = []
    for pdf_file in pdf_files:
        try:
            reader = PdfReader(pdf_file)
            filename = os.path.basename(pdf_file)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    chunks.append(text.strip())
                    metadata.append({"file": filename, "page": page_num + 1})
        except: continue
    
    if chunks:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(chunks)
        return vectorizer, tfidf_matrix, chunks, metadata
    return None, None, [], []

# Sidebar
with st.sidebar:
    st.image("https://www.seliggroup.com/wp-content/themes/selig/assets/images/logo.png", width=200)
    st.title("Settings")
    
    # Model Selector
    selected_model = st.selectbox(
        "Choose AI Model:",
        ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
        index=0,
        help="2.5-flash is the newest high-performance model. 2.5-pro is best for extremely complex reasoning."
    )
    
    st.info("This assistant uses the 'LLM Wiki' pattern for high-level reasoning and raw datasheets for technical specs.")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.subheader("Loaded Brand Wiki:")
    wiki_files = [os.path.basename(f) for f in glob.glob("wiki/*.md")]
    for f in wiki_files:
        st.write(f"📄 {f}")

# Main Chat UI
st.title("🛡️ Selig Group Knowledge Assistant")
st.write("Ask anything about our sealing and venting solutions.")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Load Data
model = init_gemini(API_KEY, selected_model)
wiki_context = load_wiki_content()
vectorizer, tfidf_matrix, pdf_chunks, pdf_metadata = load_pdf_index()

# Chat Input
if prompt := st.chat_input("How can I help you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Retrieval Step
        tech_context = ""
        if vectorizer is not None:
            query_vec = vectorizer.transform([prompt])
            similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
            top_indices = similarities.argsort()[-2:][::-1]
            for idx in top_indices:
                if similarities[idx] > 0.1:
                    meta = pdf_metadata[idx]
                    tech_context += f"\n--- RAW DATA: {meta['file']} (Page {meta['page']}) ---\n{pdf_chunks[idx]}\n"

        if model:
            full_prompt = f"""You are the 'Selig Group Knowledge Assistant'.
            Use the LLM Wiki for general knowledge and the RAW DATA for technical specs.
            
            CRITICAL CORRECTION FOR TECHNICAL SPECS: 
            Many of Selig's original PDF datasheets (especially ALPHASEAL and Lasan Polyester) contain a typographical error where the units for MVTR and GTR Oxygen are swapped. 
            When quoting these values, you MUST use the correct scientific units:
            - MVTR (Moisture Vapor Transmission Rate) is a mass value and should be grams: < 0.004 gm/100 sq.in./24hrs.
            - GTR Oxygen (Gas Transmission Rate) is a volume value and should be cc's: < 0.15 cc's/100 sq.in./24hrs.
            Basically, the PDF says "MVTR: 0.004 cc" and "GTR: 0.15 gm". You should report it as "MVTR: 0.004 gm" and "GTR: 0.15 cc".
            Briefly mention you corrected the units from the source document to be scientifically accurate.
            
            WIKI:
            {wiki_context}
            
            {f"RAW DATA:{tech_context}" if tech_context else ""}
            
            Question: {prompt}
            """
            try:
                response = model.generate_content(full_prompt)
                answer = response.text
            except Exception as e:
                answer = f"⚠️ API Error: {str(e)}"
        else:
            answer = "❌ API Key not found. Please check your .env file."
        
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
