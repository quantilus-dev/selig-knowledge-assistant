import os
import glob
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# Load key from .env file
def load_env_key():
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                if line.startswith("GEMINI_API_KEY="):
                    return line.split("=")[1].strip()
    return os.environ.get("GEMINI_API_KEY")

API_KEY = load_env_key()

if API_KEY:
    genai.configure(api_key=API_KEY)
    # Using the 2.0-flash model which we confirmed exists for this key
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
    except:
        model = genai.GenerativeModel('gemini-1.5-flash')
else:
    model = None

def check_models():
    if not API_KEY: return
    try:
        print("Checking available models with your key...")
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f" - Found: {m.name}")
    except Exception as e:
        print(f"Warning during model check: {e}")

def load_wiki(wiki_dir="wiki"):
    print(f"Loading LLM Wiki from {wiki_dir}...")
    wiki_files = glob.glob(os.path.join(wiki_dir, "*.md"))
    wiki_context = ""
    for wiki_file in wiki_files:
        try:
            with open(wiki_file, 'r', encoding='utf-8') as f:
                filename = os.path.basename(wiki_file)
                content = f.read()
                wiki_context += f"\n\n--- WIKI PAGE: {filename} ---\n{content}\n"
        except Exception as e:
            print(f"Error reading {wiki_file}: {e}")
    return wiki_context

def load_pdf_chunks(directory="."):
    print(f"Loading raw PDF chunks for technical reference...")
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
        except Exception as e:
            print(f"Error reading {pdf_file}: {e}")
            
    return chunks, metadata

class SeligWikiChatbot:
    def __init__(self, wiki_context, pdf_chunks, pdf_metadata):
        self.wiki_context = wiki_context
        self.pdf_chunks = pdf_chunks
        self.pdf_metadata = pdf_metadata
        self.vectorizer = TfidfVectorizer(stop_words='english')
        if self.pdf_chunks:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.pdf_chunks)
        else:
            self.tfidf_matrix = None

    def retrieve_technical(self, query, top_k=2):
        if self.tfidf_matrix is None:
            return ""
        
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        tech_context = ""
        for idx in top_indices:
            if similarities[idx] > 0.1: # higher threshold for tech specs
                meta = self.pdf_metadata[idx]
                tech_context += f"\n--- RAW DATA: {meta['file']} (Page {meta['page']}) ---\n{self.pdf_chunks[idx]}\n"
        return tech_context

    def ask(self, query):
        tech_context = self.retrieve_technical(query)
        
        if model:
            prompt = f"""You are the 'Selig Group Knowledge Assistant', an expert on container sealing and venting solutions.
You have access to a distilled 'LLM Wiki' which contains organized knowledge about the company, its brands, and technologies.
You also have access to 'RAW DATA' from technical datasheets for specific queries.

LLM WIKI KNOWLEDGE:
{self.wiki_context}

{f"SUPPLEMENTAL TECHNICAL DATA (if relevant):{tech_context}" if tech_context else ""}

User Question: {query}

Instructions:
1. Use the LLM Wiki as your primary source for high-level information, brand stories, and general capabilities.
2. Use the Technical Data only if the user asks for specific measurements, thicknesses, or material compositions not found in the wiki.
3. Be concise and professional.
4. If you don't know the answer, say you don't have enough information in the wiki or datasheets.
"""
            try:
                response = model.generate_content(prompt)
                return response.text
            except Exception as e:
                return f"Error: {e}\n\n(Note: Using raw wiki context due to API error)\n{self.wiki_context[:1000]}..."
        else:
            return f"Gemini API key not found. In 'LLM Wiki' mode, I recommend using an AI. Here is a search result from the technical PDFs:\n{tech_context}"

def main():
    print("Welcome to the Selig Group 'LLM Wiki' Chatbot!")
    check_models()
    
    wiki_context = load_wiki()
    pdf_chunks, pdf_metadata = load_pdf_chunks()
    
    bot = SeligWikiChatbot(wiki_context, pdf_chunks, pdf_metadata)
    
    print("\nChatbot is ready (Powered by Karpathy's LLM Wiki concept)!")
    print("Type 'exit' or 'quit' to stop.")
    
    while True:
        try:
            query = input("\nYou: ")
            if query.lower() in ['exit', 'quit']:
                break
            if not query.strip():
                continue
                
            print("Thinking...")
            answer = bot.ask(query)
            print(f"\nBot: {answer}")
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()