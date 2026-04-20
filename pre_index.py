import glob
import pickle
import os
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer

def create_index(directory="."):
    print("Pre-indexing PDFs... this may take a minute.")
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
            print(f"Skipping {pdf_file}: {e}")

    if chunks:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(chunks)
        
        data = {
            "vectorizer": vectorizer,
            "tfidf_matrix": tfidf_matrix,
            "chunks": chunks,
            "metadata": metadata
        }
        
        with open("search_index.pkl", "wb") as f:
            pickle.dump(data, f)
        print(f"Index created successfully with {len(chunks)} chunks!")

if __name__ == "__main__":
    create_index()
