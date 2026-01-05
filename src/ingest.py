import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# PATH CONSTANTS
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
ROOT_DIR = os.path.dirname(BASE_DIR)                  
DATA_PATH = os.path.join(ROOT_DIR, "data")
DB_FAISS_PATH = os.path.join(ROOT_DIR, "vectorstore", "db_faiss")

# INGESTION FUNCTION
def create_vector_db():
    print("[INFO] 1. Loading PDF...")
    loader = PyPDFLoader(os.path.join(DATA_PATH, "ipc.pdf"))
    documents = loader.load()
    print(f"[SUCCESS] Loaded {len(documents)} pages.")

    print("[INFO] 2. Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    texts = text_splitter.split_documents(documents)
    print(f"[SUCCESS] Created {len(texts)} text chunks.")

    print("[INFO] 3. Creating Embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    print("[INFO] 4. Building Vector Database...")
    db = FAISS.from_documents(texts, embeddings)
    
    db.save_local(DB_FAISS_PATH)
    print(f"[SUCCESS] Vector Database saved to '{DB_FAISS_PATH}'")

if __name__ == "__main__":
    create_vector_db()