import os
import sys
import time
import json
import concurrent.futures
from tqdm import tqdm  
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS 
from langchain_core.documents import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
JSON_FILE_PATH = os.path.join(os.getcwd(), "backend", "data", "processed", "legal_data_final.json")
FAISS_INDEX_PATH = os.path.join(os.getcwd(), "backend", "data", "faiss_index") 

class RAGEngine:
    def __init__(self):
        print("🔌 Initializing Local Brain (Hugging Face + FAISS)...")
        
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            print("⚠️ WARNING: HF_TOKEN not found!")

        # API-BASED EMBEDDINGS (Memory Safe for Render)
        self.embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            huggingfacehub_api_token=hf_token
        )
        
        self.db = None
        self.load_index()

    def load_index(self):
        """Connects to the existing local FAISS index."""
        try:
            if os.path.exists(FAISS_INDEX_PATH):
                # allow_dangerous_deserialization is required to load FAISS indices safely on your own server
                self.db = FAISS.load_local(
                    folder_path=FAISS_INDEX_PATH, 
                    embeddings=self.embeddings, 
                    allow_dangerous_deserialization=True 
                )
                print("✅ Local FAISS Memory Connected Successfully!")
            else:
                print("⚠️ FAISS index not found. You need to build it first.")
        except Exception as e:
            print(f"⚠️ Error connecting to local FAISS memory: {e}")

    def process_single_entry(self, entry):
        """Fast extraction for JSON records."""
        try:
            text_content = entry.get('text') or entry.get('content') or str(entry)
            title = entry.get('title') or entry.get('section') or "Legal Document"
            source = entry.get('source') or "Official PDF"
            doc_type = entry.get('type') or "Act"

            if text_content:
                return Document(
                    page_content=text_content, 
                    metadata={
                        "title": title,
                        "source": source,
                        "type": doc_type
                    }
                )
        except Exception:
            return None
        return None

    def build_index_from_json(self):
        """Extreme speed local build and save to FAISS disk with Auto-Save."""
        print(f"🏗️  Loading data from: {JSON_FILE_PATH}")
        if not os.path.exists(JSON_FILE_PATH):
            print(f"❌ Error: JSON file not found at {JSON_FILE_PATH}")
            return

        with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"⚡ Extracting {len(data)} records via Threads...")
        documents = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(self.process_single_entry, data))
            documents = [doc for doc in results if doc]

        print("✂️  Splitting into chunks (1000 size)...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_docs = text_splitter.split_documents(documents)
        
        total_chunks = len(final_docs)
        print(f"🚀 Building Local FAISS Vector Store with {total_chunks} chunks...")

        # Build in batches to save RAM
        batch_size = 50 
        
        for i in tqdm(range(0, total_chunks, batch_size), desc="Local FAISS Progress"):
            batch = final_docs[i : i + batch_size]
            
            # Ensure metadata stays under limits just in case
            for doc in batch:
                if len(doc.page_content.encode('utf-8')) > 38000:
                    doc.page_content = doc.page_content[:38000]

            # --- 🛡️ THE FIX: ROBUST RETRY LOOP ---
            success = False
            retries = 0
            while not success and retries < 5:
                try:
                    if self.db is None:
                        self.db = FAISS.from_documents(batch, self.embeddings)
                    else:
                        self.db.add_documents(batch)
                    success = True
                except Exception as e:
                    retries += 1
                    print(f"\n⚠️ Hugging Face API Overloaded. Sleeping 15s... (Attempt {retries}/5)")
                    time.sleep(15)
            
            # If it fails 5 times in a row, save everything done so far and exit cleanly.
            if not success:
                print("\n❌ Fatal Error: API completely unresponsive. Saving progress before quitting...")
                if self.db:
                    self.db.save_local(FAISS_INDEX_PATH)
                sys.exit(1)

            # --- 💾 THE FIX: AUTO-SAVE FEATURE ---
            # Save progress every 2,500 chunks (50 batches) so you never start from zero!
            if i > 0 and i % 2500 == 0 and self.db:
                self.db.save_local(FAISS_INDEX_PATH)

        # Final Save
        if self.db:
            self.db.save_local(FAISS_INDEX_PATH)
            print(f"\n✅ SUCCESS! Local FAISS Index completely built and saved to {FAISS_INDEX_PATH}.")

    def search(self, query, k=20):
        if not self.db:
            return []
        try:
            # 🚀 UPGRADED: Uses MMR to force FAISS to find diverse legal chunks!
            results = self.db.max_marginal_relevance_search(
                query, 
                k=k, 
                fetch_k=k * 3, # Fetches 60 chunks, then picks the 20 best/most diverse
                lambda_mult=0.5 # 0.5 is the perfect balance between relevance and diversity
            )
            return [
                {
                    "text": doc.page_content, 
                    "title": doc.metadata.get('title', 'Unknown')
                } for doc in results
            ]
        except Exception as e:
            print(f"⚠️ Search error: {e}")
            return []
        
# --- WINDOWS SAFETY GUARD ---
if __name__ == "__main__":
    # Create engine and build ONLY if run directly
    rag = RAGEngine()
    # Uncomment the line below, run this file once to generate the faiss_index folder, then comment it out again!
    rag.build_index_from_json()