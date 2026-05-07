import os
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- CONFIGURATION ---
RAW_DIR = "backend/data/raw"
PROCESSED_DIR = "backend/data/processed"
JSON_PATH = os.path.join(PROCESSED_DIR, "legal_data_final.json")
FAISS_PATH = PROCESSED_DIR 
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def update_memory():
    print("🔍 Step 1: Analyzing existing memory...")
    existing_json_data = []
    processed_titles_clean = set()
    
    # Load JSON and clean up titles for comparison
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH, 'r', encoding='utf-8') as f:
            existing_json_data = json.load(f)
            for item in existing_json_data:
                title = item.get("title", "")
                # Normalize: remove .pdf, lowercase, and strip whitespace
                clean_title = str(title).replace(".pdf", "").strip().lower()
                processed_titles_clean.add(clean_title)
    
    print(f"📊 JSON currently contains {len(processed_titles_clean)} processed legal documents.")

    print("\n🔍 Step 2: Filtering for ONLY new files...")
    all_pdfs = [f for f in os.listdir(RAW_DIR) if f.lower().endswith('.pdf')]
    
    new_pdfs = []
    for pdf in all_pdfs:
        clean_filename = pdf.replace(".pdf", "").strip().lower()
        if clean_filename not in processed_titles_clean:
            new_pdfs.append(pdf)
        else:
            # This confirms the script is correctly identifying old files
            print(f"✅ Skipping: {pdf} (Already in memory)")

    if not new_pdfs:
        print("\n✨ All files are already processed! Memory is 100% up to date.")
        return

    print(f"\n🚀 SUCCESS: Found exactly {len(new_pdfs)} new files to inject.")

    # Step 3: Load existing FAISS without rebuilding
    print("\n🧠 Step 3: Loading existing AI brain (FAISS)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    try:
        vectorstore = FAISS.load_local(
            FAISS_PATH, 
            embeddings, 
            index_name="faiss_index", 
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        print(f"❌ Error: Could not load existing FAISS index. Error: {e}")
        return

    # Step 4: Process New Files Only
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    new_json_entries = []

    for pdf in new_pdfs:
        print(f"⏳ Embedding new law: {pdf}...")
        file_path = os.path.join(RAW_DIR, pdf)
        doc_title = pdf.replace(".pdf", "") # Keep original casing for title field
        
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            chunks = text_splitter.split_documents(pages)
            
            for chunk in chunks:
                new_json_entries.append({
                    "title": doc_title,
                    "text": chunk.page_content,
                    "source": "Official PDF",
                    "type": "Act"
                })
            
            # Append only these new chunks to existing FAISS
            vectorstore.add_documents(chunks)
        except Exception as e:
            print(f"⚠️ Error processing {pdf}: {e}")

    # Step 5: Save and Sync
    print("\n💾 Step 5: Saving updated database...")
    
    # Save the expanded FAISS index
    vectorstore.save_local(FAISS_PATH, index_name="faiss_index")
    
    # Append new data to JSON and overwrite
    existing_json_data.extend(new_json_entries)
    with open(JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(existing_json_data, f, ensure_ascii=False, indent=4)

    print(f"\n🎉 DONE! Added {len(new_pdfs)} new laws. Total database size increased.")

if __name__ == "__main__":
    update_memory()