import os
import json
import pandas as pd
from PyPDF2 import PdfReader

# --- PATHS ---
# We assume you run this from the project root (E:\PROJECTS\Qanoon AI)
RAW_DIR = "backend/data/raw"
PROCESSED_FILE = "backend/data/processed/legal_data_final.json"

def clean_text(text):
    """Simple cleaner to remove messy whitespace"""
    if not text: return ""
    return " ".join(text.split())

def process_my_files():
    print("🚀 Starting Data Processor...")
    all_documents = []

    # 1. PROCESS THE PDFS
    # This will grab Constitution, CPC, CrPC, PECA, and Glossary
    for filename in os.listdir(RAW_DIR):
        if filename.endswith(".pdf"):
            filepath = os.path.join(RAW_DIR, filename)
            print(f"📄 Reading PDF: {filename}...")
            
            try:
                reader = PdfReader(filepath)
                full_text = ""
                for page in reader.pages:
                    full_text += page.extract_text() or ""
                
                # Only add if we actually got text
                if len(full_text) > 100:
                    all_documents.append({
                        "title": filename.replace(".pdf", ""),
                        "text": clean_text(full_text),
                        "source": "Official PDF",
                        "type": "Act"
                    })
                    print(f"   ✅ Captured {len(full_text)} characters.")
            except Exception as e:
                print(f"   ❌ Error reading {filename}: {e}")

    # 2. PROCESS THE JSON (pakistan_laws_verified.json)
    json_path = os.path.join(RAW_DIR, "pakistan_laws_verified.json")
    if os.path.exists(json_path):
        print(f"📂 Reading JSON: pakistan_laws_verified.json...")
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            count = 0
            if isinstance(data, list):
                for item in data:
                    # Tries to find 'title' or 'file_name', and 'text' or 'content'
                    title = item.get('title', item.get('file_name', 'Unknown Law'))
                    text = item.get('text', item.get('content', ''))
                    
                    if len(text) > 50:
                        all_documents.append({
                            "title": title,
                            "text": clean_text(text),
                            "source": "Legal Dataset",
                            "type": "General"
                        })
                        count += 1
            print(f"   ✅ Added {count} laws from JSON.")
        except Exception as e:
            print(f"   ❌ Error reading JSON: {e}")

    # 3. PROCESS URDU DATASET (urdu-news-dataset-1M.csv)
    # Checks for the csv file visible in your screenshot
    for filename in os.listdir(RAW_DIR):
        if "urdu-news" in filename and filename.endswith(".csv"):
             print(f"📂 Reading Urdu Context: {filename}...")
             try:
                 # Read only the first 500 rows to save time/memory
                 df = pd.read_csv(os.path.join(RAW_DIR, filename), nrows=500)
                 
                 # We look for the first column that contains text
                 text_col = df.columns[0] # Usually the first column has the text
                 
                 if not df.empty:
                     text_sample = "\n".join(df[text_col].astype(str).tolist())
                     all_documents.append({
                         "title": "Urdu Language Context",
                         "text": clean_text(text_sample),
                         "source": "Urdu News Dataset",
                         "type": "Language"
                     })
                     print("   ✅ Added Urdu language context.")
             except Exception as e:
                 print(f"   ⚠️ Could not process Urdu CSV (skipping): {e}")

    # 4. SAVE EVERYTHING
    if all_documents:
        os.makedirs(os.path.dirname(PROCESSED_FILE), exist_ok=True)
        with open(PROCESSED_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_documents, f, indent=4, ensure_ascii=False)
        print(f"\n✨ SUCCESS! Saved {len(all_documents)} documents to {PROCESSED_FILE}")
    else:
        print("\n❌ No data was processed. Check your folder paths!")

if __name__ == "__main__":
    process_my_files()