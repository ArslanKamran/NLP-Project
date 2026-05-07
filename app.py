import os
import sys
import json
import time
import threading
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Ensure local imports work correctly
sys.path.append(os.getcwd()) 
load_dotenv()  

# 1. Store multiple keys in a list
# Add GEMINI_KEY_2 and GEMINI_KEY_3 to your .env or Render Environment Variables
GEMINI_KEYS = [
    os.getenv("GEMINI_KEY_1"),
    os.getenv("GEMINI_KEY_2"),
    os.getenv("GEMINI_KEY_3")
]

# Filter out None values to prevent initialization errors
GEMINI_KEYS = [k for k in GEMINI_KEYS if k]

if not GEMINI_KEYS:
    print("❌ ERROR: No Gemini API keys found in environment.")

def create_llm(api_key):
    """Helper to initialize the LLM with a specific key."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.3, # Slightly adjusted for better legal nuance
        api_key=api_key,
        max_tokens=4096,
        max_retries=0, # We handle retries manually via key rotation
        safety_settings={
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE"
        }
    )

# Initial setup using the first available key
try:
    llm = create_llm(GEMINI_KEYS[0])
    print(f"⚡ SUCCESS: Gemini AI Model Ready (Using Key 1/{len(GEMINI_KEYS)})")
except Exception as e:
    print(f"❌ ERROR: Initial Gemini Setup Failed - {e}")

rag = None
try:
    from backend.ai.rag_engine import RAGEngine
    rag = RAGEngine()
    
    print("🔥 Forcing Local FAISS Brain to wake up...")
    is_awake = False
    while not is_awake:
        try:
            rag.embeddings.embed_query("wake up")
            is_awake = True
            print("✅ SUCCESS: Local FAISS Memory is fully awake!")
        except Exception:
            print("⏳ Model is still booting. Knocking again in 5 seconds...")
            time.sleep(5)
            
except Exception as e:
    print(f"❌ ERROR: Local AI Memory Failed - {e}")

def keep_brain_awake():
    while True:
        time.sleep(300) 
        if rag:
            try:
                rag.embeddings.embed_query("heartbeat ping")
                print("💓 [Heartbeat] Sent signal to keep Embedding Brain awake.")
            except Exception:
                pass 

threading.Thread(target=keep_brain_awake, daemon=True).start()

app = Flask(__name__)

def generate_gemini_response(prompt):
    # This loop tries each key you've provided in GEMINI_KEYS
    for i, key in enumerate(GEMINI_KEYS):
        try:
            # Initialize the LLM specifically for this key
            current_llm = create_llm(key)
            
            for chunk in current_llm.stream(prompt):
                # 📊 THE TOKEN MONITOR
                if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                    usage = chunk.usage_metadata
                    in_tokens = usage.get('input_tokens', 0)
                    out_tokens = usage.get('output_tokens', 0)
                    total_tokens = usage.get('total_tokens', 0)
                    
                    print("\n" + "="*50)
                    print(f"📊 [LIVE TOKEN MONITOR - KEY {i+1}]")
                    print(f"📥 Input (Reading PDFs) : {in_tokens} tokens")
                    print(f"📤 Output (Writing Urdu): {out_tokens} tokens")
                    print(f"📈 Total for this query : {total_tokens} tokens")
                    print("="*50 + "\n")

                if chunk.content:
                    yield chunk.content
            
            # If the stream finishes successfully, exit the function
            return  

        except Exception as e:
            error_msg = str(e).lower()
            # If rate limited (429), log and move to the next key
            if '429' in error_msg or 'rate_limit' in error_msg or 'resource_exhausted' in error_msg:
                print(f"⚠️ Key {i+1} limit reached. Switching to next key...")
                continue 
            else:
                # Keep your original error yield response
                yield f"\n\n### ⚠️ System Interruption\nAn unexpected error occurred: {str(e)}"
                return

    # Keep your original limit reached yield response if ALL keys fail
    yield "\n\n### ⏳ Limit Reached\nPlease wait 60 seconds, take a deep breath, and ask again. If still fails then daily limit is reached. Try again tomorrow."

@app.route('/')
def home(): return render_template('index.html')

@app.route('/consult', methods=['POST'])
def consult():
    data = request.json
    user_text = data.get('text', '').strip()
    user_lang = data.get('lang', 'en') 

    context = ""
    if rag:
        try:
            docs = rag.search(user_text, k=20) 
            if docs:
                for doc in docs:
                    if hasattr(doc, 'page_content'):
                        text_snippet = doc.page_content[:2500]
                    elif isinstance(doc, dict):
                        text_snippet = doc.get('text', str(doc))[:2500]
                    else:
                        text_snippet = str(doc)[:2500]
                    
                    context += f"\n--- DOCUMENT TEXT ---\n{text_snippet}...\n"
        except Exception as e:
             return Response(f"Memory Error: {str(e)}", mimetype='text/plain')

    # 🛡️ THE HYBRID PROMPT: Never says [DATA MISSING] again!
    if user_lang == 'ur':
        lang_instruction = (
            "CRITICAL INSTRUCTION: User prefers URDU. Write ENTIRE response in formal 'Adalti' (Legal) Urdu.\n\n"
            "### 🧠 STEP 1: INTENT EVALUATION\n"
            "- If query is a greeting: Respond ONLY with 'السلام علیکم! میں قانون اے آئی ہوں، آپ کا قانونی معاون۔ میں آپ کی کیا مدد کر سکتا ہوں؟' and STOP.\n"
            "- If query is abusive/off-topic: Respond ONLY with '🛑 **[OFF-TOPIC]** میں صرف پاکستانی قانون سے متعلق سوالات کے جوابات دے سکتا ہوں۔' and STOP.\n"
            "- If valid legal question: Proceed to Step 2.\n\n"
            "### 🏛️ STEP 2: LEGAL FORMATTING\n"
            "- RULE 1: NEVER mention 'provided data', 'context', 'مہیا کردہ معلومات', or 'متن'. Act as a human lawyer speaking directly.\n"
            "- RULE 2: If the DATA is empty or irrelevant, seamlessly use your internal knowledge of Pakistani Law to answer the question. NEVER say the data is missing.\n"
            "- RULE 3: Use EXACTLY these headers:\n"
            "### ⚖️ قانونی تجزیہ\n"
            "(Detailed Urdu analysis in bullet points. Keep Section numbers in English digits, e.g., Section 302)\n"
            "### 📜 قانونی حوالہ\n"
            "(List specific Sections/Acts here. If using internal knowledge, list the correct Pakistani laws, e.g., Income Tax Ordinance, 2001.)\n"
            "- RULE 4: Stop immediately after the citations."
        )
    else:
        lang_instruction = (
            "CRITICAL INSTRUCTION: User prefers ENGLISH. Write ENTIRE response in professional English.\n\n"
            "### 🧠 STEP 1: INTENT EVALUATION\n"
            "- If query is a greeting: Respond ONLY with 'Greetings! I am Qanoon AI, a specialized legal assistant for Pakistani law. How can I assist you today?' and STOP.\n"
            "- If query is abusive/off-topic: Respond ONLY with '🛑 **[OFF-TOPIC]** I can only assist with matters related to Pakistani law.' and STOP.\n"
            "- If valid legal question: Proceed to Step 2.\n\n"
            "### 🏛️ STEP 2: LEGAL FORMATTING\n"
            "- RULE 1: NEVER mention 'provided data', 'context', or 'documents'. Act as a human lawyer speaking directly.\n"
            "- RULE 2: If the DATA is empty or irrelevant, seamlessly use your internal knowledge of Pakistani Law to answer the question. NEVER say the data is missing.\n"
            "- RULE 3: Use EXACTLY these headers:\n"
            "### ⚖️ Legal Analysis\n"
            "(Detailed English analysis in bullet points.)\n"
            "### 📜 Legal Authority\n"
            "(List specific Sections/Acts here. If using internal knowledge, cite the correct Pakistani laws.)\n"
            "- RULE 4: Stop immediately after the citations."
        )

    system_prompt = (
        f"You are Qanoon AI, an elite Legal Consultant for Pakistani Law.\n{lang_instruction}"
    )

    full_prompt = f"{system_prompt}\n\n### DATA:\n{context}\n\n### QUERY: {user_text}"
    return Response(stream_with_context(generate_gemini_response(full_prompt)), mimetype='text/plain')

LAWYERS_DB_PATH = os.path.join("backend", "data", "raw", "lawyers_db.json")

@app.route('/lawyers', methods=['GET'])
def get_lawyers():
    all_lawyers = []
    filtered_lawyers = []
    category = request.args.get('category', 'general').lower().strip()
    
    try:
        if os.path.exists(LAWYERS_DB_PATH):
            with open(LAWYERS_DB_PATH, 'r', encoding='utf-8') as f:
                all_lawyers = json.load(f)
        else:
            return jsonify([]) 
    except Exception:
        return jsonify([])

    if not all_lawyers:
        return jsonify([])

    if category == 'general' or not category:
        return jsonify(all_lawyers[:10])
    
    for lawyer in all_lawyers:
        lawyer_tags = [t.lower() for t in lawyer.get('tags', [])]
        lawyer_specialty = lawyer.get('specialty', '').lower()
        if category in lawyer_tags or category in lawyer_specialty:
            filtered_lawyers.append(lawyer)
    
    if not filtered_lawyers:
        return jsonify(all_lawyers[:5])
        
    return jsonify(filtered_lawyers)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))