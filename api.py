# api.py

import os
import csv
import json
import re
import time
import asyncio
import sqlite3
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from typing import List, Dict, Optional
from urllib.parse import quote

from dotenv import load_dotenv
load_dotenv()  # This MUST happen before any client initialization

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import google.generativeai as genai

from langsmith import traceable, Client
from langsmith.run_helpers import get_current_run_tree

client = Client()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document

from admin import setup_admin_routes

api_key = os.getenv("LANGCHAIN_API_KEY")
print(f"Using API Key: {api_key[:20]}...{api_key[-10:] if api_key else 'NONE'}")
print(f"Key length: {len(api_key) if api_key else 0}")
print(f"LANGCHAIN_TRACING_V2: {os.getenv('LANGCHAIN_TRACING_V2')}")
print(f"LANGCHAIN_PROJECT: {os.getenv('LANGCHAIN_PROJECT')}")

# Test LangSmith connection
def test_langsmith_connection():
    try:
        test_client = Client()
        projects = list(test_client.list_projects(limit=1))
        print("✅ LangSmith connection successful")
        return True
    except Exception as e:
        print(f"❌ LangSmith connection failed: {e}")
        return False

test_langsmith_connection()

last_run_ids = {}  # session_id -> run_id mapping

class FeedbackSubmission(BaseModel):
    feedback_text: str
    rating: int
    user_type: str
    session_id: Optional[str] = None  # Add this field

# Configure Gemini
if os.environ.get("GENAI_API_KEY"): # check if api key exists
    genai.configure(api_key=os.environ["GENAI_API_KEY"]) # sets up Gemini so we can send requests using the api key.
    generation_config = genai.types.GenerationConfig(
        temperature=0.1, top_p=0.8, top_k=40, max_output_tokens=2048
        # temperature=0.1 → makes output less random, more focused.
        # top_p=0.8 → limits randomness by sampling from only the most likely tokens (nucleus sampling).
        # top_k=40 → considers only the top 40 possible next words.
        # max_output_tokens=2048 → limits length of response.
    )
    # Loads Gemini 2.5 Flash with the config above.
    model = genai.GenerativeModel("gemini-2.5-flash", generation_config=generation_config)
else: # Prevents crashes if we runs the app without setting an API key
    print("⚠️ GENAI_API_KEY environment variable not set")
    model = None

app = FastAPI() # main api server
# allows requests from any origin (*).
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

LOG_FILE = "data/query_log.csv"
FEEDBACK_FILE = "data/feedback.csv"
HANDBOOK_FILE = "data/handbook.pdf"
HANDBOOK_TEXT = "" # Will hold extracted handbook text after loading
MEMORY_DB = "data/chatbot_memory.db"
CUSTOM_INFO_FILE = "data/custom_info.json"
executor = ThreadPoolExecutor()  # Runs heavy tasks in parallel threads

# Global variables
embeddings = None # will hold the HuggingFace embeddings model (turns text into vectors).
db = None # will be the Chroma vector database (stores handbook embeddings).
retriever = None # will connect to db and fetch relevant chunks when user asks a question.

last_run_ids = {}

def sanitize_text(text: str) -> str:
    """Clean text before saving to DB or CSV - Enhanced version that preserves formatting"""
    if not text:
        return ""
    
    # Dictionary of characters we want to replace with safer alternatives
    replacements = {
        '#': 'number',     # Replace '#' with the word "number"
        '＃': 'number',    # Replace full-width '＃' with "number"
        '`': "'",          # Replace backtick with single quote
        '"': "'",          # Replace double quote with single quote
        ''': "'",          # Replace right single quote with normal apostrophe
        ''': "'",          # Replace left single quote with normal apostrophe
        '—': '-',          # Replace em-dash with hyphen
        '–': '-',          # Replace en-dash with hyphen
        '…': '...',        # Replace ellipsis with three dots
        '\r': '\n',        # Convert carriage return to newline
        '\t': ' ',         # Replace tab with a space
        '\\': '/',         # Replace backslash with forward slash
    }
    
    # Apply replacements
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove control characters (ASCII 0-31 + 127) but preserve newlines (0x0A)
    text = re.sub(r'[\x00-\x09\x0b-\x1F\x7F-\x9F]', ' ', text)
    
    # Normalize whitespace but preserve line breaks
    text = re.sub(r'[ ]+', ' ', text)  # Multiple spaces -> single space
    text = re.sub(r' *\n *', '\n', text)  # Remove spaces around newlines
    text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive newlines
    
    # Ensure safe encoding
    try:
        text = text.encode("utf-8", "ignore").decode("utf-8")
    except:
        pass # If encoding fails, just keep the text as-is
    
    return text.strip() # Trim leading/trailing spaces and return the clean text

# Memory system 
class ChatbotMemory:
    def __init__(self):
        self.init_database()  # Initialize the SQLite database (create tables if not exist)
        self.custom_info = self.load_custom_info()  # Load custom info from JSON file into memory
    
    # Initialize SQLite database for conversation memory
    def init_database(self):
        os.makedirs("data", exist_ok=True)  # Make sure the "data" folder exists (creates if missing)
        conn = sqlite3.connect(MEMORY_DB)  # Connect to the SQLite database file
        cursor = conn.cursor()  # Create a cursor object to run SQL commands
        
        # Create "conversations" table if it doesn't already exist
        # Stores each chat with session ID, time, user message, bot response, and context used
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TEXT,
                user_message TEXT,
                bot_response TEXT,
                context_used TEXT
            )
        ''')
        
        # Create "sessions" table if it doesn't already exist
        # Tracks chat sessions with their creation time, last activity, and message count
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TEXT,
                last_active TEXT,
                total_messages INTEGER DEFAULT 0
            )
        ''')
        
        conn.commit()  # Save the changes
        conn.close()  # Close the database connection

    # Load custom information
    def load_custom_info(self) -> Dict:
        try:
            # If custom info file exists, load it as JSON
            if os.path.exists(CUSTOM_INFO_FILE):
                with open(CUSTOM_INFO_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading custom info: {e}")
        return {}  # Return empty dictionary if no file or error
    
    # Save custom information to file
    def save_custom_info(self):
        try:
            # Write current custom info dictionary into JSON file
            with open(CUSTOM_INFO_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.custom_info, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving custom info: {e}")
    
    # Get or create a session with error handling
    def get_or_create_session(self, session_id: str) -> str:
        try:
            conn = sqlite3.connect(MEMORY_DB)  # Connect to DB
            cursor = conn.cursor()  # Create cursor
            
            clean_session_id = sanitize_text(session_id)  # Clean session ID text
            
            # Check if session already exists
            cursor.execute('SELECT session_id FROM sessions WHERE session_id = ?', (clean_session_id,))
            if not cursor.fetchone():  # If session does not exist, create it
                cursor.execute('''
                    INSERT INTO sessions (session_id, created_at, last_active, total_messages)
                    VALUES (?, ?, ?, 0)
                ''', (clean_session_id, datetime.now().isoformat(), datetime.now().isoformat()))
            
            conn.commit()  # Save changes
            conn.close()  # Close DB
            return clean_session_id  # Return sanitized session ID
            
        except Exception as e:
            print(f"❌ Error creating/getting session: {e}")  # Print error if DB fails
            return sanitize_text(session_id)  # Still return sanitized ID
    
    # Add conversation to memory with enhanced safety
    def add_conversation(self, session_id: str, user_message: str, bot_response: str, context_used: str = ""):
        try:
            conn = sqlite3.connect(MEMORY_DB)  # Connect to DB
            cursor = conn.cursor()  # Cursor for SQL
            
            # Clean all inputs
            clean_session_id = sanitize_text(session_id)
            clean_user_message = sanitize_text(user_message)
            clean_bot_response = sanitize_text(bot_response)
            clean_context_used = sanitize_text(context_used)
            
            # Insert new conversation into DB
            cursor.execute('''
                INSERT INTO conversations (session_id, timestamp, user_message, bot_response, context_used)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                clean_session_id,
                datetime.now().isoformat(),  # Current timestamp
                clean_user_message,
                clean_bot_response,
                clean_context_used
            ))

            # Update session activity and message count
            cursor.execute('''
                UPDATE sessions SET last_active = ?, total_messages = total_messages + 1
                WHERE session_id = ?
            ''', (datetime.now().isoformat(), clean_session_id))
            
            conn.commit()  # Save changes
            print(f"✅ Conversation added to memory successfully")
            
        except Exception as e:
            print(f"❌ Error adding conversation to memory: {e}")  # Print error
        finally:
            try:
                conn.close()  # Always close DB
            except:
                pass

    # Get recent conversations for context
    def get_recent_conversations(self, session_id: str, limit: int = 3) -> List[Dict]:
        try:
            conn = sqlite3.connect(MEMORY_DB)  # Connect to DB
            cursor = conn.cursor()
            
            clean_session_id = sanitize_text(session_id)  # Clean session ID
            
            # Get most recent conversations for this session
            cursor.execute('''
                SELECT user_message, bot_response, timestamp
                FROM conversations
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (clean_session_id, limit))
            
            conversations = []  # Store results here
            for row in cursor.fetchall():  # Loop through retrieved rows
                user_msg = sanitize_text(row[0]) if row[0] else ""  # Clean user message
                bot_msg = sanitize_text(row[1]) if row[1] else ""   # Clean bot response
                
                # Shorten long text so logs are easier to read
                conversations.append({
                    'user_message': user_msg[:50] + "..." if len(user_msg) > 50 else user_msg,
                    'bot_response': bot_msg[:100] + "..." if len(bot_msg) > 100 else bot_msg,
                    'timestamp': row[2]
                })
            
            conn.close()  # Close DB
            return list(reversed(conversations))  # Return in chronological order
            
        except Exception as e:
            print(f"❌ Error retrieving conversations: {e}")
            return []  # Return empty list if error

    def add_custom_info(self, topic: str, information: str):
        """Add custom information with sanitization"""
        try:
            clean_topic = sanitize_text(topic)  # Clean topic
            clean_info = sanitize_text(information)  # Clean info
            
            # Unique ID: topic + number
            info_id = f"{clean_topic.lower().replace(' ', '_')}_{len(self.custom_info) + 1}"
            
            # Save custom info in memory
            self.custom_info[info_id] = {
                'topic': clean_topic,
                'information': clean_info,
                'added_at': datetime.now().isoformat()
            }
            self.save_custom_info()  # Save to file
            print(f"✅ Added custom info for: {clean_topic}")
            
        except Exception as e:
            print(f"❌ Error adding custom info: {e}")

    # Get relevant custom information for the query
    def get_relevant_custom_info(self, query: str) -> str:
        try:
            relevant = []  # Store matches
            clean_query = sanitize_text(query.lower())  # Clean query text
            
            # Loop through stored custom info
            for info in self.custom_info.values():
                clean_topic = sanitize_text(info['topic'].lower())
                
                # If topic or any word in topic matches query
                if (clean_topic in clean_query or 
                    any(word in clean_query for word in clean_topic.split())):
                    clean_information = sanitize_text(info['information'])
                    relevant.append(f"ADDITIONAL INFO: {info['topic']}: {clean_information}")
            
            return "\n".join(relevant) if relevant else ""  # Return joined string or empty
            
        except Exception as e:
            print(f"❌ Error getting custom info: {e}")
            return ""

# Initialize memory system
memory = ChatbotMemory()

# Setup admin routes (this adds all admin endpoints)
setup_admin_routes(app, memory, LOG_FILE, MEMORY_DB)

# Query expansion with GSU terms
GSU_SYNONYMS = {
    "core values": ["values", "principles", "beliefs", "ethics"],
    "vision": ["vision statement", "future goals", "aspirations"],
    "mission": ["mission statement", "purpose", "objectives"],
    "admission": ["admissions", "enrollment", "entry", "application"],
    "requirements": ["prerequisites", "criteria", "qualifications"],
    "grades": ["grading", "marks", "scores", "assessment"],
    "policies": ["rules", "regulations", "procedures"],
    "student life": ["campus life", "activities", "organizations"],
    "fees": ["tuition", "payment", "cost", "expenses"],
    "faculty": ["teachers", "professors", "instructors"],
}

# Expand query with relevant synonyms
def expand_query(query):
    expanded = [query]  # Start list with the original query
    query_lower = query.lower()  # Convert query to lowercase for case-insensitive matching
    
    # Loop through each term and its synonyms in GSU_SYNONYMS
    for term, synonyms in GSU_SYNONYMS.items():
        # If the term or any of its synonyms appear in the query
        if term in query_lower or any(syn in query_lower for syn in synonyms):
            expanded.extend(synonyms)  # Add those synonyms to the expanded list
    
    # Return a space-separated string of unique terms (original + synonyms)
    return " ".join(list(set(expanded)))

# Extract important keywords, excluding common stop words
def extract_keywords(query):
    # Define common stop words that should be ignored
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
        'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'
    }
    
    # Use regex to extract words (alphanumeric sequences) in lowercase
    words = re.findall(r'\b\w+\b', query.lower())
    
    # Keep only words that are not stop words and have length > 2
    return [w for w in words if w not in stop_words and len(w) > 2]

# Multi-strategy retrieval for comprehensive results
async def smart_retrieval(query, retriever, handbook_text):
    all_chunks = set()  # Stores retrieved chunks (avoids duplicates using a set)
    
    # First strategy: use vector retriever (if available)
    if retriever:
        try:
            # Run retriever in a background thread (so async is not blocked)
            docs = await asyncio.get_event_loop().run_in_executor(executor, retriever.invoke, query)
            
            # Add each retrieved doc to all_chunks with its content + chunk_id
            for doc in docs:
                all_chunks.add((doc.page_content, doc.metadata.get('chunk_id', 0)))
            
            # Expand query with synonyms
            expanded = expand_query(query)
            if expanded != query:  # Only if query was expanded
                # Run retriever again with expanded query
                docs = await asyncio.get_event_loop().run_in_executor(executor, retriever.invoke, expanded)
                for doc in docs:
                    all_chunks.add((doc.page_content, doc.metadata.get('chunk_id', 0)))
        except Exception as e:
            print(f"Retrieval error: {e}")  # Log retrieval errors, don’t crash
    
    # Second strategy: keyword-based fallback search in raw handbook text
    if len(all_chunks) < 8 and handbook_text:  # Only if retriever gave few results
        keywords = extract_keywords(query)  # Extract keywords from query
        # Split handbook into sentences longer than 20 characters
        sentences = [s.strip() for s in handbook_text.split('.') if len(s.strip()) > 20]
        
        scored_sentences = []  # Will store (sentence, score)
        for sentence in sentences:
            sentence_lower = sentence.lower()
            score = 0
            
            # Boost score if entire query appears in sentence
            if query.lower() in sentence_lower:
                score += 25
            
            # Boost score for each keyword match
            for keyword in keywords:
                if keyword in sentence_lower:
                    score += len(keyword) + 2  # Longer keyword = more weight
            
            # Only keep sentences with decent score
            if score > 5:
                scored_sentences.append((sentence, score))
        
        # Sort by score descending
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        # Keep top 15 relevant sentences
        for sentence, _ in scored_sentences[:15]:
            all_chunks.add((sentence, 999))  # Use 999 as chunk_id for keyword matches
    
    # Convert all collected chunks into Document objects
    result_docs = []
    for content, chunk_id in all_chunks:
        doc = Document(page_content=content, metadata={"chunk_id": chunk_id})
        result_docs.append(doc)
    
    return result_docs  # Final set of retrieved docs

# Build context including memory and custom info (minimal conversation history)
def build_context_with_memory(docs, query, session_id, max_length=8000):
    # List to hold all different parts of the context (history, custom info, handbook)
    context_parts = []
    
    # === 1. Add minimal conversation history (only if directly relevant) ===
    recent_conversations = memory.get_recent_conversations(session_id, limit=2)  # Get last 2 convos for this session
    if recent_conversations:
        # Lowercase query for easier keyword matching
        query_lower = query.lower()
        relevant_history = []  # Will store conversations relevant to current query
        
        # Check if user query words appear in previous messages or responses
        for conv in recent_conversations:
            if (any(word in conv['user_message'].lower() for word in query_lower.split()) or 
                any(word in conv['bot_response'].lower() for word in query_lower.split())):
                relevant_history.append(conv)  # Keep only relevant conversations
        
        # If relevant history exists, add it to context
        if relevant_history:
            history = "RELEVANT CONTEXT:\n"
            for conv in relevant_history[-1:]:  # Take only the most recent relevant conversation
                # Clean the conversation text before storing
                clean_user_msg = sanitize_text(conv['user_message'])
                history += f"Previous: {clean_user_msg}\n"
            context_parts.append(history)  # Add conversation history to context
    
    # === 2. Add custom user information ===
    custom_info = memory.get_relevant_custom_info(query)  # Look for user-provided info related to query
    if custom_info:
        clean_custom_info = sanitize_text(custom_info)  # Clean it before use
        context_parts.append(f"ADDITIONAL INFORMATION:\n{clean_custom_info}\n")
    
    # === 3. Add handbook context from retrieved documents ===
    if docs:
        keywords = extract_keywords(query)  # Extract keywords for scoring
        scored_docs = []  # Will hold (doc, score) pairs
        
        # Score each retrieved document
        for doc in docs:
            clean_content = sanitize_text(doc.page_content)  # Clean text content
            content_lower = clean_content.lower()
            score = 0
            
            # Strong score boost if full query is present
            if query.lower() in content_lower:
                score += 30
            
            # Moderate boost for keyword matches
            for keyword in keywords:
                if keyword in content_lower:
                    score += len(keyword) + 5  # Longer keyword = more weight
            
            # Save the document with its score
            clean_doc = Document(page_content=clean_content, metadata=doc.metadata)
            scored_docs.append((clean_doc, score))
        
        # Sort documents by score (highest first)
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        handbook_context = "HANDBOOK INFORMATION:\n"  # Label for handbook section
        # Track how much text is already in context (avoid exceeding max_length)
        current_length = sum(len(part) for part in context_parts)
        
        # Add top scored handbook sections until space runs out
        for i, (doc, score) in enumerate(scored_docs):
            chunk_text = doc.page_content.strip()  # Clean up whitespace
            remaining = max_length - current_length - len(handbook_context)  # Remaining budget
            
            # Stop if context is nearly full
            if remaining < 200:
                break
            
            # Truncate if text chunk is too long
            if len(chunk_text) > remaining:
                sentences = chunk_text.split('.')  # Split into sentences
                truncated = ""
                for sentence in sentences:
                    # Add sentence if it fits within remaining space
                    if len(truncated + sentence + '.') <= remaining - 50:
                        truncated += sentence + '.'
                    else:
                        break
                # Add "..." if truncated
                chunk_text = truncated + " [...]" if truncated else chunk_text[:remaining-50] + "..."
            
            # Assign relevance label based on score
            relevance = "HIGH" if score > 20 else "MEDIUM" if score > 10 else "LOW"
            handbook_context += f"[Section {i+1} - {relevance}]:\n{chunk_text}\n\n"
            current_length += len(chunk_text) + 50  # Update current length used
        
        # Add handbook info to context
        context_parts.append(handbook_context)
    
    # === 4. Finalize context ===
    full_context = "\n".join(context_parts)  # Combine all sections into one text
    return sanitize_text(full_context)  # Return clean, safe context

def log_query(query_text: str, answer_text: str, answered_flag: bool, chunks_found: int):
    """Log queries and answers to CSV with resolved date support"""

    clean_query = sanitize_text(query_text)
    clean_answer = sanitize_text(answer_text)
    
    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        file_exists = os.path.isfile(LOG_FILE)
        
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as csvfile:
            # Include resolved_date in new entries
            fieldnames = ["timestamp", "query_text", "answer_text", "answered", "chunks_found", "resolved_date"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow({
                "timestamp": datetime.now().isoformat(),
                "query_text": clean_query.strip(),
                "answer_text": clean_answer.strip()[:500],
                "answered": answered_flag,
                "chunks_found": chunks_found,
                "resolved_date": ""  # Empty for new entries
            })
            
        print(f"✅ CSV log successful")
    except Exception as csv_error:
        print(f"❌ CSV logging error: {csv_error}")

    # ---- SQLite Logging ----
    try:
        # Connect to SQLite memory database
        conn = sqlite3.connect(MEMORY_DB)
        cur = conn.cursor()
        
        # Create queries table if it doesn’t exist yet
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,  -- unique row ID
                timestamp TEXT,                        -- when the query was asked
                query_text TEXT,                       -- sanitized query
                answer_text TEXT,                      -- sanitized answer
                answered INTEGER,                      -- 1 if answered, 0 otherwise
                chunks_found INTEGER                   -- number of retrieved chunks
            )
            """
        )

        # Insert a new row with query details (uses parameterized query to prevent SQL injection)
        cur.execute(
            "INSERT INTO queries (timestamp, query_text, answer_text, answered, chunks_found) VALUES (?, ?, ?, ?, ?)",
            (
                datetime.now().isoformat(),  # current timestamp
                clean_query,                 # sanitized query
                clean_answer,                # sanitized answer
                int(answered_flag),          # convert boolean to integer
                chunks_found                 # number of retrieved chunks
            )
        )

        # Save changes to the database
        conn.commit()
        print(f"✅ SQLite log successful")  # confirmation message
        
    except Exception as sqlite_error:
        # Report SQLite logging errors but don't crash the app
        print(f"❌ SQLite logging error: {sqlite_error}")
    finally:
        # Always try to close the database connection safely
        try:
            conn.close()
        except:
            pass

def load_embeddings_and_db():
    """Load embeddings and database, create if doesn't exist"""
    try:
        import os
        
        # Load embeddings model
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("✅ Embeddings model loaded")
        
        db_path = "data/chroma_db"
        
        # Check if database exists
        if os.path.exists(db_path) and os.listdir(db_path):
            print("📂 Loading existing Chroma database...")
            db = Chroma(persist_directory=db_path, embedding_function=embeddings)
        else:
            print("🔨 Creating new Chroma database (first time)...")
            
            # Make sure the handbook is loaded
            if not HANDBOOK_TEXT:
                print("❌ Cannot create database: Handbook not loaded")
                return None, None, None
            
            # Split text into chunks
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_text(HANDBOOK_TEXT)
            print(f"📄 Split into {len(chunks)} chunks")
            
            # Create database from chunks
            db = Chroma.from_texts(
                texts=chunks,
                embedding=embeddings,
                persist_directory=db_path
            )
            db.persist()
            print("💾 Database created and saved")
        
        # Create retriever
        retriever = db.as_retriever(
            search_type="mmr", 
            search_kwargs={"k": 15, "fetch_k": 35, "lambda_mult": 0.5}
        )
        
        print("✅ Database and retriever ready")
        return embeddings, db, retriever
    
    except Exception as e:
        print(f"💥 Database loading error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def load_handbook():
    """Load handbook text from PDF (simple extraction)"""
    import os
    
    print(f"📂 Looking for PDF at: {HANDBOOK_FILE}")
    print(f"📂 Current directory: {os.getcwd()}")
    print(f"📂 PDF exists? {os.path.exists(HANDBOOK_FILE)}")
    
    if not os.path.exists(HANDBOOK_FILE):
        print(f"❌ PDF file not found!")
        print(f"📂 Files in current directory: {os.listdir('.')}")
        return ""
    
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(HANDBOOK_FILE)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        print(f"❌ Error reading handbook PDF: {e}")
        import traceback
        traceback.print_exc()
        return ""

# Register a startup event for FastAPI (runs automatically when the server starts)
@app.on_event("startup")
async def startup_event():
    global HANDBOOK_TEXT, embeddings, db, retriever
    
    print("🚀 Server starting...")
    
    # Initialize as None so server can start
    HANDBOOK_TEXT = ""
    embeddings = None
    db = None
    retriever = None
    
    # Load everything in background WITHOUT BLOCKING
    asyncio.create_task(delayed_initialization())
    
    print("✅ Server started! Initialization running in background...")

async def delayed_initialization():
    """Run after server starts"""
    global HANDBOOK_TEXT, embeddings, db, retriever
    
    # Wait 5 seconds for server to fully start
    await asyncio.sleep(5)
    
    loop = asyncio.get_event_loop()
    
    try:
        print("📚 Loading handbook...")
        HANDBOOK_TEXT = await loop.run_in_executor(executor, lambda: load_handbook())
        print(f"✅ Handbook loaded: {len(HANDBOOK_TEXT)} characters")
    except Exception as e:
        HANDBOOK_TEXT = ""
        print(f"❌ Error loading handbook: {e}")
    
    try:
        print("🧠 Loading embeddings and database (this may take 3-5 minutes)...")
        embeddings, db, retriever = await loop.run_in_executor(
            executor, lambda: load_embeddings_and_db()
        )
        
        if all([embeddings, db, retriever]):
            print("🎉 All resources ready!")
        else:
            print("❌ Resource initialization incomplete")
    except Exception as e:
        embeddings, db, retriever = None, None, None
        print(f"❌ Error initializing embeddings/db: {e}")
        import traceback
        traceback.print_exc()

async def initialize_resources():
    """Load resources in background without blocking startup"""
    global HANDBOOK_TEXT, embeddings, db, retriever
    
    loop = asyncio.get_event_loop()
    
    try:
        print("📚 Loading handbook...")
        HANDBOOK_TEXT = await loop.run_in_executor(executor, lambda: load_handbook())
        print(f"✅ Handbook loaded: {len(HANDBOOK_TEXT)} characters")
    except Exception as e:
        HANDBOOK_TEXT = ""
        print(f"❌ Error loading handbook: {e}")
    
    try:
        print("🧠 Loading embeddings and database...")
        embeddings, db, retriever = await loop.run_in_executor(
            executor, lambda: load_embeddings_and_db()
        )
        
        if all([embeddings, db, retriever]):
            print("🎉 All resources ready!")
        else:
            print("❌ Resource initialization incomplete")
    except Exception as e:
        embeddings, db, retriever = None, None, None
        print(f"❌ Error initializing embeddings/db: {e}")

# Add these endpoints
@app.get("/")
async def root():
    return {"status": "Server is running!", "message": "Hello from Render"}

@app.get("/initialize")
async def initialize():
    """Manually trigger initialization"""
    try:
        asyncio.create_task(initialize_resources())
        return {"message": "Initialization started in background"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/status")
async def status():
    return {
        "handbook_loaded": bool(HANDBOOK_TEXT),
        "embeddings_ready": embeddings is not None,
        "db_ready": db is not None,
        "retriever_ready": retriever is not None
    }

# Define a GET endpoint at /report (used to fetch usage statistics)
@app.get("/report")
def get_report():
    """Public report endpoint with simplified answered/not answered breakdown"""
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
    except FileNotFoundError:
        return {
            "total_queries": 0,
            "answered_queries": 0,
            "not_answered_queries": 0,
            "most_frequent_questions": []
        }

    if not reader:
        return {
            "total_queries": 0,
            "answered_queries": 0,
            "not_answered_queries": 0,
            "most_frequent_questions": []
        }

    total = len(reader)
    
    # Simple answered/not answered based on CSV field
    answered = sum(1 for r in reader if (r.get("answered") or "").strip().lower() in ["true", "1", "yes"])
    not_answered = total - answered

    # Count most frequent questions
    query_counter = Counter(
        (r.get("query_text") or "").strip().lower()
        for r in reader if r.get("query_text")
    )
    top_faqs = query_counter.most_common(10)

    return {
        "total_queries": total,
        "answered_queries": answered,
        "not_answered_queries": not_answered,
        "accuracy_rate": (answered / total * 100) if total > 0 else 0,
        "most_frequent_questions": [
            {"question": q, "count": c} for q, c in top_faqs
        ]
    }

@traceable(name="chat_pipeline")
async def run_chat_pipeline(clean_query: str, clean_context: str, clean_prompt: str, executor, model):
    """Run Gemini with LangSmith tracing enabled"""
    
    response = await asyncio.get_event_loop().run_in_executor(
        executor, model.generate_content, clean_prompt
    )
    
    # After the traced execution, query for the most recent run
    run_id = None
    try:
        # Wait a moment for trace to be created
        await asyncio.sleep(0.1)
        
        # Query LangSmith for recent runs
        recent_runs = list(client.list_runs(
            project_name=os.getenv("LANGCHAIN_PROJECT", "student-handbook-backend"),
            limit=1,
            execution_order=1  # Get most recent
        ))
        
        if recent_runs:
            run_id = str(recent_runs[0].id)
            print(f"✅ Found recent trace_id: {run_id}")
    except Exception as e:
        print(f"⚠️ Could not query recent runs: {e}")
    
    return response.text if hasattr(response, "text") else str(response), run_id

@traceable(name="retrieval_step")
async def traced_retrieval(query, retriever, handbook_text):
    """Retrieve handbook chunks for LangSmith tracing (safe for string outputs)"""
    docs = await smart_retrieval(query, retriever, handbook_text)

    # If smart_retrieval already returns strings
    results = []
    for chunk in docs:
        results.append({
            "content": chunk,   # the text chunk
            "metadata": {}      # no metadata available
        })
    return results

@traceable(name="context_builder")
def traced_context(docs, query, session_id):
    """Build context with LangSmith tracing"""
    text_chunks = [doc["content"] for doc in docs]
    return build_context_with_memory(text_chunks, query, session_id)

@app.post("/chat")
async def chat(request: Request):
    """Main chat endpoint with memory and timeout handling"""
    global retriever, HANDBOOK_TEXT, model
    start_time = time.time()
    
    try:
        if not model:
            return {"answer": "Gemini API is not configured."}
        
        data = await request.json()
        query = data.get("query", "").strip()
        session_id = data.get("session_id", "default_session")
        
        if not query:
            return {"answer": "Please ask a question."}
        
        memory.get_or_create_session(session_id)
        print(f"🔍 Processing focused query: {query} [Session: {session_id}]")
        
        if time.time() - start_time > 18:
            return {"answer": "Please check your internet connection and try again."}
        
        # ---- Retrieval (traced) ----
        docs = await traced_retrieval(query, retriever, HANDBOOK_TEXT)
        chunks_found = len(docs)
        
        if time.time() - start_time > 18:
            return {"answer": "Please check your internet connection and try again."}
        
        if not docs:
            no_answer_msg = "I couldn't find relevant information in the handbook for your question."
            memory.add_conversation(session_id, query, no_answer_msg)
            await asyncio.get_event_loop().run_in_executor(
                executor, log_query, query, no_answer_msg, False, 0
            )
            return {"answer": no_answer_msg}
        
        # ---- Context (traced) ----
        context = traced_context(docs, query, session_id)

        clean_context = sanitize_text(context)
        clean_query = sanitize_text(query)
        
        # Build final prompt
        prompt_parts = [
            "You are a GSU student handbook assistant. Provide a focused, direct answer to the student's question.",
            "",
            "IMPORTANT INSTRUCTIONS:",
            "- DO NOT mention previous conversations or greet the user",
            "- Focus ONLY on answering the current question", 
            "- Always prioritize ADDITIONAL INFORMATION over handbook information when they conflict",
            "- Be concise but complete",
            "- Use structured formatting for lists/numbers/categories",
            "",
            f"Student Question: {clean_query}",
            "",
            "Context:",
            clean_context,
            "",
            "Direct Answer (use proper formatting for lists and structured content):"
        ]
        clean_prompt = sanitize_text("\n".join(prompt_parts))
        if len(clean_prompt) > 30000:
            clean_prompt = clean_prompt[:30000] + "..."
        
        response_text, run_id = await run_chat_pipeline(clean_query, clean_context, clean_prompt, executor, model)

        # Store the run_id for this session
        if run_id:
            last_run_ids[session_id] = run_id
            print(f"🔗 Stored run_id {run_id} for session {session_id}")
        
        # Process response
        raw_answer = response_text
        answer = sanitize_text(raw_answer)
        if not answer:
            answer = "I couldn't generate a proper response. Please try rephrasing your question."
        print(f"DEBUG - Answer processed successfully, length: {len(answer)}")
        
        # Save conversation in memory
        memory.add_conversation(session_id, query, answer, clean_context[:200])
        print(f"✅ Response generated: {len(answer)} characters")
        
        # Answered flag
        answered = True
        if not answer.strip():
            answered = False
        elif any(phrase in answer.lower() for phrase in [
            "does not specify","not specified","doesn't specify","not mentioned",
            "not provided","doesn't mention","no information","not available",
            "not found","not included","doesn't include","not detailed",
            "doesn't contain","not contain","no details","not clear",
            "doesn't state","not stated","not outlined","not listed",
            "no specific","not specifically","doesn't provide specific",
            "is not available","are not available","not accessible",
            "couldn't find","error","technical difficulties","try again"
        ]):
            answered = False
        
        # Log query + answer
        await asyncio.get_event_loop().run_in_executor(
            executor, log_query, query, answer, answered, chunks_found
        )
        
        return {"answer": answer, "session_id": session_id}
    
    except Exception as e:
        print(f"❌ Unexpected error: {repr(e)}")
        error_msg = "Please check your internet connection and try again." if time.time() - start_time > 20 \
            else "I encountered an unexpected error. Please try again or contact support."
        
        try:
            session_id = data.get("session_id", "default_session")
            query = data.get("query", "")
            memory.add_conversation(session_id, query, error_msg)
            await asyncio.get_event_loop().run_in_executor(
                executor, log_query, query, error_msg, False, 0
            )
        except:
            pass
        
        return {"answer": error_msg}

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackSubmission):
    """Submit user feedback + send to LangSmith"""
    try:
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
        
        # Check if feedback file exists
        file_exists = os.path.isfile(FEEDBACK_FILE)
        
        with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["timestamp", "feedback_text", "rating", "user_type", "session_id"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow({
                "timestamp": datetime.now().isoformat(),
                "feedback_text": sanitize_text(feedback.feedback_text),
                "rating": feedback.rating,
                "user_type": feedback.user_type,
                "session_id": feedback.session_id
            })
        
        # ---- Send to LangSmith as a custom event ----
        try:
            # Log feedback as a custom event/annotation
            feedback_data = {
                "feedback_text": sanitize_text(feedback.feedback_text),
                "rating": feedback.rating,
                "normalized_score": feedback.rating / 5.0,
                "user_type": feedback.user_type,
                "session_id": feedback.session_id,
                "timestamp": datetime.now().isoformat()
            }
            
            # Use LangSmith's logging to track feedback
            from langsmith import traceable
            
            @traceable(name="user_feedback_submission")
            def log_feedback_event(data):
                return {"status": "feedback_logged", "data": data}
            
            log_feedback_event(feedback_data)
            print(f"✅ Feedback logged to LangSmith: {feedback.rating}/5 stars")
            
        except Exception as ls_err:
            print(f"⚠️ Could not send feedback to LangSmith: {ls_err}")
        
        print(f"✅ Feedback submitted locally: {feedback.rating}/5 stars")
        return {"message": "Feedback submitted successfully", "status": "success"}
    
    except Exception as e:
        print(f"❌ Error saving feedback: {e}")
        return {"message": "Error submitting feedback", "status": "error"}