# api.py

# python modules
import os                   # Used for file paths and environment variables
import csv                  # For reading and writing CSV files
import json                 # For handling JSON data
import re                   # Regular expressions for text pattern matching
import time                 # Used for tracking time or delays
import asyncio              # Enables asynchronous operations
import sqlite3              # Connects and interacts with SQLite database
from datetime import datetime  # To handle dates and times
from concurrent.futures import ThreadPoolExecutor  # For running tasks in background threads
from collections import Counter  # Helps count items (like top FAQ questions)
from typing import List, Dict, Optional  # Type hints for better code readability
from urllib.parse import quote  # Encodes URLs safely

from dotenv import load_dotenv  # Used to load .env configuration files
load_dotenv()  # Loads environment variables (API keys, paths, etc.) from .env file

# fastapi
from fastapi import FastAPI, Request  # FastAPI for backend API creation
from fastapi.middleware.cors import CORSMiddleware  # Allows frontend (Flutter app) to access the backend
from pydantic import BaseModel  # Used to define structured request/response data

# gemini
import google.generativeai as genai  # Google Gemini API integration for LLM responses

# langsmith
from langsmith import traceable, Client  # Used for tracking AI performance and debugging
from langsmith.run_helpers import get_current_run_tree  # Helps trace detailed run information
client = Client()  # Creates a client to log or monitor model performance in LangSmith

# langchain and chroma vector db
from langchain_huggingface import HuggingFaceEmbeddings  # Converts text into embeddings (numerical form)
from langchain_chroma import Chroma  # Local vector database (stores and retrieves documents)
from langchain.docstore.document import Document  # Represents text documents in LangChain

# admin
from admin import setup_admin_routes  # Imports routes for admin panel (feedback, reports, etc.)

api_key = os.getenv("LANGCHAIN_API_KEY")  # gets the key from the .env file
print(f"Using API Key: {api_key[:20]}...{api_key[-10:] if api_key else 'NONE'}")  # shows part of the key
print(f"Key length: {len(api_key) if api_key else 0}")  # shows how long the key is
print(f"LANGCHAIN_TRACING_V2: {os.getenv('LANGCHAIN_TRACING_V2')}")  # shows tracing setting
print(f"LANGCHAIN_PROJECT: {os.getenv('LANGCHAIN_PROJECT')}")  # shows project name

# Test if connection works
def test_langsmith_connection():
    try:
        test_client = Client()  # tries to connect
        projects = list(test_client.list_projects(limit=1))  # checks if any project can be found
        print("✅ LangSmith connection successful")
        return True  # returns true if it works
    except Exception as e:
        print(f"❌ LangSmith connection failed: {e}")
        return False

test_langsmith_connection()  # runs the test when app starts

last_run_ids = {}  # keeps a list of user sessions

class FeedbackSubmission(BaseModel):
    feedback_text: str  # user’s written feedback
    rating: int         # number rating from user
    user_type: str      # who gave the feedback (student, admin, etc.)
    session_id: Optional[str] = None  # used to track feedback per session

# Set up Gemini
if os.environ.get("GENAI_API_KEY"):  # checks if a key exists first
    genai.configure(api_key=os.environ["GENAI_API_KEY"])  # connects Gemini using the key
    generation_config = genai.types.GenerationConfig(
        temperature=0.1, top_p=0.8, top_k=40, max_output_tokens=2048
        # temperature=0.1 → makes answers steady and focused
        # top_p=0.8 → keeps only the most likely words
        # top_k=40 → limits how many choices it looks at
        # max_output_tokens=2048 → limits how long the answer can be
    )
    model = genai.GenerativeModel("gemini-2.5-flash", generation_config=generation_config)  # loads the model
else:  # runs when no API key is found
    print("⚠️ GENAI_API_KEY environment variable not set")  # warning message
    model = None  # prevents crash if no key is set

app = FastAPI()  # starts the main app
app.add_middleware(  # allows the app to receive data from anywhere
    CORSMiddleware,
    allow_origins=["*"],  # allows all sources
    allow_credentials=True,
    allow_methods=["*"],  # allows all actions (GET, POST, etc.)
    allow_headers=["*"]   # allows all headers
)

# File and data setup
LOG_FILE = "data/query_log.csv"  # where chat history is saved
FEEDBACK_FILE = "data/feedback.csv"  # where feedback is stored
HANDBOOK_FILE = "data/handbook.pdf"  # the student handbook file
HANDBOOK_TEXT = ""  # will store handbook text after loading
MEMORY_DB = "data/chatbot_memory.db"  # stores chat memory
CUSTOM_INFO_FILE = "data/custom_info.json"  # holds extra handbook data
executor = ThreadPoolExecutor()  # runs heavy tasks in the background

# Main tools (will be set up later)
embeddings = None  # converts text to numbers
db = None           # stores those numbers (database)
retriever = None    # helps find related info quickly

last_run_ids = {}  # keeps track of sessions


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
        # cursor is like a pen — it’s used to write or read data from the database.
        cursor = conn.cursor()  # Create a cursor object to run SQL commands

        # conversations table → stores actual chat messages
        # sessions table → stores info about each user’s chat session (like a summary)
        
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

    
    # Every time someone chats with the bot
    def get_or_create_session(self, session_id: str) -> str:
        try:
            conn = sqlite3.connect(MEMORY_DB)  # Connect to DB
            cursor = conn.cursor()  # Create cursor
            
            clean_session_id = sanitize_text(session_id)  # Clean session ID text
            
            # If the session already exists, it just reuses it
            cursor.execute('SELECT session_id FROM sessions WHERE session_id = ?', (clean_session_id,))
            if not cursor.fetchone():  # If it’s a new user or session, the chatbot creates a fresh record. 
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
        
    
    # the chatbot can later refer back to recent messages if the user continues asking related questions.
    def add_conversation(self, session_id: str, user_message: str, bot_response: str, context_used: str = ""):
        try:
            conn = sqlite3.connect(MEMORY_DB)  # Connect to DB
            cursor = conn.cursor()  # Cursor for SQL
            
            # Clean all inputs before saving anything
            clean_session_id = sanitize_text(session_id)
            clean_user_message = sanitize_text(user_message)
            clean_bot_response = sanitize_text(bot_response)
            clean_context_used = sanitize_text(context_used)
            
            # Adds a new record to the conversations table
            # The ? symbols are placeholders for actual values that will be inserted later.
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

            # updates the chatbot’s session log every time you talk to it.
            # WHERE session_id = ? → only update the record belonging to the current chat session.
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
            
            conversations = []  # later used by the bot to recall relevant past replies

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


    # checks if the user’s question relates to any of the custom information stored by an admin.
    def get_relevant_custom_info(self, query: str) -> str:
        try:
            relevant = []  # Store matches
            clean_query = sanitize_text(query.lower())  # Clean query text
            
            # self.custom_info is a dictionary that holds all manually added data (from custom_info.json)
            for info in self.custom_info.values():
                clean_topic = sanitize_text(info['topic'].lower())
                
                # even partial matches (like one word) will trigger a match
                if (clean_topic in clean_query or 
                    any(word in clean_query for word in clean_topic.split())):
                    clean_information = sanitize_text(info['information'])
                    # adds a formatted text to the list
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
        'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'what'
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
            all_chunks.add((sentence, 999))  # Uses chunk_id = 999 to label these as keyword-based chunks
    
    # all found chunks from both strategies are converted into standard Document objects that LangChain can process
    result_docs = []
    for content, chunk_id in all_chunks:
        doc = Document(page_content=content, metadata={"chunk_id": chunk_id})
        result_docs.append(doc)
    
    return result_docs  # Final set of retrieved docs


# Build context including memory and custom info (minimal conversation history)
# contex = the information the chatbot gives the AI model before it generates an answer
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
    """Saves every question and answer into a file and database"""

    clean_query = sanitize_text(query_text)  # cleans the question
    clean_answer = sanitize_text(answer_text)  # cleans the answer
    
    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)  # makes sure the folder exists
        file_exists = os.path.isfile(LOG_FILE)  # checks if file already exists
        
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as csvfile:  # opens file for adding logs
            fieldnames = ["timestamp", "query_text", "answer_text", "answered", "chunks_found", "resolved_date"]  # columns for CSV
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:  # if file is new
                writer.writeheader()  # adds column titles

            writer.writerow({  # adds a new log row
                "timestamp": datetime.now().isoformat(),  # date and time now
                "query_text": clean_query.strip(),        # cleaned question
                "answer_text": clean_answer.strip()[:500], # cleaned answer (shortened)
                "answered": answered_flag,                # True or False if answered
                "chunks_found": chunks_found,             # number of matching text parts found
                "resolved_date": ""                       # empty for now, can be updated later
            })
            
        print(f"✅ CSV log successful")  # tells us it worked
    except Exception as csv_error:
        print(f"❌ CSV logging error: {csv_error}")  # shows an error but keeps running

    # ---- Database Logging ----
    try:
        conn = sqlite3.connect(MEMORY_DB)  # opens local database
        cur = conn.cursor()  # lets us run SQL commands
        
        cur.execute(  # creates table if not already there
            """
            CREATE TABLE IF NOT EXISTS queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,  -- unique number for each record
                timestamp TEXT,                        -- when the question was asked
                query_text TEXT,                       -- question text
                answer_text TEXT,                      -- chatbot’s answer
                answered INTEGER,                      -- 1 if answered, 0 if not
                chunks_found INTEGER                   -- how many chunks were used
            )
            """
        )

        # adds a new record
        cur.execute(
            "INSERT INTO queries (timestamp, query_text, answer_text, answered, chunks_found) VALUES (?, ?, ?, ?, ?)",
            (
                datetime.now().isoformat(),  # date and time
                clean_query,                 # cleaned question
                clean_answer,                # cleaned answer
                int(answered_flag),          # turns True/False into 1/0
                chunks_found                 # number of text chunks found
            )
        )

        conn.commit()  # saves the record
        print(f"✅ SQLite log successful")  # success message
        
    except Exception as sqlite_error:
        print(f"❌ SQLite logging error: {sqlite_error}")  # error message
    finally:
        try:
            conn.close()  # closes the database safely
        except:
            pass  # ignores close errors


def load_embeddings_and_db():
    """Loads the AI’s memory (embeddings) and database. Creates new ones if missing."""
    try:
        # Loads a model that turns text into numbers so the AI can compare meanings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("✅ Embeddings model loaded")
        
        db_path = "data/chroma_db"  # where the database will be stored
        
        # Checks if a database already exists
        if os.path.exists(db_path) and os.listdir(db_path):
            print("📂 Loading existing Chroma database...")
            db = Chroma(persist_directory=db_path, embedding_function=embeddings)  # loads saved data
        else:
            print("🔨 Creating new Chroma database (first time)...")
            
            # Makes sure the handbook text is already loaded
            if not HANDBOOK_TEXT:
                print("❌ Cannot create database: Handbook not loaded")
                return None, None, None  # stops if no handbook text found
            
            # Breaks the handbook text into smaller parts
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            # AI reads better in small sections
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,  # each part has 1000 characters
                chunk_overlap=200  # overlaps a little so no information is lost
            )
            chunks = text_splitter.split_text(HANDBOOK_TEXT)
            print(f"📄 Split into {len(chunks)} chunks")  # tells how many pieces were made
            
            # Creates a database from the text pieces
            db = Chroma.from_texts(
                texts=chunks,             # list of handbook parts
                embedding=embeddings,     # uses the embeddings model
                persist_directory=db_path # saves it to a folder
            )
            db.persist()  # makes sure it's saved on disk
            print("💾 Database created and saved")
        
        # Creates a retriever — lets the AI look up related handbook info when asked
        retriever = db.as_retriever(
            # Maximal Marginal Relevance
            search_type="mmr", # Finds results that are relevant to the query But also diverse
            search_kwargs={"k": 15, "fetch_k": 35, "lambda_mult": 0.5}
        )
        
        print("✅ Database and retriever ready")  # everything worked
        return embeddings, db, retriever  # returns all ready-to-use parts
    
    except Exception as e:
        print(f"💥 Database loading error: {e}")  # if something fails
        import traceback
        traceback.print_exc()  # shows detailed error info
        return None, None, None  # returns nothing if there’s a problem


def load_handbook():
    """Reads the student handbook from the PDF file"""
    
    # Show where the app is looking for the handbook
    print(f"📂 Looking for PDF at: {HANDBOOK_FILE}")
    print(f"📂 Current directory: {os.getcwd()}")
    print(f"📂 PDF exists? {os.path.exists(HANDBOOK_FILE)}")
    
    # If the file doesn't exist, stop here
    if not os.path.exists(HANDBOOK_FILE):
        print(f"❌ PDF file not found!")
        return ""
    
    try:
        # Try opening the PDF file
        print("📖 Attempting to open PDF...")
        from PyPDF2 import PdfReader  # library for reading PDF text
        
        reader = PdfReader(HANDBOOK_FILE)
        print(f"✅ PDF opened! Pages: {len(reader.pages)}")
        
        # Go through each page and collect the text
        text = ""
        for i, page in enumerate(reader.pages):
            print(f"📄 Extracting page {i+1}/{len(reader.pages)}...")
            page_text = page.extract_text()
            text += page_text + "\n"
        
        # Return all the text found
        print(f"✅ Extraction complete! Total characters: {len(text)}")
        return text.strip()
        
    except Exception as e:
        # If something goes wrong, show what happened
        print(f"❌ Error reading handbook PDF: {e}")
        import traceback
        traceback.print_exc()
        return ""
    

# Runs automatically when the server starts
@app.on_event("startup")
async def startup_event():
    """Loads the handbook and database when the app starts"""
    global HANDBOOK_TEXT, embeddings, db, retriever  # allows the function to change these global variables

    print("🚀 Starting server...")  # simple startup message

    loop = asyncio.get_event_loop()  # gets the event loop used for async tasks

    try:
        # Load the handbook file in a separate thread
        HANDBOOK_TEXT = await loop.run_in_executor(executor, lambda: load_handbook())
        print(f"✅ Handbook loaded: {len(HANDBOOK_TEXT)} characters")  # show success message
    except Exception as e:
        HANDBOOK_TEXT = ""  # set empty if failed
        print(f"❌ Error loading handbook: {e}")  # show what went wrong

    try:
        # Load embeddings and database (done in background too)
        embeddings, db, retriever = await loop.run_in_executor(
            executor, lambda: load_embeddings_and_db()
        )
        
        # Check if everything was loaded successfully
        if all([embeddings, db, retriever]):
            print("🎉 Server ready!")  # everything loaded fine
        else:
            print("❌ Server initialization failed")  # one or more parts missing
    except Exception as e:
        # If something breaks, clear the variables and show error
        embeddings, db, retriever = None, None, None
        print(f"❌ Error initializing embeddings/db: {e}")


# Support HEAD requests for UptimeRobot monitoring
@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return {"status": "ChatBook AI is running"}


# This route shows a public report of chatbot performance
@app.get("/report")
def get_report():
    """Shows how many questions were answered or not, and lists frequent ones"""
    try:
        # Try to open the log file where all queries are saved
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))  # read the file as a list of dictionaries
    except FileNotFoundError:
        # If the log file doesn't exist yet, return empty results
        return {
            "total_queries": 0,
            "answered_queries": 0,
            "not_answered_queries": 0,
            "most_frequent_questions": []
        }

    # If the log exists but is empty
    if not reader:
        return {
            "total_queries": 0,
            "answered_queries": 0,
            "not_answered_queries": 0,
            "most_frequent_questions": []
        }

    total = len(reader)  # total number of questions in the log
    
    # Count how many were answered
    answered = sum(
        1 for r in reader 
        if (r.get("answered") or "").strip().lower() in ["true", "1", "yes"]
    )

    not_answered = total - answered  # remaining ones were not answered

    # Count which questions appeared most often
    query_counter = Counter(
        (r.get("query_text") or "").strip().lower()
        for r in reader if r.get("query_text")
    )
    top_faqs = query_counter.most_common(10)  # get top 10 repeated questions

    # Return everything in an easy-to-read format
    return {
        "total_queries": total,
        "answered_queries": answered,
        "not_answered_queries": not_answered,
        "accuracy_rate": (answered / total * 100) if total > 0 else 0,
        "most_frequent_questions": [
            {"question": q, "count": c} for q, c in top_faqs
        ]
    }


# This function runs the main chat pipeline and tracks it in LangSmith
@traceable(name="chat_pipeline")
async def run_chat_pipeline(clean_query: str, clean_context: str, clean_prompt: str, executor, model):
    """Run the LLM model with LangSmith tracing enabled"""

    # Run Gemini model asynchronously in a background thread
    response = await asyncio.get_event_loop().run_in_executor(
        executor, model.generate_content, clean_prompt
    )

    run_id = None
    try:
        # Wait briefly to allow LangSmith to record the trace
        await asyncio.sleep(0.1)

        # Get the latest recorded LangSmith run for this project
        recent_runs = list(client.list_runs(
            project_name=os.getenv("LANGCHAIN_PROJECT", "student-handbook-backend"),
            limit=1,              # Only get one (the latest)
            execution_order=1     # Get most recent run
        ))

        # If there’s a trace (record) available, store its ID
        if recent_runs:
            run_id = str(recent_runs[0].id)
            print(f"✅ Found recent trace_id: {run_id}")
    except Exception as e:
        print(f"⚠️ Could not query recent runs: {e}")

    # Return both the model’s text output and its LangSmith trace ID
    return response.text if hasattr(response, "text") else str(response), run_id


# This function handles the retrieval process with LangSmith tracing
@traceable(name="retrieval_step")
async def traced_retrieval(query, retriever, handbook_text):
    """Retrieve handbook chunks for LangSmith tracing"""
    
    # Use custom retrieval function to get relevant text parts
    docs = await smart_retrieval(query, retriever, handbook_text)

    # Format retrieved results as a list of chunks with metadata
    results = []
    for chunk in docs:
        results.append({
            "content": chunk,   # the actual text
            "metadata": {}      # no metadata attached
        })
    return results


# This function builds the context before sending it to the LLM
@traceable(name="context_builder")
def traced_context(docs, query, session_id):
    """Build context with LangSmith tracing"""
    
    # Extract just the text from retrieved docs
    text_chunks = [doc["content"] for doc in docs]
    
    # Combine memory + retrieved text + new query into a final context
    return build_context_with_memory(text_chunks, query, session_id)


@app.post("/chat")
async def chat(request: Request): # request = user query
    """Main chat endpoint that handles student questions"""
    global retriever, HANDBOOK_TEXT, model # allows this function to access global variables
    start_time = time.time() # Record start time to measure how long the chatbot takes to respond
    
    try:
        # Check if the AI model is set up
        if not model:
            return {"answer": "Gemini API is not configured."}
        
        # Get the student’s message and session info
        data = await request.json()
        query = data.get("query", "").strip()
        session_id = data.get("session_id", "default_session")
        
        # Return a message if no question was provided
        if not query:
            return {"answer": "Please ask a question."}
        
        # Make sure a chat session exists (for memory)
        memory.get_or_create_session(session_id)
        print(f"🔍 Processing query: {query} [Session: {session_id}]")
        
        # Timeout protection
        if time.time() - start_time > 18:
            return {"answer": "Please check your internet connection and try again."}
        
        # Step 1: Find related handbook sections
        docs = await traced_retrieval(query, retriever, HANDBOOK_TEXT)
        chunks_found = len(docs)
        
        # Timeout check again
        if time.time() - start_time > 18:
            return {"answer": "Please check your internet connection and try again."}
        
        # If no related text found in handbook
        if not docs:
            no_answer_msg = "I couldn't find relevant information in the handbook for your question."
            memory.add_conversation(session_id, query, no_answer_msg) # Save this failed search into the chatbot’s memory
            await asyncio.get_event_loop().run_in_executor(
                executor, log_query, query, no_answer_msg, False, 0
            )
            return {"answer": no_answer_msg} # Send the “no information found” message back to the user
        
        # Step 2: Build the context for the model
        context = traced_context(docs, query, session_id)
        clean_context = sanitize_text(context)
        clean_query = sanitize_text(query)
        
        # Step 3: Create the model’s prompt
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
            clean_prompt = clean_prompt[:30000] + "..." # gets cut down to 30,000 characters
        
        # Step 4: Run the model and track it with LangSmith
        response_text, run_id = await run_chat_pipeline(
            clean_query, clean_context, clean_prompt, executor, model
        )

        # Store trace ID for this chat session
        if run_id:
            last_run_ids[session_id] = run_id
            print(f"🔗 Stored run_id {run_id} for session {session_id}")
        
        # Step 5: Clean and finalize the model’s answer
        raw_answer = response_text
        answer = sanitize_text(raw_answer)
        if not answer:
            answer = "I couldn't generate a proper response. Please try rephrasing your question."
        print(f"✅ Answer generated successfully ({len(answer)} characters)")
        
        # Save question and answer to memory
        # Take only the first 200 characters of clean_context
        memory.add_conversation(session_id, query, answer, clean_context[:200])
        
        # Step 6: Check if the response was complete or unclear
        answered = True
        unclear_phrases = [
            "does not specify","not specified","doesn't specify","not mentioned",
            "not provided","doesn't mention","no information","not available",
            "not found","not included","doesn't include","not detailed",
            "doesn't contain","not contain","no details","not clear",
            "doesn't state","not stated","not outlined","not listed",
            "no specific","not specifically","doesn't provide specific",
            "is not available","are not available","not accessible",
            "couldn't find","error","technical difficulties","try again"
        ]
        if not answer.strip() or any(p in answer.lower() for p in unclear_phrases):
            answered = False
        
        # Step 7: Log everything (query, answer, etc.)
        await asyncio.get_event_loop().run_in_executor(
            executor, log_query, query, answer, answered, chunks_found
        )
        
        # Return the final response
        return {"answer": answer, "session_id": session_id}
    
    except Exception as e:
        # Handles any unexpected system error
        print(f"❌ Unexpected error: {repr(e)}")
        error_msg = "Please check your internet connection and try again." if time.time() - start_time > 20 \
            else "I encountered an unexpected error. Please try again or contact support."
        
        # Try saving this failed attempt in the log
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
    """Handles feedback from users"""
    try:
        # make sure the “data” folder exists
        os.makedirs("data", exist_ok=True)
        
        # check if feedback.csv already exists
        file_exists = os.path.isfile(FEEDBACK_FILE)
        
        # open the CSV file and prepare to add feedback
        with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["timestamp", "feedback_text", "rating", "user_type", "session_id"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # write column names if file is new
            if not file_exists:
                writer.writeheader()

            # add new feedback entry
            writer.writerow({
                "timestamp": datetime.now().isoformat(),             # current date and time
                "feedback_text": sanitize_text(feedback.feedback_text), # cleaned feedback text
                "rating": feedback.rating,                           # user’s rating (1–5)
                "user_type": feedback.user_type,                     # who gave it (student, admin, etc.)
                "session_id": feedback.session_id                    # which chat session it came from
            })
        
        # also try sending feedback to LangSmith (for monitoring)
        try:
            feedback_data = {
                "feedback_text": sanitize_text(feedback.feedback_text), # cleaned text again
                "rating": feedback.rating,
                "normalized_score": feedback.rating / 5.0,              # turns score into 0–1 scale
                "user_type": feedback.user_type,
                "session_id": feedback.session_id,
                "timestamp": datetime.now().isoformat()
            }
            
            # simple helper to record feedback activity
            from langsmith import traceable # a tool for observing, debugging, and improving AI apps
            @traceable(name="user_feedback_submission") # When this function runs, record its activity in LangSmith under this name
            def log_feedback_event(data):
                return {"status": "feedback_logged", "data": data}
            
            log_feedback_event(feedback_data)  # send feedback data to LangSmith
            print(f"✅ Feedback logged to LangSmith: {feedback.rating}/5 stars")
            
        except Exception as ls_err:
            # if feedback can’t be sent online, just show a warning
            print(f"⚠️ Could not send feedback to LangSmith: {ls_err}")
        
        print(f"✅ Feedback saved locally: {feedback.rating}/5 stars")
        return {"message": "Feedback submitted successfully", "status": "success"}
    
    except Exception as e:
        # show error if saving feedback fails
        print(f"❌ Error saving feedback: {e}")
        return {"message": "Error submitting feedback", "status": "error"}