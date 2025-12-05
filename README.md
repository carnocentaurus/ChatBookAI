# ChatBook AI
An intelligent chatbot system for Guimaras State University (GSU) that provides instant answers to student queries using the university handbook as its knowledge base.

## Overview
ChatBook AI is a conversational assistant built specifically for GSU students, faculty, and staff. It uses advanced AI (Google Gemini) combined with vector search to provide accurate, context-aware responses to questions about university policies, procedures, admissions, programs, and more.

### Key Features
- **AI-Powered Responses** - Uses Google Gemini 2.5 Flash for natural language understanding
- **Handbook Integration** - Automatically extracts information from PDF handbook
- **Smart Search** - Vector database with semantic search and fuzzy matching
- **Conversation Memory** - Maintains context across multiple questions
- **Admin Dashboard** - Comprehensive analytics and content management
- **Flutter Mobile App** - Native mobile interface for iOS and Android
- **Feedback System** - Collects user ratings and feedback
- **Custom Information** - Admin-controlled knowledge base additions
- **Typo Tolerance** - Handles common misspellings and variations
- **LangSmith Integration** - Performance monitoring and debugging

## Dependencies

### Core Backend
- **FastAPI 0.115.0** - Modern web framework
- **Uvicorn 0.30.6** - ASGI server
- **Python-dotenv 1.0.1** - Environment variable management
- **Pydantic 2.0+** - Data validation

### LangChain Ecosystem
- **LangChain 0.2.14** - LLM orchestration
- **LangChain-Community 0.2.12** - Community integrations
- **LangChain-Chroma 0.1.4** - Vector database integration
- **LangChain-HuggingFace 0.0.3** - Embedding models
- **LangSmith 0.1.0+** - Performance monitoring

### AI & Machine Learning
- **Google Generative AI 0.7.2** - Gemini API
- **Sentence-Transformers 3.0.1** - Text embeddings
- **Transformers 4.44.2** - HuggingFace models
- **Tokenizers** - Fast tokenization

### Vector Database
- **ChromaDB 0.5.6+** - Vector storage and retrieval

### Text Processing
- **FuzzyWuzzy 0.18.0** - Fuzzy string matching
- **Python-Levenshtein 0.27.3** - Fast string distance
- **RapidFuzz 3.14.3** - High-performance fuzzy matching

### Utilities
- **PyPDF2 3.0.1** - PDF parsing
- **Python-multipart** - File upload handling
- **Requests** - HTTP library
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **tqdm** - Progress bars

- **Python**: 3.10.11 (required for LangChain compatibility)
- **Flutter**: Latest stable version (for mobile app)
- **Google Gemini API Key**: Get from [Google AI Studio](https://makersuite.google.com/app/apikey)
- **LangSmith Account**: For monitoring (optional but recommended)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <https://github.com/carnocentaurus/ChatBookAI>
cd CapstoneSystemV6
```

### 2. Backend Setup

#### Create Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

#### Install Dependencies

```bash
pip install -r requirements.txt
```

All required packages including fuzzy matching libraries are included in `requirements.txt`.

#### Create Environment File

Create a `.env` file in the root directory:

```env
# Google Gemini API
GENAI_API_KEY=your_gemini_api_key_here

# LangSmith (Optional - for monitoring)
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=chatbook-ai

# Admin Credentials
ADMIN_USERNAME=admin
ADMIN_PASSWORD=change_this_secure_password
```

**CRITICAL SECURITY NOTES:**
- **NEVER commit the `.env` file to version control**
- **Change ALL default values before deployment**
- **Use strong, unique passwords**
- **Keep API keys confidential**
- The `.env` file is already in `.gitignore` to prevent accidental commits

### 3. Prepare Data Directory

```bash
mkdir -p data
```
Place your university handbook PDF in the `data` folder:
```
data/handbook.pdf
```

### 4. Frontend Setup (Flutter)

```bash
cd FlutterApp  # or your Flutter app directory
flutter pub get
```

## Running the Application

### Start the Backend Server

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

The server will:
1. Load the handbook PDF
2. Create/load the vector database (first run takes longer)
3. Start the API server

**Access Points:**
- API Server: `http://localhost:8000`
- Admin Dashboard: `http://localhost:8000/admin`
- Health Check: `http://localhost:8000/health`
- API Report: `http://localhost:8000/report`

### Start the Flutter App

```bash
cd FlutterApp
flutter run
```

Or build for production:

```bash
# Android
flutter build apk

# iOS
flutter build ios
```

## Features Overview

### For Students/Users
1. **Ask Questions**
   - Natural language queries about GSU
   - Supports typos and variations
   - Maintains conversation context
2. **Get Instant Answers**
   - Powered by AI and handbook data
   - Accurate, sourced information
   - Follow-up question support
3. **Provide Feedback**
   - Rate responses (1-5 stars)
   - Submit written feedback
   - Help improve the system

### For Administrators
Access the admin panel at: `http://localhost:8000/admin`
**Default Credentials:**
- Username: `admin`
- Password: (set in `.env` file)

#### Admin Dashboard Features:

1. **System Overview**
   - Total queries processed
   - Success rate statistics
   - Active sessions
   - Recent queries log
2. **Manage Information**
   - Add custom Q&A entries
   - Update handbook PDF
   - Delete outdated information
   - Export system data
3. **FAQ Analysis**
   - Most popular questions
   - Unanswered queries tracking
   - Query grouping (similar questions)
   - Success rate per question
4. **Manage Queries**
   - View unresolved queries
   - Multi-select bulk actions
   - Mark queries as resolved
   - Add information for specific questions
5. **Feedback Review**
   - View all user feedback
   - Rating statistics
   - Feedback by user type
   - Average ratings

## Project Structure

```
CapstoneSystemV6/
├── api.py                      # Main FastAPI application
├── admin.py                    # Admin panel routes and HTML
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (create this)
├── data/
│   ├── handbook.pdf           # University handbook
│   ├── query_log.csv          # Query logs
│   ├── feedback.csv           # User feedback
│   ├── custom_info.json       # Custom Q&A data
│   ├── chatbot_memory.db      # SQLite conversation database
│   └── chroma_db/             # Vector database
└── FlutterApp/               # Mobile application
    ├── lib/
    ├── pubspec.yaml
    └── ...
```

## Configuration

### Adjusting AI Parameters

In `api.py`, modify the `generation_config`:

```python
generation_config = genai.types.GenerationConfig(
    temperature=0.1,      # Lower = more focused (0.0-1.0)
    top_p=0.8,           # Nucleus sampling (0.0-1.0)
    top_k=40,            # Top-k sampling
    max_output_tokens=2048  # Maximum response length
)
```

### Vector Database Settings

In `load_embeddings_and_db()`:

```python
chunk_size=800,          # Text chunk size
chunk_overlap=150,       # Overlap between chunks
batch_size=50,           # Processing batch size
```

### Retrieval Settings

```python
search_kwargs={
    "k": 10,             # Number of chunks to retrieve
    "fetch_k": 25,       # Candidates to consider
    "lambda_mult": 0.5   # MMR diversity (0=diverse, 1=relevant)
}
```

## Maintenance Tasks

### Update the Handbook
1. Login to admin panel: `http://localhost:8000/admin`
2. Navigate to **Manage Information**
3. Click **Update Handbook PDF**
4. Upload new PDF file
5. System will automatically reindex

### Add Custom Information
1. Go to **Manage Information** → **Add New Information**
2. Enter topic (e.g., "Scholarship Deadline")
3. Enter information
4. Click **Add Information**

### Monitor Performance
1. Check **Dashboard** for success rates
2. Review **FAQ Analysis** for common questions
3. Check **Manage Queries** for unanswered questions
4. Review **Feedback** for user satisfaction

### View LangSmith Traces
1. Login to [LangSmith](https://smith.langchain.com/)
2. Select your project
3. View traces, latency, and errors

## API Endpoints

### Public Endpoints

```
POST /chat
  - Main chat endpoint
  - Body: { "query": "question", "session_id": "user_session" }
  - Returns: { "answer": "response", "session_id": "session" }

POST /feedback
  - Submit user feedback
  - Body: { "feedback_text": "...", "rating": 5, "user_type": "student" }

GET /health
  - System health check
  - Returns system status

GET /report
  - Public statistics
  - Returns query counts and FAQ
```

### Admin Endpoints

All admin endpoints require HTTP Basic Auth:

```
GET  /admin                          # Dashboard
GET  /admin/custom-info              # Manage information
POST /admin/custom-info/add          # Add new info
GET  /admin/custom-info/delete/{id}  # Delete info
POST /admin/upload-handbook          # Upload PDF
GET  /admin/faq                      # FAQ analysis
GET  /admin/manage-queries           # Manage queries
POST /admin/mark-resolved            # Mark query resolved
POST /admin/bulk-mark-resolved       # Bulk mark resolved
GET  /admin/feedback                 # View feedback
GET  /admin/export-data              # Export system data
```

## Troubleshooting

### Issue: Server won't start

**Solution:**
1. Check Python version: `python --version` (should be 3.10.11)
2. Verify all dependencies: `pip install -r requirements.txt`
3. Check `.env` file exists with API keys

### Issue: "Handbook not found"

**Solution:**
1. Ensure `data/handbook.pdf` exists
2. Check file permissions
3. Verify PDF is not corrupted

### Issue: "No results found"

**Solution:**
1. Check if vector database is created (`data/chroma_db/` exists)
2. Delete `chroma_db` folder and restart (will reindex)
3. Add custom information via admin panel

### Issue: Slow responses

**Solution:**
1. Reduce chunk retrieval: Lower `k` value
2. Check internet connection (Gemini API)
3. Monitor LangSmith for bottlenecks

### Issue: Admin login not working

**Solution:**
1. Check `.env` file has correct credentials
2. Clear browser cache
3. Verify `ADMIN_USERNAME` and `ADMIN_PASSWORD` are set

## Security Considerations

**CRITICAL**: Follow these security best practices:

1. **Never Commit Secrets**
   - The `.env` file contains sensitive credentials
   - Ensure `.env` is in `.gitignore`
   - Never share your `.env` file publicly
   - Use environment-specific `.env` files
2. **Change Default Credentials**
   - Update `ADMIN_PASSWORD` from default value
   - Use strong passwords (12+ characters, mixed case, numbers, symbols)
   - Never use `admin/admin` or common passwords
3. **Secure API Keys**
   - Keep `GENAI_API_KEY` confidential
   - Keep `LANGCHAIN_API_KEY` confidential
   - Rotate keys periodically
   - Use different keys for development/production
4. **Production Deployment**
   - Use HTTPS/SSL certificates (not HTTP)
   - Consider adding rate limiting
   - Implement proper authentication
   - Use firewall rules
   - Keep server software updated
5. **Input Validation**
   - All user inputs are sanitized (see `sanitize_text()`)
   - SQL injection protection built-in
   - File upload validation enabled
6. **Monitoring**
   - Review LangSmith traces for suspicious activity
   - Monitor admin dashboard for unusual patterns
   - Keep logs for security auditing

## Performance Optimization
1. **Vector Database**: Already optimized with MMR search
2. **Caching**: Consider adding Redis for frequently asked questions
3. **Batch Processing**: Handbook chunks processed in batches
4. **Memory Management**: Garbage collection implemented
5. **Async Operations**: FastAPI handles concurrent requests

## Updating the System

### Update Dependencies

```bash
pip install --upgrade -r requirements.txt
```

### Backup Data

```bash
# Backup important data before updates
cp -r data/ data_backup/
```

### Reset Vector Database

```bash
# If having issues, delete and rebuild
rm -rf data/chroma_db/
# Restart server - will rebuild automatically
```

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Google Gemini API](https://ai.google.dev/)
- [LangChain Documentation](https://python.langchain.com/)
- [Flutter Documentation](https://flutter.dev/docs)
- [LangSmith Docs](https://docs.smith.langchain.com/)

## Development History

### Initial Setup (August 2024)
1. Set up Python 3.10.11 environment (LangChain compatibility)
2. Configured environment variables and paths
3. Explored RAG LLM architectures and implementations
4. Initial experimentation with local LLMs (Mistral, Phi3:mini, TinyDolphin)

### V1 (August 9, 2025)
- Created `data.md` for basic GSU information
- Built `vector_db.py` for data management
- Implemented `query.py` for LLM queries
- Added `sections.pkl` for section-based data storage

### V2 (August 21, 2025)
- Removed LLM dependency for offline operation
- Implemented pure section-based retrieval (no hallucination)
- Enhanced section storage in `sections.pkl`

### V2.5 (August 27, 2025)
- Replaced `data.md` with `handbook.pdf`
- Switched from section-based to chunk-based processing
- Implemented vector embeddings in ChromaDB
- Stored chunks in `sections.pkl`

### V3 (August 24, 2025)
- Git version control setup
- Android Studio and SDK installation (Android 14)
- Flutter SDK integration and environment configuration
- Created `gsu_chatbot_app` Flutter project
- Desktop development environment setup
- Added HTTP package for API communication
- Built `chat_page.dart` - main chat interface
- Created `report_page.dart` - system reports

### V3.5 (August 30, 2025)
- Integrated official GSU Student Handbook PDF
- Replaced top sections with FAQ system
- Transitioned from section-based to FAQ-based architecture

### V4 (September 7, 2025)
- Enhanced chatbot response quality
- Auto-scroll functionality in chat
- Conversation memory implementation
- User feedback system for corrections
- Custom information input via chat
- Web-based admin panel
- Converted reports page to FAQ page
- UI/UX improvements

### V5 (November 13, 2025)
- Clickable FAQ items (auto-query on click)
- Ranked FAQ system (queries asked 2+ times)
- Manage Queries page (replaced Analytics)
- Resolved date tracking system
- Fully responsive admin panel design
- Feedback system in Flutter app and admin panel
- LangSmith dashboard integration
- Custom app logo, splash screen, and fonts
- APK deployment via mobile hotspot

### V6 (December 5, 2025) - Current
- **Chatbot Response Visibility** - Full responses shown in Recent Queries
- **Stop Response Feature** - Ability to halt ongoing chatbot responses
- **Unanswered Queries Section** - Dedicated section in FAQ Analysis
- **Bulk Query Management** - Multi-select mark as resolved
- **Smart Query Grouping** - Similar queries grouped with fuzzy matching
- **Enhanced Fuzzy Matching** - Triple-layer matching (FuzzyWuzzy + Levenshtein + RapidFuzz)
- **Typo Tolerance** - Handles common misspellings (e.g., "bision" → "vision")
- **Query Preprocessing** - Automatic typo correction and normalization
- **Improved FAQ Analysis** - Side-by-side popular vs unanswered questions
- **Advanced Custom Info Matching** - Multi-strategy relevance scoring

## Support
For issues or questions:
1. Check troubleshooting section above
2. Review LangSmith traces for errors
3. Check admin dashboard for system health
4. Ensure all dependencies match versions in `requirements.txt`

## Version History

### v1.0.0 (Current)
- AI-powered chat with Gemini 2.5 Flash (v0.7.2)
- Admin dashboard with analytics
- FAQ analysis and smart grouping
- Triple fuzzy matching (FuzzyWuzzy, Levenshtein, RapidFuzz)
- Feedback collection system
- Multi-select query management
- Custom information management
- PDF handbook integration (PyPDF2)
- LangSmith monitoring and tracing
- Flutter mobile app
- Vector search with ChromaDB 0.5.6+
- HuggingFace embeddings integration