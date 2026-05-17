# Project Reviewer Document: ChatBook AI

## 1. What is this System?
**ChatBook AI** is an intelligent, Retrieval-Augmented Generation (RAG) chatbot system designed specifically for **Guimaras State University (GSU)**. It serves as a **Prototype**—an early, functional model built to validate technical feasibility, catch design flaws early, and gather immediate feedback from GSU students and faculty before committing to full-scale development.

Its primary purpose is to serve as a digital assistant that provides students, faculty, and staff with instant, accurate answers to queries regarding the university handbook, policies, admissions, and campus life.

### Core Components:
*   **Mobile & Web Frontend:** A cross-platform application built with **Flutter**, providing a user-friendly chat interface, FAQ section, and feedback system.
*   **Backend API:** A **FastAPI** (Python) server that handles natural language processing, data retrieval, and system management.
*   **AI Engine:** Integrates **Google Gemini 2.5 Flash** for natural language understanding and response generation.
*   **Knowledge Base:** A combination of the official **University Handbook (PDF)** and **Admin-defined Custom Information (JSON)**.
*   **Vector Database:** **ChromaDB** is used to store and retrieve document embeddings for semantic search.
*   **Admin Dashboard:** A web-based interface for university administrators to monitor performance, update knowledge, and analyze student needs.

---

## 2. How does it work? (Technical Workflow)

The system operates through a sophisticated pipeline designed to ensure accuracy and minimize "hallucinations" (AI-generated misinformation).

### A. Data Preparation (The "Knowledge")
1.  **Ingestion:** The system reads the `handbook.pdf` using `PyPDF2`.
2.  **Chunking:** The text is broken into small, overlapping segments (800 characters each) using `RecursiveCharacterTextSplitter`.
3.  **Embedding:** Each chunk is converted into a numerical vector using the `sentence-transformers/all-MiniLM-L6-v2` model from HuggingFace.
4.  **Storage:** These vectors are stored in **ChromaDB**.

### B. The Query Pipeline (The "Chat")
1.  **Input Processing:** When a user asks a question, the system applies **Triple-Layer Fuzzy Matching** (FuzzyWuzzy + Levenshtein + RapidFuzz) to handle typos and local terminology (e.g., "bisyon" instead of "vision").
2.  **Retrieval:** The system searches ChromaDB for the most relevant handbook chunks using **Maximal Marginal Relevance (MMR)** to ensure both relevance and diversity in information.
3.  **Context Construction:** The system combines:
    *   Retrieved Handbook Chunks
    *   Recent Conversation History (Short-term Memory)
    *   Relevant Custom Information (Admin Updates)
4.  **Generation:** This combined "context" is sent to **Gemini 2.5 Flash** with a strict prompt instructing it to *only* answer based on the provided data.
5.  **Output:** The sanitized response is sent back to the Flutter app.

### C. Feedback and Iteration
1.  **Feedback:** Users rate responses (1-5 stars) and provide comments, which are logged to `feedback.csv`.
2.  **Monitoring:** The **LangSmith** integration allows developers to trace every AI thought process, identifying where the model might be struggling.
3.  **Admin Intervention:** Admins use the dashboard to mark unanswered queries as "Resolved" and add "Custom Information" to fill gaps in the handbook.

---

## 3. Why is it designed this way? (Design Rationale)

*   **RAG over Fine-Tuning:** Fine-tuning an AI model is expensive and static. By using RAG (Retrieval-Augmented Generation), the system can be updated instantly by simply uploading a new PDF or adding a Q&A entry in the admin panel.
*   **Gemini 2.5 Flash:** Chosen for its extremely low latency and massive 1-million+ token context window, which is ideal for processing long university documents.
*   **Local Vector DB (ChromaDB):** Ensures the university's data stays local and reduces reliance on expensive third-party database subscriptions.
*   **Triple-Layer Fuzzy Matching:** Specifically implemented to handle the diverse linguistic patterns and common spelling variations found in a local university environment.
*   **FastAPI & Asynchronous Design:** Allows the system to handle multiple student queries simultaneously without slowing down, even when performing complex searches.

---

## 4. Core Technical Foundations

To understand the architecture of ChatBook AI, three key concepts are foundational:

*   **Retrieval-Augmented Generation (RAG):** This is the **Blueprint/Architecture** of our system. It is the theoretical framework that dictates how the AI should behave: "Before answering, first fetch relevant facts from a local database." RAG ensures the AI stays grounded in GSU-specific facts rather than general knowledge.
*   **LangChain:** This is the **Development Toolkit**. If RAG is the blueprint for a hybrid car, LangChain is the factory machinery used to build it. It provides the actual Python code modules that "chain" together document splitting, vector database connection, and Gemini API calls into a single pipeline.
*   **LangSmith:** This is the **DevOps & Observability Platform**. AI systems are traditionally "black boxes" where it's hard to see why a bad answer occurred. LangSmith acts as an **X-ray machine**, allowing us to trace exactly where a query might have failed—whether it was a retrieval error, a prompt issue, or a model hallucination—ensuring system reliability through real-time monitoring and debugging.

---

## 5. Search & Retrieval Logic

The system uses two distinct methods to find information, ensuring that both specific rules and broad concepts are captured:

### A. Triple-Layer Fuzzy Matching (Section-Based)
This logic is designed for **Precision**. It looks for specific topics or sections defined in the handbook or by administrators.
1.  **Normalization:** Fixes common typos (e.g., "bisyon" → "vision").
2.  **Keyword Extraction:** Identifies the "meat" of the user's question.
3.  **Fuzzy Scoring:** Uses algorithms (Levenshtein/RapidFuzz) to compare the query against known section titles. If a student asks "What is the mission?", the system recognizes the "Mission" section even if phrased as a question, giving it a high priority match.

### B. Semantic Search (Vector-Based)
This logic is designed for **Context**. It uses the **ChromaDB** vector database to find information based on *meaning*, not just keywords.
1.  **Embeddings:** Converts the user's query into a mathematical "vector" (a list of numbers) that represents its semantic meaning.
2.  **Similarity Search:** Compares that vector against the handbook's stored vectors.
3.  **The Benefit:** If a student asks "How do I get in?", Semantic Search understands that "get in" is mathematically similar to "Admission" or "Enrollment," even if those exact words weren't in the student's query.

---

## 6. Development Process: Iterative vs. Agile

ChatBook AI was developed using an **Iterative Model** rather than a standard Agile approach.

*   **Iterative Model:** We defined the core requirements—mapping the GSU Handbook—from the start. We then built the system through repeated refinement cycles: making a functional rough version first, followed by technical optimizations (like fuzzy matching), and finally a polished user interface.
*   **Why Iterative for GSU?** Agile is best for commercial apps with constantly changing scopes. However, ChatBook AI is built upon an official university handbook where the rules are strictly set and pre-approved. The Iterative model is perfect here because the end goal is fixed; we simply use each cycle to get closer to 100% technical accuracy and stability.

---

## 6. Impact Analysis: What if a component was removed?

| Missing/Removed Component | Impact on the System |
| :--- | :--- |
| **Google Gemini API** | **System Failure:** The chatbot would be unable to generate natural language responses. It would revert to a simple "keyword search" tool with no conversational ability. |
| **ChromaDB (Vector DB)** | **Loss of Context:** The AI would have no access to the handbook. It would only be able to answer based on its general training data, which would lead to incorrect information about specific GSU policies. |
| **Handbook PDF** | **Knowledge Void:** The primary source of truth is gone. The system becomes a shell with no university-specific data to provide. |
| **Admin Panel** | **Static System:** The knowledge base could never be updated. Errors would persist, and new university policies could not be added without redeploying the entire codebase. |
| **Fuzzy Matching Logic** | **Poor User Experience:** Students making simple typos (e.g., "scholaship") would receive "Information not found" errors, making the system feel "dumb" and frustrating to use. |
| **LangSmith Tracing** | **Blind Optimization:** Developers wouldn't know *why* a bot gave a wrong answer, making it nearly impossible to debug complex AI failures or improve accuracy over time. |
| **Feedback System** | **No Quality Control:** University staff would have no metric to judge if students are actually finding the bot helpful or if it is providing confusing answers.