# admin.py

from fastapi import HTTPException, Depends, Form, Request, UploadFile, File  # lets the app handle forms, files, and web requests
from fastapi.responses import HTMLResponse, RedirectResponse  # shows web pages or moves the user to another page
from fastapi.security import HTTPBasic, HTTPBasicCredentials  # checks username and password for admin login
import secrets  # keeps passwords and keys safe
import os  # works with folders and files
import csv  # reads and saves feedback data
import sqlite3  # connects to the local database
import shutil  # copies or replaces files
from datetime import datetime  # records the date and time of actions
from collections import Counter  # counts how many times something appears
from urllib.parse import quote, unquote  # cleans or restores text used in web links
from typing import List

from dotenv import load_dotenv
load_dotenv()

# Security setup
security = HTTPBasic()  # sets up basic login checking
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")  # gets the admin name or uses "admin" if none is set
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "gsu2025")  # gets the admin password or uses "gsu2025" if none is set


# Function to check if the admin login details are correct
def verify_admin(credentials: HTTPBasicCredentials = Depends(security)):
    is_correct_username = secrets.compare_digest(credentials.username, ADMIN_USERNAME)  # checks if the name matches
    is_correct_password = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)  # checks if the password matches
    
    if not (is_correct_username and is_correct_password):  # if either is wrong
        raise HTTPException(  # stop the request and show an error message
            status_code=401,
            detail="Invalid admin credentials",
            headers={"WWW-Authenticate": "Basic"}, # tells the browser: “I’m using HTTP Basic Authentication
        )
    
    return credentials.username  # if correct, allow access and return the name


# Attach all admin-related routes to the FastAPI app
def setup_admin_routes(app, memory, LOG_FILE, MEMORY_DB):
    @app.get("/admin", response_class=HTMLResponse)   # When someone visits /admin, show the admin dashboard as a web page
    async def admin_dashboard(credentials: HTTPBasicCredentials = Depends(verify_admin)):  # Only allow access if the admin logs in

        total_custom_info = len(memory.custom_info)   # Count how many custom items are saved in memory

        # Try to read the activity log for statistics
        try:
            # with = automatic close
            with open(LOG_FILE, "r", encoding="utf-8") as f:   # Open the log file
                reader = list(csv.DictReader(f)) # Read its content as a list of rows

            total_queries = len(reader) # Count all recorded user questions
        
            # Count how many were answered or not
            answered_count = sum(1 for r in reader if (r.get("answered") or "").strip().lower() in ["true", "1", "yes"])
            not_answered_count = total_queries - answered_count  # The rest are unanswered
            accuracy_rate = (answered_count / total_queries * 100) if total_queries > 0 else 0  # Get percentage of answered ones
        
            recent_queries = list(reversed(reader[-10:])) if reader else []  # Show the 10 most recent entries
        except FileNotFoundError: # If the log file doesn’t exist
            total_queries = 0 # Use default values
            answered_count = 0
            not_answered_count = 0
            accuracy_rate = 0
            recent_queries = []

        # Open the database to count stored records
        conn = sqlite3.connect(MEMORY_DB) # Connect to the database
        cursor = conn.cursor() # Create a cursor for queries
        cursor.execute("SELECT COUNT(*) FROM conversations") # Count all conversations
        total_conversations = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM sessions") # Count all sessions
        total_sessions = cursor.fetchone()[0]
        conn.close()  # Close the database connection

        # Show all the gathered info in a web dashboard
        return HTMLResponse(content=get_updated_dashboard_html(
            total_queries, answered_count, not_answered_count, accuracy_rate,
            total_conversations, total_sessions, total_custom_info, recent_queries
        ))


    @app.get("/admin/custom-info", response_class=HTMLResponse)  # When admin visits this page, show all custom info
    async def admin_custom_info(credentials: HTTPBasicCredentials = Depends(verify_admin)):  # Make sure admin is logged in
        # Admin page for viewing and managing saved info
        return HTMLResponse(content=get_custom_info_html(memory.custom_info))  # Show the page with all current info

    @app.get("/admin/custom-info/add", response_class=HTMLResponse)  # Page for adding new custom info
    async def admin_add_info_form(
        credentials: HTTPBasicCredentials = Depends(verify_admin),  # Check admin login
        prefill_topic: str = None # Optional topic to fill in automatically
    ):
        # Show the form to add new info
        prefilled_topic = unquote(prefill_topic) if prefill_topic else ""  # If there’s a topic, show it in the form
        return HTMLResponse(content=get_add_custom_info_form_html(prefilled_topic))  # Display the form page

    @app.post("/admin/custom-info/add")  # When admin submits the form, this handles it
    async def admin_add_info(
        credentials: HTTPBasicCredentials = Depends(verify_admin),  # Check admin login
        topic: str = Form(...), # Get topic text from the form
        information: str = Form(...), # Get information text from the form
    ):
        # Add the new custom info to memory
        memory.add_custom_info(topic, information) # Save the topic and info
        # 303 = redirect client
        return RedirectResponse(url="/admin/custom-info", status_code=303)  # Go back to the list after adding
    
    @app.get("/admin/custom-info/delete/{info_id}")
    async def admin_delete_info(
        info_id: str,
        credentials: HTTPBasicCredentials = Depends(verify_admin)
    ):
        # Delete a custom info entry if it exists
        if info_id in memory.custom_info:
            del memory.custom_info[info_id]
            memory.save_custom_info()
        return RedirectResponse(url="/admin/custom-info", status_code=303)
    
    
    @app.get("/admin/upload-handbook", response_class=HTMLResponse)  # When admin visits the upload page
    async def admin_upload_handbook_form(credentials: HTTPBasicCredentials = Depends(verify_admin)):  # Check admin login
        handbook_path = "data/handbook.pdf"  # Location where the handbook is saved
        handbook_info = ""  # Info to show about the current handbook
    
        if os.path.exists(handbook_path):  # If a handbook file already exists
            file_size = os.path.getsize(handbook_path)  # Get how big the file is
            file_size_mb = file_size / (1024 * 1024)  # Convert size to MB
            modified_time = datetime.fromtimestamp(os.path.getmtime(handbook_path))  # Get last updated time
            handbook_info = f"Current handbook: {file_size_mb:.2f} MB, last updated {modified_time.strftime('%Y-%m-%d %H:%M:%S')}"  # Show file info
        else:
            handbook_info = "No handbook currently uploaded"  # Message if no file found

        return HTMLResponse(content=get_upload_handbook_html(handbook_info))  # Show upload page with handbook info

    @app.post("/admin/upload-handbook")  # When admin uploads a new handbook
    async def admin_upload_handbook(
        credentials: HTTPBasicCredentials = Depends(verify_admin),  # Check admin login
        file: UploadFile = File(...)  # Get the uploaded file
    ):
        try:
            if not file.filename.lower().endswith('.pdf'):  # Check if the file is a PDF
                # 404 = bad/invalid request
                raise HTTPException(status_code=400, detail="Only PDF files are allowed")

            os.makedirs("data", exist_ok=True)  # Make sure the folder exists

            handbook_path = "data/handbook.pdf"  # Main handbook file
            temp_path = "data/handbook_temp.pdf"  # Temporary file while uploading
            backup_path = "data/handbook_backup.pdf"  # Backup of the old file

            if os.path.exists(handbook_path):  # If there’s already a handbook
                shutil.copy2(handbook_path, backup_path)  # Make a backup before replacing

            with open(temp_path, "wb") as buffer:  # Save uploaded file temporarily
                shutil.copyfileobj(file.file, buffer)

            if os.path.getsize(temp_path) == 0:  # If uploaded file is empty
                os.remove(temp_path)  # Delete it
                raise HTTPException(status_code=400, detail="Uploaded file is empty")

            shutil.move(temp_path, handbook_path)  # Replace the old handbook with the new one

            if os.path.exists(backup_path):  # If backup exists
                os.remove(backup_path)  # Delete it since upload succeeded

            return RedirectResponse(url="/admin/custom-info?upload=success", status_code=303)

        except HTTPException:
            raise  # Rethrow expected errors
        except Exception as e:  # If something goes wrong
            if os.path.exists(backup_path) and not os.path.exists(handbook_path):  # If upload failed
                shutil.move(backup_path, handbook_path)  # Restore old handbook

            if os.path.exists(temp_path):  # Clean up temp file
                os.remove(temp_path)
            raise HTTPException(status_code=500, detail=f"Error uploading handbook: {str(e)}")  # Show error message
    

    @app.get("/admin/faq", response_class=HTMLResponse)
    async def admin_faq(credentials: HTTPBasicCredentials = Depends(verify_admin)):
        """Shows a full list of frequently asked questions from user logs"""

        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                reader = list(csv.DictReader(f))

            # ===== FREQUENTLY ASKED QUESTIONS (asked 2+ times) =====
            query_counter = Counter(
                (r.get("query_text") or "").strip().lower() for r in reader if r.get("query_text")
            )
    
            frequent_questions = [(q, count) for q, count in query_counter.most_common() if count >= 2]

            faq_list = []

            for question, count in frequent_questions:
                question_entries = [r for r in reader if (r.get("query_text") or "").strip().lower() == question]
        
                answered_count = 0
                for entry in question_entries:
                    answered_field = (entry.get("answered") or "").strip().lower()
                    if answered_field in ["true", "1", "yes"]:
                        answered_count += 1

                success_rate = (answered_count / count * 100) if count > 0 else 0

                faq_data = {
                    "question": question,
                    "total_asked": count,
                    "answered_count": answered_count,
                    "success_rate": success_rate
                }
                faq_list.append(faq_data)

            # ===== UNANSWERED QUESTIONS (all questions not answered) =====
            not_answered_counter = Counter()
        
            for record in reader:
                answered_field = (record.get("answered") or "").strip().lower()
                query_text = (record.get("query_text") or "").strip().lower()
            
                # Only count if not answered
                if answered_field not in ["true", "1", "yes"] and query_text:
                    not_answered_counter[query_text] += 1
        
            # Build list with resolved status
            not_answered_list = []
            for question, count in not_answered_counter.most_common():
                # Check if this question has been marked as resolved
                question_records = [r for r in reader if (r.get("query_text") or "").strip().lower() == question]
                is_resolved = any((r.get("resolved_date") or "").strip() for r in question_records)
            
                not_answered_list.append((question, count, is_resolved))

        except FileNotFoundError:
            faq_list = []
            not_answered_list = []
        except Exception as e:
            print(f"Error in admin_faq: {e}")
            faq_list = []
            not_answered_list = []

        return HTMLResponse(content=get_full_faq_html(faq_list, not_answered_list))

    
    @app.get("/admin/manage-queries", response_class=HTMLResponse)  # Page for viewing unanswered questions
    async def admin_manage_queries(credentials: HTTPBasicCredentials = Depends(verify_admin)):  # Only admin can open this page
        """Shows questions that have not been answered or resolved yet"""

        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:  # Open the log file that stores all user questions
                reader = list(csv.DictReader(f))  # Read all entries from the log

            not_answered_queries = []  # This will store questions that still need answers

            for record in reader:  # Go through each question record
                answered_field = (record.get("answered") or "").strip().lower()  # Check if marked as answered
                query_text = (record.get("query_text") or "").strip()  # Get the question text
                resolved_date = (record.get("resolved_date") or "").strip()  # Check if marked as resolved
        
                # Add to list only if it’s not answered, not resolved, and not empty
                if (answered_field not in ["true", "1", "yes"] and 
                    query_text and 
                    not resolved_date):
                    not_answered_queries.append(record)

            # Count how many times each unanswered question appears
            not_answered_counter = Counter()
            for query in not_answered_queries:
                query_text = (query.get("query_text") or "").strip().lower()
                if query_text:
                    # Adds one each time the same question appears
                    not_answered_counter[query_text] += 1

            # Sort questions by how often they appear
            all_needing_attention = not_answered_counter.most_common()

        except FileNotFoundError:
            all_needing_attention = []  # If no log file exists yet, show nothing
        except Exception as e:
            print(f"Error in manage queries: {e}")  # Print any problem found
            all_needing_attention = []  # Show an empty list if something goes wrong

        # Show the Manage Queries page with all questions that still need answers
        return HTMLResponse(content=get_manage_queries_with_resolved_html(all_needing_attention))
    
    
    @app.get("/admin/feedback", response_class=HTMLResponse)  # Page to view user feedback
    async def admin_feedback(credentials: HTTPBasicCredentials = Depends(verify_admin)):  # Only admin can access
        """Admin page to view user feedback"""

        try:
            with open("data/feedback.csv", "r", encoding="utf-8") as f:  # Open the feedback file
                reader = list(csv.DictReader(f))  # Read all feedback entries

            # Sort feedback from newest to oldest
            feedback_list = sorted(reader, key=lambda x: x.get('timestamp', ''), reverse=True)

            # Count how many feedback entries exist
            total_feedback = len(feedback_list)

            if total_feedback > 0:  # If there is at least one feedback entry
                # Collect all rating numbers from feedback
                ratings = [int(f.get('rating', 0)) for f in feedback_list if f.get('rating', '').isdigit()]
            
                # Compute the average rating (total divided by number of ratings)
                avg_rating = sum(ratings) / len(ratings) if ratings else 0
            
                # Count how many people gave each rating (e.g., how many 5s, 4s, etc.)
                rating_distribution = Counter(ratings)
            else:
                # If no feedback yet, set everything to zero
                avg_rating = 0
                rating_distribution = Counter()

        except FileNotFoundError:
            # If the feedback file doesn’t exist yet, just show an empty list
            feedback_list = []
            total_feedback = 0
            avg_rating = 0
            rating_distribution = Counter()

        # Show the feedback page with all details (list, total, average, rating counts)
        return HTMLResponse(content=get_feedback_html(feedback_list, total_feedback, avg_rating, rating_distribution))

    
    @app.post("/admin/mark-resolved")  # When the admin marks a question as resolved
    async def mark_query_resolved(
        request: Request,
        credentials: HTTPBasicCredentials = Depends(verify_admin),  # Only admin can do this
        question: str = Form(...)  # The question to mark as resolved
    ):
        """Mark all instances of a question as resolved by updating CSV"""
        try:
            # Open the log file where all chatbot questions are saved
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                reader = list(csv.DictReader(f))  # Read all the records

            # Make sure there’s a column for “resolved_date” in csv file
            fieldnames = reader[0].keys() if reader else []
            if 'resolved_date' not in fieldnames:
                # if not create one
                fieldnames = list(fieldnames) + ['resolved_date']

            # Get today’s date to mark the question as resolved
            today = datetime.now().strftime("%Y-%m-%d")
            updated_count = 0  # Counter for how many records were updated

            # Go through every record in the file
            for record in reader:
                # Check if this record matches the question the admin wants resolved
                if record.get("query_text", "").strip().lower() == question.lower():
                    # If not yet resolved, mark it with today’s date
                    if not record.get("resolved_date", "").strip():
                        record["resolved_date"] = today
                        updated_count += 1
                    # Handle older files that don’t have this column yet
                    elif "resolved_date" not in record:
                        record["resolved_date"] = today
                        updated_count += 1

            # Save all updates back into the file
            if reader:
                with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(reader)

            # Print a note in the the terminal showing how many were marked as resolved
            print(f"Marked {updated_count} instances of '{question}' as resolved")

            # After marking, send the admin back to the manage queries page
            return RedirectResponse(url="/admin/manage-queries", status_code=303)

        except Exception as e:
            # If something goes wrong, show an error and go back to manage queries
            print(f"Error marking query as resolved: {e}")
            return RedirectResponse(url="/admin/manage-queries", status_code=303)
        

    @app.post("/admin/bulk-mark-resolved")
    async def bulk_mark_resolved(
        request: Request,
        credentials: HTTPBasicCredentials = Depends(verify_admin),
        selected_queries: List[str] = Form(...)
    ):
        """Mark multiple queries as resolved at once"""
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                reader = list(csv.DictReader(f))

            fieldnames = reader[0].keys() if reader else []
            if 'resolved_date' not in fieldnames:
                fieldnames = list(fieldnames) + ['resolved_date']

            today = datetime.now().strftime("%Y-%m-%d")
            updated_count = 0

            # Mark all selected queries as resolved
            for record in reader:
                query_text = record.get("query_text", "").strip().lower()
            
                # Check if this record matches any of the selected queries
                if query_text in [q.lower() for q in selected_queries]:
                    if not record.get("resolved_date", "").strip():
                        record["resolved_date"] = today
                        updated_count += 1
                    elif "resolved_date" not in record:
                        record["resolved_date"] = today
                        updated_count += 1

            # Save updates back to file
            if reader:
                with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(reader)

            print(f"✅ Bulk resolved {updated_count} query instances across {len(selected_queries)} unique questions")

            return RedirectResponse(url="/admin/manage-queries", status_code=303)

        except Exception as e:
            print(f"❌ Error in bulk resolve: {e}")
            return RedirectResponse(url="/admin/manage-queries", status_code=303)


    @app.get("/admin/export-data")  # When the admin visits this page to export data
    async def admin_export_data(credentials: HTTPBasicCredentials = Depends(verify_admin)):
        # Only the admin can access this function

        # Prepare the data to be exported
        export_data = {
            "custom_info": memory.custom_info,  # Get stored chatbot data
            "export_timestamp": datetime.now().isoformat()  # Add the time when it was exported
        }

        # Return the data and a suggested file name for download
        return {
            "data": export_data,
            "filename": f"gsu_chatbot_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        }


# HTML Templates
def get_base_style():
    """Clean, modern CSS with proper responsive design"""
    return """
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

        /*CSS variables (custom properties).reused throughout the stylesheet*/
        :root {
            --blue: #1976d2;
            --blue-dark: #1565c0;
            --green: #388e3c;
            --green-dark: #2e7d32;
            --orange: #f57c00;
            --orange-dark: #ef6c00;
            --red: #d32f2f;
            --red-dark: #c62828;
            --gray: #6c757d;
            --gray-dark: #5a6268;
            --muted: #666666;
            --light-gray: #f8f9fa;
            --border: #ddd;
        }

        /*ensures padding and borders are calculated neatly*/
        * { box-sizing: border-box; }

        body {
            font-family: 'Poppins', Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
            font-size: 16px;
            line-height: 1.5;
        }

        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
            padding: 0 16px; 
        }

        /* white panels around content sections (with rounded corners and shadow) */
        .card {
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        /* styles the dashboard statistic boxes */
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }
        
        .stat {
            text-align: center;
            padding: 20px;
            border-radius: 8px;
            background: #e3f2fd;
        }
        
        .stat h3 { 
            margin: 0; 
            font-size: 2.2rem; 
            color: var(--blue);
            word-break: break-word; 
        }
        
        .stat p { 
            margin: 8px 0 0 0; 
            font-size: 1rem; 
            color: var(--muted);
        }

        .not-answered-stat { background: #fff3e0; }
        .not-answered-stat h3 { color: var(--orange); }

        /* makes data tables clean, scrollable, and consistent */
        .table-container {
            width: 100%;
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
            margin-top: 20px;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            min-width: 600px;
        }
        
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid var(--border);
            word-wrap: break-word;
        }
        
        th { 
            background-color: var(--light-gray);
            font-weight: bold;
            position: sticky;
            top: 0;
            z-index: 3;
        }

        /* Buttons */
        .btn { 
            background: var(--blue); 
            color: white; 
            padding: 10px 20px; 
            border-radius: 4px; 
            display: inline-block; 
            margin: 5px; 
            border: none; 
            cursor: pointer; 
            font-size: 14px;
            text-decoration: none;
            min-height: 44px;
            line-height: 1.4;
            text-align: center;
        }
        
        .btn:hover { background: var(--blue-dark); }
        .btn-success { background: var(--green); }
        .btn-success:hover { background: var(--green-dark); }
        .btn-warning { background: var(--orange); }
        .btn-warning:hover { background: var(--orange-dark); }
        .btn-danger { background: var(--red); }
        .btn-danger:hover { background: var(--red-dark); }
        .btn-secondary { background: var(--gray); }
        .btn-secondary:hover { background: var(--gray-dark); }

        /* Typography */
        h1, h2 { 
            color: #333;
            word-wrap: break-word; 
        }
        h1 { font-size: 2rem; margin-bottom: 1rem; }
        h2 { font-size: 1.4rem; margin-bottom: 0.8rem; }

        /* styles the top navigation area — the blue tabs or menu links */
        .nav {
            background: var(--blue);
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }
        
        .nav a { 
            color: white; 
            text-decoration: none;
            padding: 5px 10px;
            border-radius: 4px;
            transition: background-color 0.2s;
        }
        
        .nav a:hover { 
            background-color: rgba(255,255,255,0.1);
            text-decoration: none;
        }

        /* Form Elements */
        .form-group { margin: 20px 0; }
        
        label { 
            display: block; 
            margin-bottom: 5px; 
            font-weight: bold; 
        }
        
        input, textarea { 
            width: 100%; 
            padding: 12px; 
            border: 1px solid var(--border); 
            border-radius: 4px; 
            font-size: 16px;
            font-family: inherit;
        }
        
        input:focus, textarea:focus {
            outline: none;
            border-color: var(--blue);
            box-shadow: 0 0 0 2px rgba(25, 118, 210, 0.2);
        }
        
        textarea { 
            height: 100px; 
            resize: vertical; 
        }

        /* Special Elements */
        .prefill-notice { 
            background: #e8f5e8; 
            padding: 10px; 
            border-radius: 4px; 
            margin-bottom: 15px; 
            color: #2e7d32; 
            border-left: 4px solid var(--green);
        }
        
        .needs-attention { 
            background: #fff3e0; 
            border-left: 4px solid var(--orange);
        }
        
        .section-header { 
            background: var(--light-gray); 
            padding: 15px; 
            margin: 20px 0 10px 0; 
            border-radius: 4px; 
            font-weight: bold; 
        }
        
        .success { 
            background: #e8f5e8;
            border-left: 4px solid var(--green);
        }

        /* FAQ Page Specific */
        .faq-table {
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .faq-table table { margin: 0; }
        
        .faq-table tr:hover {
            background: #f8f9fa;
            transition: background 0.2s ease;
        }
        
        .faq-header {
            background: linear-gradient(135deg, var(--blue), var(--blue-dark));
            color: white;
            padding: 20px;
            text-align: center;
        }
        
        .faq-stats {
            background: var(--light-gray);
            padding: 15px;
            text-align: center;
            border-bottom: 1px solid var(--border);
            font-size: 14px;
            color: var(--muted);
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            body { margin: 15px; }
            .container { padding: 0 10px; }
            .card { padding: 15px; margin: 15px 0; }
            
            .stats { 
                grid-template-columns: repeat(2, 1fr);
                gap: 15px;
            }
            
            .stat { padding: 15px; }
            .stat h3 { font-size: 2rem; }
            .stat p { font-size: 0.9rem; }
            
            th, td { 
                padding: 10px 8px; 
                font-size: 14px;
            }
            
            h1 { font-size: 1.8rem; }
            h2 { font-size: 1.3rem; }
            
            .btn { 
                padding: 12px 16px;
                font-size: 14px;
            }
            
            .nav { 
                padding: 12px;
                gap: 8px;
            }
            
            table { min-width: 500px; }
        }

        @media (max-width: 480px) {
            body { margin: 10px; }
            .card { padding: 12px; margin: 10px 0; }
            
            .stats { 
                grid-template-columns: 1fr;
                gap: 12px;
            }
            
            .stat { padding: 12px; }
            .stat h3 { font-size: 1.8rem; }
            .stat p { font-size: 0.85rem; }
            
            th, td { 
                padding: 8px 6px; 
                font-size: 13px;
            }
            
            h1 { font-size: 1.6rem; }
            h2 { font-size: 1.2rem; }
            
            .btn { 
                padding: 10px 14px;
                font-size: 13px;
                margin: 3px 2px;
            }
            
            /* Stack form buttons on mobile */
            form .btn { 
                display: block; 
                width: 100%; 
                margin: 5px 0;
                text-align: center;
            }
            
            .nav { 
                padding: 10px;
                text-align: center;
            }
            
            .nav a { 
                display: inline-block;
                margin: 2px;
                padding: 8px 12px;
            }
            
            table { min-width: 400px; }
            
            .section-header { 
                padding: 12px;
                font-size: 0.9rem;
            }
        }

        @media (max-width: 320px) {
            .stats { grid-template-columns: 1fr; }
            .nav a { 
                display: block; 
                margin: 5px 0;
                text-align: center;
            }
            table { min-width: 300px; }
            th, td { padding: 6px 4px; }
        }
    """


def get_nav_html():
    """Navigation HTML with feedback page"""
    return """
        <div class="nav">
            <a href="/admin">Dashboard</a>
            <a href="/admin/custom-info">Manage Information</a>
            <a href="/admin/faq">FAQ Analysis</a>
            <a href="/admin/manage-queries">Manage Queries</a>
            <a href="/admin/feedback">Feedback</a>
        </div>
    """


def get_updated_dashboard_html(
    total_queries, answered_count, not_answered_count, 
    accuracy_rate, total_conversations, total_sessions, 
    total_custom_info, recent_queries
):
    """Builds the HTML for the Admin Dashboard page"""

    # Start with an empty string for the table of recent queries
    recent_queries_html = ""

    # Go through each query record in the list
    for query in recent_queries:
        # Format the timestamp (only keep date and time)
        timestamp = query.get('timestamp', '')[:19].replace('T', ' ')
        
        # Get the query text and shorten it if too long
        query_text = query.get('query_text', '')
        query_display = query_text[:80] + ('...' if len(query_text) > 80 else '')
        
        # Get the response text (column name is 'answer_text' in your CSV)
        response_text = query.get('answer_text', 'No response recorded')
        response_display = response_text[:100] + ('...' if len(response_text) > 100 else '')
        
        # Check if the query was answered
        answered_field = (query.get('answered') or '').strip().lower()
        if answered_field in ["true", "1", "yes"]:
            status_icon = '✅'  # Mark as answered
        else:
            status_icon = '❌'  # Mark as not answered
            
        # Add one row for this query in the table
        recent_queries_html += f"""
            <tr>
                <td>{timestamp}</td>
                <td title="{query_text}">{query_display}</td>
                <td title="{response_text}">{response_display}</td>
                <td style="text-align: center;">{status_icon}</td>
            </tr>"""
    
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ChatBook AI Admin Panel</title>
        <style>{get_base_style()}</style>
    </head>
    <body>
        <div class="container">
            {get_nav_html()}
            
            <h1>ChatBook AI Admin Dashboard</h1>
            
            <div class="card">
                <h2>System Overview</h2>
                <div class="stats">
                    <div class="stat">
                        <h3>{total_queries}</h3>
                        <p>Total Queries</p>
                    </div>
                    <div class="stat">
                        <h3>{accuracy_rate:.1f}%</h3>
                        <p>Success Rate</p>
                    </div>
                    <div class="stat">
                        <h3>{answered_count}</h3>
                        <p>Successfully Answered</p>
                    </div>
                    <div class="stat not-answered-stat">
                        <h3>{not_answered_count}</h3>
                        <p>Not Answered</p>
                    </div>
                    <div class="stat">
                        <h3>{total_conversations}</h3>
                        <p>Total Conversations</p>
                    </div>
                    <div class="stat">
                        <h3>{total_sessions}</h3>
                        <p>Active Sessions</p>
                    </div>
                    <div class="stat">
                        <h3>{total_custom_info}</h3>
                        <p>Information Items</p>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>Recent Queries</h2>
                <p style="color: var(--muted); margin-bottom: 15px;">✅ = Successfully Answered | ❌ = Not Answered</p>
                <div class="table-container">
                    <table>
                        <tr>
                            <th>Time</th>
                            <th>Query</th>
                            <th>Response</th>
                            <th>Status</th>
                        </tr>
                        {recent_queries_html if recent_queries_html else '<tr><td colspan="4" style="text-align: center; color: var(--muted);">No recent queries</td></tr>'}
                    </table>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    """Generate HTML for Custom Information management page with responsive design"""
 
    # .items() returns a list of (key, value) pairs
    sorted_info = sorted(custom_info.items(), key=lambda x: x[1]['added_at'], reverse=True)

    info_html = ""
    for info_id, info in sorted_info:
        topic = info['topic']
        information = info['information'][:100] + ('...' if len(info['information']) > 100 else '')
        added = info['added_at'][:10] # first 10 chars of a timestamp like 2025-10-18
        
        info_html += f"""
            <tr>
                <td>{topic}</td>
                <td>{information}</td>
                <td>{added}</td>
                <td>
                    <a href="/admin/custom-info/delete/{info_id}" class="btn btn-danger" 
                       onclick="return confirm('Delete this information?')">Delete</a>
                </td>
            </tr>"""
    
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Manage Information</title>
        <style>{get_base_style()}</style>
        <script>
            window.addEventListener('DOMContentLoaded', function() {{
                // window.location.search gives you the query part of the current URL
                // new URLSearchParams(...) creates an easy-to-use object for reading or modifying those query parameters.
                const urlParams = new URLSearchParams(window.location.search);

                if (urlParams.get('upload') === 'success') {{
                    const successDiv = document.getElementById('success-message');
                    if (successDiv) successDiv.style.display = 'block';
                    const url = new URL(window.location);
                    url.searchParams.delete('upload'); // Removes the upload query parameter
                    window.history.replaceState({{}}, '', url); // update the browser's address bar silently
                }}
            }});
        </script>
    </head>
    <body>
        <div class="container">
            {get_nav_html()}
            
            <h1>Manage Information</h1>
            
            <div id="success-message" style="display: none; background: #e8f5e8; border-left: 4px solid var(--green); padding: 15px; margin-bottom: 20px; border-radius: 4px;">
                <strong>Success!</strong> Handbook PDF has been updated successfully.
            </div>
            
            <div class="card">
                <h2>Quick Actions</h2>
                <a href="/admin/custom-info/add" class="btn">Add New Information</a>
                <a href="/admin/upload-handbook" class="btn btn-warning">Update Handbook PDF</a>
                <a href="/admin/export-data" class="btn btn-secondary">Export System Data</a>
                <p style="color: var(--muted); margin-top: 10px;">Add new information, update the handbook PDF, or export all system data as backup.</p>
            </div>
            
            <div class="card">
                <h2>Existing Information ({len(sorted_info)} items)</h2>
                <div class="table-container">
                    <table>
                        <tr>
                            <th>Topic</th>
                            <th>Information</th>
                            <th>Added</th>
                            <th>Actions</th>
                        </tr>
                        {info_html if info_html else '<tr><td colspan="4" style="text-align: center; color: var(--muted);">No custom information added yet</td></tr>'}
                    </table>
                </div>
                
                {f'''
                <div style="margin-top: 20px; padding: 15px; background: var(--light-gray); border-radius: 4px;">
                    <p style="margin: 0; color: var(--muted);">
                        <strong>Tip:</strong> Information is displayed newest first. Click "Delete" to remove outdated information.
                    </p>
                </div>
                ''' if info_html else ''}
            </div>
        </div>
    </body>
    </html>
    """
    """Feedback management HTML with enhanced statistics"""
    
    feedback_rows = ""
    for feedback in feedback_list:
        timestamp = feedback.get('timestamp', '')[:19].replace('T', ' ')
        feedback_text = feedback.get('feedback_text', '')
        rating = feedback.get('rating', 1)  # Default to 1 since that's the minimum rating
        user_type = feedback.get('user_type', 'student')
        
        # Truncate long feedback for display
        display_text = feedback_text[:100] + ('...' if len(feedback_text) > 100 else '')
        
        # Star rating display
        stars = '⭐' * int(rating) + '☆' * (5 - int(rating))
        
        feedback_rows += f"""
            <tr>
                <td>{timestamp}</td>
                <td title="{feedback_text}">{display_text}</td>
                <td style="text-align: center;">{stars}<br><small>({rating}/5)</small></td>
                <td style="text-align: center; text-transform: capitalize;">{user_type}</td>
            </tr>"""
    
    # Calculate user type distribution (Counter already imported at top of admin.py)
    user_type_counter = {}
    for feedback in feedback_list:
        user_type = feedback.get('user_type', 'student').lower()
        user_type_counter[user_type] = user_type_counter.get(user_type, 0) + 1
    
    # Rating distribution bars
    rating_bars = ""
    for rating in range(5, 0, -1):
        count = rating_distribution.get(rating, 0)
        percentage = (count / total_feedback * 100) if total_feedback > 0 else 0
        rating_bars += f"""
            <div style="display: flex; align-items: center; margin: 5px 0;">
                <span style="width: 60px;">{rating} ⭐</span>
                <div style="flex: 1; background: #e0e0e0; border-radius: 10px; margin: 0 10px; height: 20px;">
                    <div style="background: #4caf50; height: 20px; border-radius: 10px; width: {percentage}%;"></div>
                </div>
                <span style="width: 80px; text-align: right;">{count} ({percentage:.1f}%)</span>
            </div>"""
    
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>User Feedback</title>
        <style>{get_base_style()}</style>
    </head>
    <body>
        <div class="container">
            {get_nav_html()}
            
            <h1>User Feedback</h1>
            
            <div class="card">
                <h2>Overall Statistics</h2>
                <div class="stats">
                    <div class="stat">
                        <h3>{total_feedback}</h3>
                        <p>Total Feedback</p>
                    </div>
                    <div class="stat">
                        <h3>{avg_rating:.1f}</h3>
                        <p>Average Rating</p>
                    </div>
                    <div class="stat">
                        <h3>{rating_distribution.get(5, 0)}</h3>
                        <p>5-Star Reviews</p>
                    </div>
                    <div class="stat">
                        <h3>{rating_distribution.get(4, 0)}</h3>
                        <p>4-Star Reviews</p>
                    </div>
                    <div class="stat">
                        <h3>{rating_distribution.get(3, 0)}</h3>
                        <p>3-Star Reviews</p>
                    </div>
                    <div class="stat">
                        <h3>{rating_distribution.get(2, 0)}</h3>
                        <p>2-Star Reviews</p>
                    </div>
                    <div class="stat">
                        <h3>{rating_distribution.get(1, 0)}</h3>
                        <p>1-Star Reviews</p>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>Feedback by User Type</h2>
                <div class="stats">
                    <div class="stat">
                        <h3>{user_type_counter.get('student', 0)}</h3>
                        <p>Students</p>
                    </div>
                    <div class="stat">
                        <h3>{user_type_counter.get('faculty', 0)}</h3>
                        <p>Faculty</p>
                    </div>
                    <div class="stat">
                        <h3>{user_type_counter.get('staff', 0)}</h3>
                        <p>Staff</p>
                    </div>
                    <div class="stat">
                        <h3>{user_type_counter.get('visitor', 0)}</h3>
                        <p>Visitors</p>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>Rating Distribution</h2>
                {rating_bars if rating_bars else '<p style="text-align: center; color: var(--muted);">No ratings yet</p>'}
            </div>
            
            <div class="card">
                <h2>All Feedback</h2>
                <div class="table-container">
                    <table>
                        <tr>
                            <th>Date</th>
                            <th>Feedback</th>
                            <th>Rating</th>
                            <th>User Type</th>
                        </tr>
                        {feedback_rows if feedback_rows else '<tr><td colspan="4" style="text-align: center; color: var(--muted);">No feedback submitted yet</td></tr>'}
                    </table>
                </div>
            </div>
        </div>
    </body>
    </html>
    """


def get_custom_info_html(custom_info):
    """Generate HTML for Custom Information management page with responsive design"""
 
    # .items() returns a list of (key, value) pairs
    sorted_info = sorted(custom_info.items(), key=lambda x: x[1]['added_at'], reverse=True)

    info_html = ""
    for info_id, info in sorted_info:
        topic = info['topic']
        information = info['information'][:100] + ('...' if len(info['information']) > 100 else '')
        added = info['added_at'][:10] # first 10 chars of a timestamp like 2025-10-18
        
        info_html += f"""
            <tr>
                <td>{topic}</td>
                <td>{information}</td>
                <td>{added}</td>
                <td>
                    <a href="/admin/custom-info/delete/{info_id}" class="btn btn-danger" 
                       onclick="return confirm('Delete this information?')">Delete</a>
                </td>
            </tr>"""
    
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Manage Information</title>
        <style>{get_base_style()}</style>
        <script>
            window.addEventListener('DOMContentLoaded', function() {{
                // window.location.search gives you the query part of the current URL
                // new URLSearchParams(...) creates an easy-to-use object for reading or modifying those query parameters.
                const urlParams = new URLSearchParams(window.location.search);

                if (urlParams.get('upload') === 'success') {{
                    const successDiv = document.getElementById('success-message');
                    if (successDiv) successDiv.style.display = 'block';
                    const url = new URL(window.location);
                    url.searchParams.delete('upload'); // Removes the upload query parameter
                    window.history.replaceState({{}}, '', url); // update the browser's address bar silently
                }}
            }});
        </script>
    </head>
    <body>
        <div class="container">
            {get_nav_html()}
            
            <h1>Manage Information</h1>
            
            <div id="success-message" style="display: none; background: #e8f5e8; border-left: 4px solid var(--green); padding: 15px; margin-bottom: 20px; border-radius: 4px;">
                <strong>Success!</strong> Handbook PDF has been updated successfully.
            </div>
            
            <div class="card">
                <h2>Quick Actions</h2>
                <a href="/admin/custom-info/add" class="btn">Add New Information</a>
                <a href="/admin/upload-handbook" class="btn btn-warning">Update Handbook PDF</a>
                <a href="/admin/export-data" class="btn btn-secondary">Export System Data</a>
                <p style="color: var(--muted); margin-top: 10px;">Add new information, update the handbook PDF, or export all system data as backup.</p>
            </div>
            
            <div class="card">
                <h2>Existing Information ({len(sorted_info)} items)</h2>
                <div class="table-container">
                    <table>
                        <tr>
                            <th>Topic</th>
                            <th>Information</th>
                            <th>Added</th>
                            <th>Actions</th>
                        </tr>
                        {info_html if info_html else '<tr><td colspan="4" style="text-align: center; color: var(--muted);">No custom information added yet</td></tr>'}
                    </table>
                </div>
                
                {f'''
                <div style="margin-top: 20px; padding: 15px; background: var(--light-gray); border-radius: 4px;">
                    <p style="margin: 0; color: var(--muted);">
                        <strong>Tip:</strong> Information is displayed newest first. Click "Delete" to remove outdated information.
                    </p>
                </div>
                ''' if info_html else ''}
            </div>
        </div>
    </body>
    </html>
    """
    """Feedback management HTML with enhanced statistics"""
    
    feedback_rows = ""
    for feedback in feedback_list:
        timestamp = feedback.get('timestamp', '')[:19].replace('T', ' ')
        feedback_text = feedback.get('feedback_text', '')
        rating = feedback.get('rating', 1)  # Default to 1 since that's the minimum rating
        user_type = feedback.get('user_type', 'student')
        
        # Truncate long feedback for display
        display_text = feedback_text[:100] + ('...' if len(feedback_text) > 100 else '')
        
        # Star rating display
        stars = '⭐' * int(rating) + '☆' * (5 - int(rating))
        
        feedback_rows += f"""
            <tr>
                <td>{timestamp}</td>
                <td title="{feedback_text}">{display_text}</td>
                <td style="text-align: center;">{stars}<br><small>({rating}/5)</small></td>
                <td style="text-align: center; text-transform: capitalize;">{user_type}</td>
            </tr>"""
    
    # Calculate user type distribution (Counter already imported at top of admin.py)
    user_type_counter = {}
    for feedback in feedback_list:
        user_type = feedback.get('user_type', 'student').lower()
        user_type_counter[user_type] = user_type_counter.get(user_type, 0) + 1
    
    # Rating distribution bars
    rating_bars = ""
    for rating in range(5, 0, -1):
        count = rating_distribution.get(rating, 0)
        percentage = (count / total_feedback * 100) if total_feedback > 0 else 0
        rating_bars += f"""
            <div style="display: flex; align-items: center; margin: 5px 0;">
                <span style="width: 60px;">{rating} ⭐</span>
                <div style="flex: 1; background: #e0e0e0; border-radius: 10px; margin: 0 10px; height: 20px;">
                    <div style="background: #4caf50; height: 20px; border-radius: 10px; width: {percentage}%;"></div>
                </div>
                <span style="width: 80px; text-align: right;">{count} ({percentage:.1f}%)</span>
            </div>"""
    
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>User Feedback</title>
        <style>{get_base_style()}</style>
    </head>
    <body>
        <div class="container">
            {get_nav_html()}
            
            <h1>User Feedback</h1>
            
            <div class="card">
                <h2>Overall Statistics</h2>
                <div class="stats">
                    <div class="stat">
                        <h3>{total_feedback}</h3>
                        <p>Total Feedback</p>
                    </div>
                    <div class="stat">
                        <h3>{avg_rating:.1f}</h3>
                        <p>Average Rating</p>
                    </div>
                    <div class="stat">
                        <h3>{rating_distribution.get(5, 0)}</h3>
                        <p>5-Star Reviews</p>
                    </div>
                    <div class="stat">
                        <h3>{rating_distribution.get(4, 0)}</h3>
                        <p>4-Star Reviews</p>
                    </div>
                    <div class="stat">
                        <h3>{rating_distribution.get(3, 0)}</h3>
                        <p>3-Star Reviews</p>
                    </div>
                    <div class="stat">
                        <h3>{rating_distribution.get(2, 0)}</h3>
                        <p>2-Star Reviews</p>
                    </div>
                    <div class="stat">
                        <h3>{rating_distribution.get(1, 0)}</h3>
                        <p>1-Star Reviews</p>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>Feedback by User Type</h2>
                <div class="stats">
                    <div class="stat">
                        <h3>{user_type_counter.get('student', 0)}</h3>
                        <p>Students</p>
                    </div>
                    <div class="stat">
                        <h3>{user_type_counter.get('faculty', 0)}</h3>
                        <p>Faculty</p>
                    </div>
                    <div class="stat">
                        <h3>{user_type_counter.get('staff', 0)}</h3>
                        <p>Staff</p>
                    </div>
                    <div class="stat">
                        <h3>{user_type_counter.get('visitor', 0)}</h3>
                        <p>Visitors</p>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>Rating Distribution</h2>
                {rating_bars if rating_bars else '<p style="text-align: center; color: var(--muted);">No ratings yet</p>'}
            </div>
            
            <div class="card">
                <h2>All Feedback</h2>
                <div class="table-container">
                    <table>
                        <tr>
                            <th>Date</th>
                            <th>Feedback</th>
                            <th>Rating</th>
                            <th>User Type</th>
                        </tr>
                        {feedback_rows if feedback_rows else '<tr><td colspan="4" style="text-align: center; color: var(--muted);">No feedback submitted yet</td></tr>'}
                    </table>
                </div>
            </div>
        </div>
    </body>
    </html>
    """


def get_add_custom_info_form_html(prefilled_topic=""):
    """Add custom info form HTML with responsive design and optional pre-filled topic"""

    prefill_notice = ""
    if prefilled_topic:
        prefill_notice = f"""
            <div class="prefill-notice">
                ℹ️ Topic has been pre-filled from FAQ analysis. You can modify it if needed.
            </div>
        """
    
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Add Information</title>
        <style>{get_base_style()}</style>
    </head>
    <body>
        <div class="container">
            {get_nav_html()}
            
            <h1>Add New Information</h1>
            
            <div class="card">
                {prefill_notice}

                <form method="post" action="/admin/custom-info/add">
                    <div class="form-group">
                        <label for="topic">Topic:</label>
                        <input type="text" id="topic" name="topic" required 
                               value="{prefilled_topic}" 
                               placeholder="e.g., Campus WiFi Password, Admission Requirements, Library Hours"
                               style="font-size: 16px;">
                        <small style="color: var(--muted); display: block; margin-top: 5px;">
                            Keep it concise but descriptive - this helps the chatbot match user questions
                        </small>
                    </div>
                    
                    <div class="form-group">
                        <label for="information">Information:</label>
                        <textarea id="information" name="information" required 
                                  placeholder="Provide the complete and accurate information about this topic that the chatbot should use in its responses.&#10;&#10;Example:&#10;The campus WiFi network is 'GSU-Student' with password 'StudentLife2024'. Connect by selecting the network and entering the password. For technical support, contact IT Help Desk at (555) 123-4567."
                                  style="min-height: 120px; font-size: 16px; font-family: inherit;"></textarea>
                    </div>
                    
                    <div style="display: flex; gap: 10px; flex-wrap: wrap; justify-content: flex-end; margin-top: 20px;">
                        <a href="/admin/custom-info" class="btn btn-secondary">Cancel</a>
                        <button type="submit" class="btn">Add Information</button>
                    </div>
                </form>
            </div>
        </div>
    </body>
    </html>
    """


def get_full_faq_html(faq_list, not_answered_list):
    """Full FAQ page HTML with side-by-side layout"""

    faq_html = ""
    for index, faq in enumerate(faq_list):
        rank = index + 1
        question = faq['question']
        count = faq['total_asked']
        success_rate = faq['success_rate']
        
        if success_rate >= 80:
            status_color = "#4caf50"
            status_icon = "✅"
        elif success_rate >= 50:
            status_color = "#ff9800"
            status_icon = "⚠️"
        else:
            status_color = "#f44336"
            status_icon = "❌"
        
        display_question = question.capitalize() if question else "Unknown question"
        
        faq_html += f"""
            <tr>
                <td style="text-align: center; width: 60px;">
                    <div style="
                        width: 40px; 
                        height: 40px; 
                        background: linear-gradient(135deg, var(--blue), #42a5f5);
                        color: white; 
                        border-radius: 50%; 
                        display: flex; 
                        align-items: center; 
                        justify-content: center; 
                        font-weight: bold; 
                        font-size: 16px;
                        margin: 0 auto;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                    ">
                        {rank}
                    </div>
                </td>
                <td style="padding: 15px;">
                    <div style="font-weight: 500; font-size: 16px; margin-bottom: 5px;">
                        {display_question}
                    </div>
                    <div style="color: var(--muted); font-size: 14px;">
                        Asked {count} time{'s' if count != 1 else ''} • {success_rate:.1f}% success rate
                    </div>
                </td>
                <td style="text-align: center; width: 80px;">
                    <div style="color: {status_color}; font-size: 18px;">
                        {status_icon}
                    </div>
                </td>
            </tr>"""

    not_answered_html = ""
    for question, count, is_resolved in not_answered_list:
        query_encoded = quote(question)
        display_question = question.capitalize() if question else "Unknown question"
        
        if is_resolved:
            status_badge = '<span style="background: #4caf50; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px;">Resolved</span>'
            row_style = 'opacity: 0.6;'
        else:
            status_badge = '<span style="background: #f44336; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px;">Needs Attention</span>'
            row_style = ''
        
        not_answered_html += f"""
            <tr style="{row_style}">
                <td style="padding: 15px;">
                    <div style="font-weight: 500; font-size: 16px; margin-bottom: 5px;">
                        {display_question}
                    </div>
                    <div style="color: var(--muted); font-size: 14px;">
                        Asked {count} time{'s' if count != 1 else ''}
                    </div>
                </td>
                <td style="text-align: center; padding: 15px;">{status_badge}</td>
                <td style="padding: 15px; text-align: center;">
                    <a href="/admin/custom-info/add?prefill_topic={query_encoded}" 
                       class="btn btn-success" style="font-size: 13px; padding: 8px 12px;">Add Info</a>
                    {'' if is_resolved else f'''
                    <form method="post" action="/admin/mark-resolved" style="display: inline;">
                        <input type="hidden" name="question" value="{question}">
                        <button type="submit" class="btn btn-secondary" style="font-size: 13px; padding: 8px 12px;"
                                onclick="return confirm('Mark this question as resolved?')">
                            Mark Resolved
                        </button>
                    </form>
                    '''}
                </td>
            </tr>"""

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Frequently Asked Questions</title>
        <style>
            {get_base_style()}
            
            .faq-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin-top: 20px;
            }}
            
            @media (max-width: 1024px) {{
                .faq-grid {{
                    grid-template-columns: 1fr;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            {get_nav_html()}
            
            <h1>FAQ Analysis</h1>
            
            <div class="faq-grid">
                <!-- LEFT: Most Popular Questions -->
                <div class="card faq-table">
                    <div class="faq-header">
                        <h2 style="margin: 0; color: white;">Most Popular Questions</h2>
                        <p style="margin: 10px 0 0 0; opacity: 0.9;">Questions asked by students, ranked by frequency</p>
                    </div>
                    
                    <div class="faq-stats">
                        Showing {len(faq_list)} frequently asked questions (asked 2+ times each)
                    </div>
                    
                    <div class="table-container">
                        <table style="margin: 0;">
                            <tr style="background: #fafafa; border-bottom: 2px solid #e0e0e0;">
                                <th style="text-align: center; width: 60px; padding: 15px;">Rank</th>
                                <th style="padding: 15px;">Question</th>
                                <th style="text-align: center; width: 80px; padding: 15px;">Status</th>
                            </tr>
                            {faq_html if faq_html else '''
                            <tr>
                                <td colspan="3" style="text-align: center; padding: 40px; color: var(--muted); font-style: italic;">
                                    No frequently asked questions yet.<br>
                                    Questions need to be asked at least twice to appear here.
                                </td>
                            </tr>
                            '''}
                        </table>
                    </div>
                </div>
                
                <!-- RIGHT: Unanswered Questions -->
                <div class="card faq-table">
                    <div class="faq-header" style="background: linear-gradient(135deg, #f44336, #d32f2f);">
                        <h2 style="margin: 0; color: white;">Unanswered Questions</h2>
                        <p style="margin: 10px 0 0 0; opacity: 0.9;">Questions where the chatbot could not provide an answer</p>
                    </div>
                    
                    <div class="faq-stats">
                        Showing {len(not_answered_list)} unanswered questions
                    </div>
                    
                    <div class="table-container">
                        <table style="margin: 0;">
                            <tr style="background: #fafafa; border-bottom: 2px solid #e0e0e0;">
                                <th style="padding: 15px;">Question</th>
                                <th style="text-align: center; width: 120px; padding: 15px;">Status</th>
                                <th style="text-align: center; width: 200px; padding: 15px;">Actions</th>
                            </tr>
                            {not_answered_html if not_answered_html else '''
                            <tr>
                                <td colspan="3" style="text-align: center; padding: 40px; color: var(--muted); font-style: italic;">
                                    🎉 Excellent! No unanswered questions at the moment.
                                </td>
                            </tr>
                            '''}
                        </table>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>Legend & Actions</h2>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; align-items: center;">
                    <div>
                        <h3 style="margin-bottom: 10px;">Status Indicators:</h3>
                        <div style="display: flex; flex-direction: column; gap: 8px;">
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <span style="color: #4caf50; font-size: 18px;">✅</span>
                                <span>High success rate (80%+)</span>
                            </div>
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <span style="color: #ff9800; font-size: 18px;">⚠️</span>
                                <span>Medium success rate (50-79%)</span>
                            </div>
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <span style="color: #f44336; font-size: 18px;">❌</span>
                                <span>Low success rate (<50%)</span>
                            </div>
                        </div>
                    </div>
                    <div>
                        <h3 style="margin-bottom: 10px;">Unanswered Status:</h3>
                        <div style="display: flex; flex-direction: column; gap: 8px;">
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <span style="background: #f44336; color: white; padding: 4px 8px; border-radius: 4px; font-size: 11px;">Needs Attention</span>
                                <span>Not yet addressed</span>
                            </div>
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <span style="background: #4caf50; color: white; padding: 4px 8px; border-radius: 4px; font-size: 11px;">Resolved</span>
                                <span>Marked as handled</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """


def get_manage_queries_with_resolved_html(all_needing_attention):
    """Manage Queries HTML with multi-select functionality"""
    
    attention_html = ""
    for query, count in all_needing_attention:
        query_encoded = quote(query)
        query_display = query[:80] + ('...' if len(query) > 80 else '')
        
        attention_html += f"""
            <tr>
                <td style="text-align: center; width: 50px;">
                    <input type="checkbox" name="selected_queries" value="{query}" 
                           class="query-checkbox" style="width: 18px; height: 18px; cursor: pointer;">
                </td>
                <td title="{query}">{query_display}</td>
                <td style="text-align: center;">{count}</td>
                <td style="text-align: center;">
                    <a href="/admin/custom-info/add?prefill_topic={query_encoded}" 
                       class="btn btn-success">Add Info</a>
                </td>
            </tr>"""

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Manage Queries</title>
        <style>{get_base_style()}</style>
        <script>
            function toggleSelectAll(source) {{
                const checkboxes = document.querySelectorAll('.query-checkbox');
                checkboxes.forEach(checkbox => {{
                    checkbox.checked = source.checked;
                }});
                updateButtonStates();
            }}
            
            function updateButtonStates() {{
                const checkboxes = document.querySelectorAll('.query-checkbox:checked');
                const bulkResolveBtn = document.getElementById('bulk-resolve-btn');
                const selectAllCheckbox = document.getElementById('select-all');
                const selectedCount = document.getElementById('selected-count');
                
                if (checkboxes.length > 0) {{
                    bulkResolveBtn.disabled = false;
                    bulkResolveBtn.style.opacity = '1';
                    selectedCount.textContent = `(${{checkboxes.length}} selected)`;
                    selectedCount.style.display = 'inline';
                }} else {{
                    bulkResolveBtn.disabled = true;
                    bulkResolveBtn.style.opacity = '0.5';
                    selectedCount.style.display = 'none';
                }}
                
                // Update select all checkbox state
                const allCheckboxes = document.querySelectorAll('.query-checkbox');
                selectAllCheckbox.checked = allCheckboxes.length > 0 && 
                                           checkboxes.length === allCheckboxes.length;
            }}
            
            function submitBulkResolve() {{
                const checkboxes = document.querySelectorAll('.query-checkbox:checked');
                if (checkboxes.length === 0) {{
                    alert('Please select at least one query to mark as resolved.');
                    return false;
                }}
                
                const count = checkboxes.length;
                const confirmMsg = `Mark ${{count}} question${{count > 1 ? 's' : ''}} as resolved?`;
                
                if (confirm(confirmMsg)) {{
                    document.getElementById('bulk-resolve-form').submit();
                }}
                return false;
            }}
            
            // Add event listeners when page loads
            document.addEventListener('DOMContentLoaded', function() {{
                document.querySelectorAll('.query-checkbox').forEach(checkbox => {{
                    checkbox.addEventListener('change', updateButtonStates);
                }});
                updateButtonStates();
            }});
        </script>
    </head>
    <body>
        <div class="container">
            {get_nav_html()}
            
            <h1>Manage Queries</h1>
            
            <div class="card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; flex-wrap: wrap; gap: 10px;">
                    <div>
                        <h2 style="margin: 0;">Unresolved Queries</h2>
                        <p style="margin: 5px 0 0 0; color: var(--muted);">Questions that have not been successfully answered</p>
                    </div>
                    <div>
                        <button id="bulk-resolve-btn" onclick="submitBulkResolve()" 
                                class="btn btn-warning" disabled style="opacity: 0.5;">
                            Mark as Resolved <span id="selected-count" style="display: none;"></span>
                        </button>
                    </div>
                </div>
                
                <form id="bulk-resolve-form" method="post" action="/admin/bulk-mark-resolved">
                    <div class="table-container">
                        <table>
                            <tr>
                                <th style="text-align: center; width: 50px;">
                                    <input type="checkbox" id="select-all" onchange="toggleSelectAll(this)" 
                                           style="width: 18px; height: 18px; cursor: pointer;"
                                           title="Select/Deselect All">
                                </th>
                                <th>Question</th>
                                <th style="text-align: center; width: 120px;">Times Asked</th>
                                <th style="text-align: center; width: 150px;">Actions</th>
                            </tr>
                            {attention_html if attention_html else '<tr><td colspan="4" style="text-align: center; color: var(--muted); padding: 40px;">No unresolved queries - excellent job!</td></tr>'}
                        </table>
                    </div>
                </form>
                
                <div style="margin-top: 20px; padding: 15px; background: var(--light-gray); border-radius: 4px;">
                    <p style="margin: 0; color: var(--muted);">
                        <strong>💡 Tip:</strong> Check the boxes next to questions and click "Mark as Resolved" to handle multiple queries at once, 
                        or use "Add Info" to provide missing information for individual questions.
                    </p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """


def get_upload_handbook_html(handbook_info):
    """Upload handbook PDF form HTML"""
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Update Handbook PDF</title>
        <style>{get_base_style()}</style>
    </head>
    <body>
        <div class="container">
            {get_nav_html()}
            
            <h1>Update Handbook PDF</h1>
            
            <div class="card">
                <h2>Current Handbook Status</h2>
                <p style="padding: 15px; background: var(--light-gray); border-radius: 4px; margin-bottom: 20px;">
                    {handbook_info}
                </p>
                
                <div style="background: #fff3e0; border-left: 4px solid var(--orange); padding: 15px; margin-bottom: 20px; border-radius: 4px;">
                    <strong>Warning:</strong> Uploading a new handbook will replace the existing one. 
                    A backup will be created automatically during the upload process.
                </div>
                
                <form method="post" action="/admin/upload-handbook" enctype="multipart/form-data" id="uploadForm">
                    <div class="form-group">
                        <label for="file">Select PDF File:</label>
                        <input type="file" id="file" name="file" accept=".pdf" required 
                               style="font-size: 16px; padding: 10px; border: 2px dashed var(--border); border-radius: 4px;">
                        <small style="color: var(--muted); display: block; margin-top: 8px;">
                            Only PDF files are accepted.
                        </small>
                    </div>
                    
                    <div id="uploadProgress" style="display: none; margin: 20px 0;">
                        <div style="background: var(--light-gray); height: 30px; border-radius: 15px; overflow: hidden;">
                            <div id="progressBar" style="background: linear-gradient(90deg, var(--blue), var(--green)); height: 100%; width: 0%; transition: width 0.3s; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 14px;"></div>
                        </div>
                        <p id="progressText" style="text-align: center; margin-top: 8px; color: var(--muted);"></p>
                    </div>
                    
                    <div style="display: flex; gap: 10px; flex-wrap: wrap; justify-content: flex-end; margin-top: 20px;">
                        <a href="/admin/custom-info" class="btn btn-secondary">Cancel</a>
                        <button type="submit" class="btn btn-warning" id="uploadBtn">Upload Handbook</button>
                    </div>
                </form>
            </div>
            
            <div class="card">
                <h2>Important Notes</h2>
                <ul style="line-height: 1.8;">
                    <li>The chatbot will automatically use the new handbook once uploaded</li>
                    <li>Large files may take a moment to upload - please be patient</li>
                    <li>Ensure the PDF is not corrupted before uploading</li>
                    <li>The previous handbook will be backed up automatically</li>
                </ul>
            </div>
        </div>
        
        <script>
            const form = document.getElementById('uploadForm');
            const uploadBtn = document.getElementById('uploadBtn');
            const uploadProgress = document.getElementById('uploadProgress');
            const progressBar = document.getElementById('progressBar');
            const progressText = document.getElementById('progressText');
            const fileInput = document.getElementById('file');
            
            fileInput.addEventListener('change', function(e) {{
                const file = e.target.files[0];
                if (file) {{
                    if (!file.name.toLowerCase().endsWith('.pdf')) {{
                        alert('Please select a PDF file only.');
                        fileInput.value = '';
                        return;
                    }}
                    const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
                    progressText.textContent = `Selected: ${{file.name}} (${{sizeMB}} MB)`;
                    progressText.style.display = 'block';
                }}
            }});
            
            form.addEventListener('submit', function(e) {{
                const file = fileInput.files[0];
                if (!file) {{
                    e.preventDefault();
                    alert('Please select a PDF file to upload.');
                    return;
                }}
                
                uploadBtn.disabled = true;
                uploadBtn.textContent = 'Uploading...';
                uploadProgress.style.display = 'block';
                progressBar.style.width = '0%';
                progressBar.textContent = '0%';
                progressText.textContent = 'Uploading handbook...';
                
                // creates a fake progress animation (because real upload progress requires a backend API event)
                let progress = 0;
                const interval = setInterval(function() {{
                    progress += 5;
                    if (progress <= 90) {{
                        progressBar.style.width = progress + '%';
                        progressBar.textContent = progress + '%';
                    }}
                }}, 200);
                
                // After 5 seconds (5000 ms), it stops the fake animation
                setTimeout(function() {{
                    clearInterval(interval);
                }}, 5000);
            }});
        </script>
    </body>
    </html>
    """


def get_feedback_html(feedback_list, total_feedback, avg_rating, rating_distribution):
    """Feedback management HTML with enhanced statistics"""
    
    feedback_rows = ""
    for feedback in feedback_list:
        timestamp = feedback.get('timestamp', '')[:19].replace('T', ' ')
        feedback_text = feedback.get('feedback_text', '')
        rating = feedback.get('rating', 1)  # Default to 1 since that's the minimum rating
        user_type = feedback.get('user_type', 'student')
        
        # Truncate long feedback for display
        display_text = feedback_text[:100] + ('...' if len(feedback_text) > 100 else '')
        
        # Star rating display
        stars = '⭐' * int(rating) + '☆' * (5 - int(rating))
        
        feedback_rows += f"""
            <tr>
                <td>{timestamp}</td>
                <td title="{feedback_text}">{display_text}</td>
                <td style="text-align: center;">{stars}<br><small>({rating}/5)</small></td>
                <td style="text-align: center; text-transform: capitalize;">{user_type}</td>
            </tr>"""
    
    # Calculate user type distribution (Counter already imported at top of admin.py)
    user_type_counter = {}
    for feedback in feedback_list:
        user_type = feedback.get('user_type', 'student').lower()
        user_type_counter[user_type] = user_type_counter.get(user_type, 0) + 1
    
    # Rating distribution bars
    rating_bars = ""
    for rating in range(5, 0, -1):
        count = rating_distribution.get(rating, 0)
        percentage = (count / total_feedback * 100) if total_feedback > 0 else 0
        rating_bars += f"""
            <div style="display: flex; align-items: center; margin: 5px 0;">
                <span style="width: 60px;">{rating} ⭐</span>
                <div style="flex: 1; background: #e0e0e0; border-radius: 10px; margin: 0 10px; height: 20px;">
                    <div style="background: #4caf50; height: 20px; border-radius: 10px; width: {percentage}%;"></div>
                </div>
                <span style="width: 80px; text-align: right;">{count} ({percentage:.1f}%)</span>
            </div>"""
    
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>User Feedback</title>
        <style>{get_base_style()}</style>
    </head>
    <body>
        <div class="container">
            {get_nav_html()}
            
            <h1>User Feedback</h1>
            
            <div class="card">
                <h2>Overall Statistics</h2>
                <div class="stats">
                    <div class="stat">
                        <h3>{total_feedback}</h3>
                        <p>Total Feedback</p>
                    </div>
                    <div class="stat">
                        <h3>{avg_rating:.1f}</h3>
                        <p>Average Rating</p>
                    </div>
                    <div class="stat">
                        <h3>{rating_distribution.get(5, 0)}</h3>
                        <p>5-Star Reviews</p>
                    </div>
                    <div class="stat">
                        <h3>{rating_distribution.get(4, 0)}</h3>
                        <p>4-Star Reviews</p>
                    </div>
                    <div class="stat">
                        <h3>{rating_distribution.get(3, 0)}</h3>
                        <p>3-Star Reviews</p>
                    </div>
                    <div class="stat">
                        <h3>{rating_distribution.get(2, 0)}</h3>
                        <p>2-Star Reviews</p>
                    </div>
                    <div class="stat">
                        <h3>{rating_distribution.get(1, 0)}</h3>
                        <p>1-Star Reviews</p>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>Feedback by User Type</h2>
                <div class="stats">
                    <div class="stat">
                        <h3>{user_type_counter.get('student', 0)}</h3>
                        <p>Students</p>
                    </div>
                    <div class="stat">
                        <h3>{user_type_counter.get('faculty', 0)}</h3>
                        <p>Faculty</p>
                    </div>
                    <div class="stat">
                        <h3>{user_type_counter.get('staff', 0)}</h3>
                        <p>Staff</p>
                    </div>
                    <div class="stat">
                        <h3>{user_type_counter.get('visitor', 0)}</h3>
                        <p>Visitors</p>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>Rating Distribution</h2>
                {rating_bars if rating_bars else '<p style="text-align: center; color: var(--muted);">No ratings yet</p>'}
            </div>
            
            <div class="card">
                <h2>All Feedback</h2>
                <div class="table-container">
                    <table>
                        <tr>
                            <th>Date</th>
                            <th>Feedback</th>
                            <th>Rating</th>
                            <th>User Type</th>
                        </tr>
                        {feedback_rows if feedback_rows else '<tr><td colspan="4" style="text-align: center; color: var(--muted);">No feedback submitted yet</td></tr>'}
                    </table>
                </div>
            </div>
        </div>
    </body>
    </html>
    """