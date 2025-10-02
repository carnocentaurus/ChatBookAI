# admin.py

from fastapi import HTTPException, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets
import os
import csv
import sqlite3
from datetime import datetime
from collections import Counter
from urllib.parse import quote, unquote

# Security setup
security = HTTPBasic()
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "gsu2025")

# Function to verify provided admin credentials
def verify_admin(credentials: HTTPBasicCredentials = Depends(security)):
    is_correct_username = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
    is_correct_password = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    
    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=401,
            detail="Invalid admin credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    return credentials.username

# Attach all admin-related routes to the FastAPI app
def setup_admin_routes(app, memory, LOG_FILE, MEMORY_DB):
    @app.get("/admin", response_class=HTMLResponse)
    async def admin_dashboard(credentials: HTTPBasicCredentials = Depends(verify_admin)):
        # Admin dashboard view (requires authentication)

        total_custom_info = len(memory.custom_info)

        # Load log data for statistics
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                reader = list(csv.DictReader(f))

            total_queries = len(reader)
            
            # Simple count based on CSV answered field
            answered_count = sum(1 for r in reader if (r.get("answered") or "").strip().lower() in ["true", "1", "yes"])
            not_answered_count = total_queries - answered_count
            accuracy_rate = (answered_count / total_queries * 100) if total_queries > 0 else 0
            
            recent_queries = list(reversed(reader[-10:])) if reader else []
        except FileNotFoundError:
            total_queries = 0
            answered_count = 0
            not_answered_count = 0
            accuracy_rate = 0
            recent_queries = []

        # Get database counts
        conn = sqlite3.connect(MEMORY_DB)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM conversations")
        total_conversations = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM sessions")
        total_sessions = cursor.fetchone()[0]
        conn.close()

        # Render dashboard HTML
        return HTMLResponse(content=get_updated_dashboard_html(
            total_queries, answered_count, not_answered_count, accuracy_rate,
            total_conversations, total_sessions, total_custom_info, recent_queries
        ))

    @app.get("/admin/custom-info", response_class=HTMLResponse)
    async def admin_custom_info(credentials: HTTPBasicCredentials = Depends(verify_admin)):
        # Admin page for managing custom information
        return HTMLResponse(content=get_custom_info_html(memory.custom_info))

    @app.get("/admin/custom-info/add", response_class=HTMLResponse)
    async def admin_add_info_form(
        credentials: HTTPBasicCredentials = Depends(verify_admin),
        prefill_topic: str = None
    ):
        # Form to add custom info, optionally pre-filling a topic
        prefilled_topic = unquote(prefill_topic) if prefill_topic else ""
        return HTMLResponse(content=get_add_custom_info_form_html(prefilled_topic))

    @app.post("/admin/custom-info/add")
    async def admin_add_info(
        credentials: HTTPBasicCredentials = Depends(verify_admin),
        topic: str = Form(...),
        information: str = Form(...),
    ):
        # Handle submission of new custom info
        memory.add_custom_info(topic, information)
        return RedirectResponse(url="/admin/custom-info", status_code=303)

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

    @app.get("/admin/faq", response_class=HTMLResponse)
    async def admin_faq(credentials: HTTPBasicCredentials = Depends(verify_admin)):
        # Full FAQ page with all frequently asked questions

        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                reader = list(csv.DictReader(f))

            # Count frequency of ALL questions
            query_counter = Counter(
                (r.get("query_text") or "").strip().lower() for r in reader if r.get("query_text")
            )
        
            # Get questions asked at least twice, sorted by frequency
            frequent_questions = [(q, count) for q, count in query_counter.most_common() if count >= 2]

            # Build FAQ data for each frequent question
            faq_list = []
            for question, count in frequent_questions:
                # Get all log entries for this exact question
                question_entries = [r for r in reader if (r.get("query_text") or "").strip().lower() == question]
            
                # Count successful answers
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

        except FileNotFoundError:
            faq_list = []

        return HTMLResponse(content=get_full_faq_html(faq_list))
    
    @app.get("/admin/manage-queries", response_class=HTMLResponse)
    async def admin_manage_queries(credentials: HTTPBasicCredentials = Depends(verify_admin)):
        """Manage Queries page - shows unresolved queries"""

        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                reader = list(csv.DictReader(f))

            # Get not-answered queries that aren't resolved
            not_answered_queries = []
            for record in reader:
                answered_field = (record.get("answered") or "").strip().lower()
                query_text = (record.get("query_text") or "").strip()
                resolved_date = (record.get("resolved_date") or "").strip()
            
                # Include if: not answered, not resolved, and has query text
                if (answered_field not in ["true", "1", "yes"] and 
                    query_text and 
                    not resolved_date):
                    not_answered_queries.append(record)
        
            # Count frequency
            not_answered_counter = Counter()
            for query in not_answered_queries:
                query_text = (query.get("query_text") or "").strip().lower()
                if query_text:
                    not_answered_counter[query_text] += 1
        
            all_needing_attention = not_answered_counter.most_common()
        
        except FileNotFoundError:
            all_needing_attention = []
        except Exception as e:
            print(f"Error in manage queries: {e}")
            all_needing_attention = []

        return HTMLResponse(content=get_manage_queries_with_resolved_html(all_needing_attention))
    
    @app.get("/admin/feedback", response_class=HTMLResponse)
    async def admin_feedback(credentials: HTTPBasicCredentials = Depends(verify_admin)):
        """Admin page to view user feedback"""
    
        try:
            with open("data/feedback.csv", "r", encoding="utf-8") as f:
                reader = list(csv.DictReader(f))
        
            # Sort by timestamp (newest first)
            feedback_list = sorted(reader, key=lambda x: x.get('timestamp', ''), reverse=True)
        
            # Calculate stats
            total_feedback = len(feedback_list)
            if total_feedback > 0:
                ratings = [int(f.get('rating', 0)) for f in feedback_list if f.get('rating', '').isdigit()]
                avg_rating = sum(ratings) / len(ratings) if ratings else 0
                rating_distribution = Counter(ratings)
            else:
                avg_rating = 0
                rating_distribution = Counter()
            
        except FileNotFoundError:
            feedback_list = []
            total_feedback = 0
            avg_rating = 0
            rating_distribution = Counter()
    
        return HTMLResponse(content=get_feedback_html(feedback_list, total_feedback, avg_rating, rating_distribution))
    
    @app.post("/admin/mark-resolved")
    async def mark_query_resolved(
        request: Request,
        credentials: HTTPBasicCredentials = Depends(verify_admin),
        question: str = Form(...)
    ):
        """Mark all instances of a question as resolved by updating CSV"""
        try:
            # Read current CSV
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                reader = list(csv.DictReader(f))
        
            # Add resolved_date column if it doesn't exist
            fieldnames = reader[0].keys() if reader else []
            if 'resolved_date' not in fieldnames:
                fieldnames = list(fieldnames) + ['resolved_date']
        
            # Update all instances of this question
            today = datetime.now().strftime("%Y-%m-%d")
            updated_count = 0
        
            for record in reader:
                if record.get("query_text", "").strip().lower() == question.lower():
                    if not record.get("resolved_date", "").strip():  # Only update if not already resolved
                        record["resolved_date"] = today
                        updated_count += 1
                    elif "resolved_date" not in record:  # Handle old CSV format
                        record["resolved_date"] = today
                        updated_count += 1
        
            # Write updated CSV with resolved_date column
            if reader:
                with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(reader)
        
            print(f"Marked {updated_count} instances of '{question}' as resolved")
            return RedirectResponse(url="/admin/manage-queries", status_code=303)
        
        except Exception as e:
            print(f"Error marking query as resolved: {e}")
            return RedirectResponse(url="/admin/manage-queries", status_code=303)

    @app.get("/admin/export-data")
    async def admin_export_data(credentials: HTTPBasicCredentials = Depends(verify_admin)):
        # Export chatbot system data
        export_data = {
            "custom_info": memory.custom_info,
            "export_timestamp": datetime.now().isoformat()
        }

        return {
            "data": export_data,
            "filename": f"gsu_chatbot_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        }

# HTML Templates
def get_base_style():
    """Clean, modern CSS with proper responsive design"""
    return """
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

        * { box-sizing: border-box; }

        body {
            font-family: Arial, sans-serif;
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

        .card {
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        /* Stats Grid */
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

        /* Tables */
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

        /* Navigation */
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

def get_updated_dashboard_html(total_queries, answered_count, not_answered_count, accuracy_rate, total_conversations, total_sessions, total_custom_info, recent_queries):
    """Generate HTML for the updated Admin Dashboard with clean responsive design"""

    recent_queries_html = ""
    for query in recent_queries:
        timestamp = query.get('timestamp', '')[:19].replace('T', ' ')
        query_text = query.get('query_text', '')
        query_display = query_text[:80] + ('...' if len(query_text) > 80 else '')
        
        answered_field = (query.get('answered') or '').strip().lower()
        if answered_field in ["true", "1", "yes"]:
            status_icon = '✅'
        else:
            status_icon = '❌'
            
        recent_queries_html += f"""
            <tr>
                <td>{timestamp}</td>
                <td title="{query_text}">{query_display}</td>
                <td style="text-align: center;">{status_icon}</td>
            </tr>"""
    
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>GSU Chatbot Admin Panel</title>
        <style>{get_base_style()}</style>
    </head>
    <body>
        <div class="container">
            {get_nav_html()}
            
            <h1>GSU Chatbot Admin Dashboard</h1>
            
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
                <h2>Export Data</h2>
                <a href="/admin/export-data" class="btn">Export System Data</a>
                <p style="color: var(--muted); margin-top: 10px;">Download all custom information as JSON backup.</p>
            </div>
            
            <div class="card">
                <h2>Recent Queries</h2>
                <p style="color: var(--muted); margin-bottom: 15px;">✅ = Successfully Answered | ❌ = Not Answered</p>
                <div class="table-container">
                    <table>
                        <tr>
                            <th>Time</th>
                            <th>Query</th>
                            <th>Status</th>
                        </tr>
                        {recent_queries_html if recent_queries_html else '<tr><td colspan="3" style="text-align: center; color: var(--muted);">No recent queries</td></tr>'}
                    </table>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

def get_custom_info_html(custom_info):
    """Generate HTML for Custom Information management page with responsive design"""

    sorted_info = sorted(custom_info.items(), key=lambda x: x[1]['added_at'], reverse=True)

    info_html = ""
    for info_id, info in sorted_info:
        topic = info['topic']
        information = info['information'][:100] + ('...' if len(info['information']) > 100 else '')
        added = info['added_at'][:10]
        
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
    </head>
    <body>
        <div class="container">
            {get_nav_html()}
            
            <h1>Manage Information</h1>
            
            <div class="card">
                <h2>Quick Actions</h2>
                <a href="/admin/custom-info/add" class="btn">Add New Information</a>
                <p style="color: var(--muted); margin-top: 10px;">Add new information to improve chatbot responses.</p>
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

def get_full_faq_html(faq_list):
    """Full FAQ page HTML with responsive design and rank numbers"""



    faq_html = ""
    for index, faq in enumerate(faq_list):
        rank = index + 1  # Rank starts from 1
        question = faq['question']
        count = faq['total_asked']
        success_rate = faq['success_rate']
        
        # Determine status color based on success rate
        if success_rate >= 80:
            status_color = "#4caf50"  # Green for high success
            status_icon = "✅"
        elif success_rate >= 50:
            status_color = "#ff9800"  # Orange for medium success
            status_icon = "⚠️"
        else:
            status_color = "#f44336"  # Red for low success
            status_icon = "❌"
        
        # Format question for display (capitalize first letter)
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

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Frequently Asked Questions</title>
        <style>{get_base_style()}</style>
    </head>
    <body>
        <div class="container">
            {get_nav_html()}
            
            <h1>Frequently Asked Questions</h1>
            
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
                        <h3 style="margin-bottom: 10px;">Improvement Actions:</h3>
                        <p style="color: var(--muted); margin: 0;">
                            Questions with low success rates show an "Add Info" button to help improve chatbot responses.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

def get_manage_queries_with_resolved_html(all_needing_attention):
    """Manage Queries HTML with clean responsive design"""
    
    attention_html = ""
    for query, count in all_needing_attention:
        query_encoded = quote(query)
        query_display = query[:80] + ('...' if len(query) > 80 else '')
        
        attention_html += f"""
            <tr>
                <td title="{query}">{query_display}</td>
                <td>{count}</td>
                <td>
                    <a href="/admin/custom-info/add?prefill_topic={query_encoded}" 
                       class="btn btn-success">Add Info</a>
                    <form method="post" action="/admin/mark-resolved" style="display: inline;">
                        <input type="hidden" name="question" value="{query}">
                        <button type="submit" class="btn btn-secondary"
                                onclick="return confirm('Mark this question as resolved?')">
                            Resolved
                        </button>
                    </form>
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
    </head>
    <body>
        <div class="container">
            {get_nav_html()}
            
            <h1>Manage Queries</h1>
            
            <div class="card">
                <h2>Unresolved Queries</h2>
                <p>Queries that have not been successfully answered and have not been marked as resolved.</p>
                
                <div class="table-container">
                    <table>
                        <tr>
                            <th>Question</th>
                            <th>Times Asked</th>
                            <th>Actions</th>
                        </tr>
                        {attention_html if attention_html else '<tr><td colspan="3" style="text-align: center; color: var(--muted);">No unresolved queries - excellent job!</td></tr>'}
                    </table>
                </div>
                
                <div style="margin-top: 20px; padding: 15px; background: var(--light-gray); border-radius: 4px;">
                    <p style="margin: 0; color: var(--muted);">
                        <strong>Actions:</strong> Use "Add Info" to provide the missing information, or "Resolved" to mark as administratively handled.
                    </p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

def get_feedback_html(feedback_list, total_feedback, avg_rating, rating_distribution):
    """Feedback management HTML"""
    
    feedback_rows = ""
    for feedback in feedback_list:
        timestamp = feedback.get('timestamp', '')[:19].replace('T', ' ')
        feedback_text = feedback.get('feedback_text', '')
        rating = feedback.get('rating', 0)
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
                <span style="width: 40px; text-align: right;">{count}</span>
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
                <h2>Feedback Overview</h2>
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
                        <h3>{rating_distribution.get(1, 0) + rating_distribution.get(2, 0)}</h3>
                        <p>Low Ratings (1-2★)</p>
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