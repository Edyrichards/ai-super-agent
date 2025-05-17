"""
Web interface for the AI Super Agent.

This Flask application provides a web interface for interacting with the 
AI Super Agent system, allowing users to:
1. Submit research queries
2. View research progress
3. See results in real-time
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import uuid
import threading
import time
from datetime import datetime
from dotenv import load_dotenv

# Import the research crew functionality
from research_agent_example import create_research_crew

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Store ongoing and completed research tasks
research_tasks = {}

# Function to run research task in the background
def run_research_task(task_id, topic, description=None):
    try:
        # Update task status
        research_tasks[task_id]["status"] = "in_progress"
        
        # Create and run the research crew
        crew = create_research_crew(topic)
        result = crew.kickoff()
        
        # Update task with results
        research_tasks[task_id]["status"] = "completed"
        research_tasks[task_id]["completed_at"] = datetime.now().isoformat()
        research_tasks[task_id]["result"] = result
        
    except Exception as e:
        # Update task with error
        research_tasks[task_id]["status"] = "failed"
        research_tasks[task_id]["completed_at"] = datetime.now().isoformat()
        research_tasks[task_id]["error"] = str(e)


@app.route('/')
def index():
    """Render the home page with form to submit research queries"""
    return render_template('index.html')


@app.route('/submit', methods=['POST'])
def submit_research():
    """Handle research submission and start background task"""
    topic = request.form.get('topic')
    description = request.form.get('description', '')
    
    if not topic:
        return jsonify({"error": "Topic is required"}), 400
    
    # Create a unique task ID
    task_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    # Create task record
    research_tasks[task_id] = {
        "task_id": task_id,
        "topic": topic,
        "description": description,
        "status": "queued",
        "created_at": timestamp
    }
    
    # Start background task
    thread = threading.Thread(
        target=run_research_task,
        args=(task_id, topic, description)
    )
    thread.daemon = True
    thread.start()
    
    # Redirect to task status page
    return redirect(url_for('view_task', task_id=task_id))


@app.route('/task/<task_id>')
def view_task(task_id):
    """View status and results of a specific task"""
    if task_id not in research_tasks:
        return render_template('error.html', message="Research task not found"), 404
    
    return render_template('task.html', task_id=task_id)


@app.route('/api/task/<task_id>')
def api_task_status(task_id):
    """API endpoint to get task status and results"""
    if task_id not in research_tasks:
        return jsonify({"error": "Research task not found"}), 404
    
    return jsonify(research_tasks[task_id])


@app.route('/tasks')
def list_tasks():
    """List all research tasks"""
    return render_template('tasks.html', tasks=research_tasks)


if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    app.run(host='0.0.0.0', port=8002, debug=True)