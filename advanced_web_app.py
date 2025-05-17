"""
Advanced Web Interface for AI Super Agent

This Flask application provides a web interface for the advanced AI agent system,
with real-time updates, better visualization, and more interactive features.
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import time
import threading
import uuid
from datetime import datetime
from dotenv import load_dotenv

# Import the advanced agent system
from advanced_agent import agent_system

# Load environment variables
load_dotenv()

app = Flask(__name__)

# ----------------------
# Routes
# ----------------------

@app.route('/')
def index():
    """Render the home page with form to submit research queries"""
    return render_template('advanced_index.html')


@app.route('/submit', methods=['POST'])
def submit_research():
    """Handle research submission and start background task"""
    topic = request.form.get('topic')
    description = request.form.get('description', '')
    depth = request.form.get('depth', 'standard')
    sources = int(request.form.get('sources', '3'))
    
    if not topic:
        return jsonify({"error": "Topic is required"}), 400
    
    # Create a task
    task_id = agent_system.create_task(
        topic=topic,
        description=description,
        depth=depth,
        sources_required=sources
    )
    
    # Start background task
    thread = threading.Thread(
        target=agent_system.execute_task,
        args=(task_id,)
    )
    thread.daemon = True
    thread.start()
    
    # Redirect to task status page
    return redirect(url_for('view_task', task_id=task_id))


@app.route('/task/<task_id>')
def view_task(task_id):
    """View status and results of a specific task"""
    try:
        task = agent_system.get_task(task_id)
        return render_template('advanced_task.html', task_id=task_id, task=task)
    except ValueError:
        return render_template('error.html', message="Research task not found"), 404


@app.route('/api/task/<task_id>')
def api_task_status(task_id):
    """API endpoint to get task status and results"""
    try:
        task = agent_system.get_task(task_id)
        
        # Format the response
        response = {
            "id": task["id"],
            "topic": task["request"].topic,
            "description": task["request"].description,
            "status": task["status"],
            "created_at": task["created_at"],
            "updated_at": task.get("updated_at"),
            "completed_at": task.get("completed_at"),
            "error": task.get("error")
        }
        
        # If task is completed, include the report
        if task["status"] == "completed" and task.get("result"):
            result = task["result"]
            response["report"] = agent_system.format_report_as_markdown(result)
            
        return jsonify(response)
    
    except ValueError:
        return jsonify({"error": "Research task not found"}), 404


@app.route('/tasks')
def list_tasks():
    """List all research tasks"""
    tasks = agent_system.list_tasks()
    return render_template('advanced_tasks.html', tasks=tasks)


@app.route('/api/tasks')
def api_list_tasks():
    """API endpoint to list all tasks"""
    tasks = agent_system.list_tasks()
    
    # Format the response
    response = []
    for task in tasks:
        response.append({
            "id": task["id"],
            "topic": task["request"].topic,
            "description": task["request"].description,
            "status": task["status"],
            "created_at": task["created_at"],
            "updated_at": task.get("updated_at"),
            "completed_at": task.get("completed_at")
        })
    
    return jsonify(response)


@app.route('/system')
def system_info():
    """View information about the agent system"""
    # Count tasks by status
    tasks = agent_system.list_tasks()
    status_counts = {}
    for task in tasks:
        status = task["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    
    # Calculate completion time statistics
    completed_tasks = [t for t in tasks if t.get("completed_at")]
    avg_completion_time = 0
    if completed_tasks:
        completion_times = []
        for task in completed_tasks:
            start_time = datetime.strptime(task["created_at"], "%Y-%m-%d %H:%M:%S")
            end_time = datetime.strptime(task["completed_at"], "%Y-%m-%d %H:%M:%S")
            completion_time = (end_time - start_time).total_seconds()
            completion_times.append(completion_time)
        
        avg_completion_time = sum(completion_times) / len(completion_times)
    
    # Get agent info
    agents = [
        {
            "name": agent_system.research_agent.name,
            "role": agent_system.research_agent.role,
            "tools": [t.name for t in agent_system.research_agent.tools]
        },
        {
            "name": agent_system.analysis_agent.name,
            "role": agent_system.analysis_agent.role,
            "tools": [t.name for t in agent_system.analysis_agent.tools]
        },
        {
            "name": agent_system.report_agent.name,
            "role": agent_system.report_agent.role,
            "tools": []
        }
    ]
    
    return render_template(
        'system_info.html',
        agents=agents,
        task_count=len(tasks),
        status_counts=status_counts,
        avg_completion_time=avg_completion_time
    )


@app.route('/api/system')
def api_system_info():
    """API endpoint to get system information"""
    # Count tasks by status
    tasks = agent_system.list_tasks()
    status_counts = {}
    for task in tasks:
        status = task["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    
    # Get agent info
    agents = [
        {
            "name": agent_system.research_agent.name,
            "role": agent_system.research_agent.role,
            "tools": [t.name for t in agent_system.research_agent.tools]
        },
        {
            "name": agent_system.analysis_agent.name,
            "role": agent_system.analysis_agent.role,
            "tools": [t.name for t in agent_system.analysis_agent.tools]
        },
        {
            "name": agent_system.report_agent.name,
            "role": agent_system.report_agent.role,
            "tools": []
        }
    ]
    
    return jsonify({
        "agents": agents,
        "task_count": len(tasks),
        "status_counts": status_counts
    })


if __name__ == "__main__":
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    app.run(host='0.0.0.0', port=8001, debug=True)