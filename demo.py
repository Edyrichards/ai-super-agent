#!/usr/bin/env python3
"""
Demo Script for AI Super Agent

This script demonstrates the advanced AI agent system by setting up
some sample research tasks and running the web interface.
"""

from advanced_agent import agent_system
import threading
import time
import sys
import os
import webbrowser

def print_colored(text, color):
    """Print colored text to the console."""
    colors = {
        'green': '\033[92m',
        'blue': '\033[94m',
        'yellow': '\033[93m',
        'red': '\033[91m',
        'purple': '\033[95m',
        'cyan': '\033[96m',
        'end': '\033[0m'
    }
    print(f"{colors.get(color, '')}{text}{colors['end']}")

def create_sample_tasks():
    """Create some sample research tasks to demonstrate the system."""
    print_colored("\n[*] Creating sample research tasks...", "blue")
    
    # Create sample tasks with different statuses
    task1_id = agent_system.create_task(
        topic="Artificial Intelligence in Healthcare",
        description="Focus on recent applications and future trends",
        depth="standard",
        sources_required=3
    )
    print_colored(f"[+] Created task: {task1_id} - Artificial Intelligence in Healthcare", "green")
    
    task2_id = agent_system.create_task(
        topic="Renewable Energy Technologies",
        description="Compare solar, wind, and hydroelectric power",
        depth="deep",
        sources_required=5
    )
    print_colored(f"[+] Created task: {task2_id} - Renewable Energy Technologies", "green")
    
    task3_id = agent_system.create_task(
        topic="Machine Learning for Beginners",
        description="Basic concepts and practical applications",
        depth="basic",
        sources_required=3
    )
    print_colored(f"[+] Created task: {task3_id} - Machine Learning for Beginners", "green")
    
    # Execute tasks in background threads
    def execute_task(task_id):
        print_colored(f"[*] Executing task: {task_id}...", "blue")
        result = agent_system.execute_task(task_id)
        print_colored(f"[+] Task {task_id} completed with status: {result}", "green")
    
    # Start executing tasks in threads
    threading.Thread(target=execute_task, args=(task1_id,)).start()
    
    # Wait a bit before starting the next task
    time.sleep(2)
    threading.Thread(target=execute_task, args=(task2_id,)).start()
    
    # Wait a bit before starting the next task
    time.sleep(2)
    threading.Thread(target=execute_task, args=(task3_id,)).start()
    
    return [task1_id, task2_id, task3_id]

def start_web_server(port=8001):
    """Start the Flask web server."""
    print_colored("\n[*] Starting web server...", "blue")
    
    # Check if the templates directory exists
    if not os.path.exists('templates'):
        print_colored("[!] Templates directory not found. Creating it...", "yellow")
        os.makedirs('templates', exist_ok=True)
    
    # Import the Flask app
    from advanced_web_app import app
    
    # Open the web browser
    url = f"http://localhost:{port}"
    threading.Thread(target=lambda: webbrowser.open(url)).start()
    
    print_colored(f"[+] Web server started at {url}", "green")
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=port, debug=False)

if __name__ == "__main__":
    print_colored("\n" + "=" * 50, "purple")
    print_colored("       AI SUPER AGENT DEMONSTRATION", "purple")
    print_colored("=" * 50, "purple")
    print_colored("\nThis demo will showcase the advanced AI agent system with a web interface.", "cyan")
    print_colored("It will create sample research tasks and display them in the web interface.", "cyan")
    
    # Create sample tasks
    task_ids = create_sample_tasks()
    
    # Start the web server
    start_web_server()