#!/usr/bin/env python3
"""
AI Super Agent Client Tool

A simple command-line client for interacting with the AI Super Agent API.
This tool allows you to start research tasks, check their status, and view results.

Usage:
  python client.py research "Quantum computing advances"
  python client.py status <task_id>
  python client.py list
  python client.py results <task_id>
"""

import os
import sys
import json
import time
import argparse
import requests
from typing import Dict, List, Optional
from datetime import datetime

# Default API endpoint
API_URL = os.environ.get("API_URL", "http://localhost:8000")

def research(topic: str, description: Optional[str] = None) -> Dict:
    """Start a new research task."""
    data = {"topic": topic}
    if description:
        data["description"] = description
    
    response = requests.post(f"{API_URL}/research", json=data)
    response.raise_for_status()
    return response.json()

def get_status(task_id: str) -> Dict:
    """Get the status of a research task."""
    response = requests.get(f"{API_URL}/research/{task_id}")
    response.raise_for_status()
    return response.json()

def list_tasks() -> List[Dict]:
    """List all research tasks."""
    response = requests.get(f"{API_URL}/research")
    response.raise_for_status()
    return response.json()

def display_task(task: Dict) -> None:
    """Display task information in a formatted way."""
    print(f"Task ID:     {task['task_id']}")
    print(f"Topic:       {task['topic']}")
    print(f"Status:      {task['status']}")
    print(f"Created:     {task['created_at']}")
    
    if task.get('completed_at'):
        print(f"Completed:   {task['completed_at']}")
    
    print("")

def display_result(task: Dict) -> None:
    """Display task result in a formatted way."""
    display_task(task)
    
    if task.get('result'):
        print("===== RESEARCH RESULTS =====")
        print(task['result'])
        print("============================")
    else:
        print("No results available yet.")

def wait_for_completion(task_id: str, polling_interval: int = 5) -> Dict:
    """Wait for a task to complete."""
    print(f"Waiting for task {task_id} to complete...")
    while True:
        task = get_status(task_id)
        status = task["status"]
        
        if status == "queued":
            print("Task is queued...")
        elif status == "in_progress":
            print("Task is in progress...")
        elif status in ["completed", "failed"]:
            print(f"Task {status}!")
            return task
        
        time.sleep(polling_interval)

def main():
    parser = argparse.ArgumentParser(description="AI Super Agent Client Tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Research command
    research_parser = subparsers.add_parser("research", help="Start a new research task")
    research_parser.add_argument("topic", help="Topic to research")
    research_parser.add_argument("--description", "-d", help="Additional description")
    research_parser.add_argument("--wait", "-w", action="store_true", help="Wait for completion")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Check the status of a task")
    status_parser.add_argument("task_id", help="ID of the task to check")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all tasks")
    
    # Results command
    results_parser = subparsers.add_parser("results", help="Get the results of a task")
    results_parser.add_argument("task_id", help="ID of the task to get results for")
    
    args = parser.parse_args()
    
    if args.command == "research":
        task = research(args.topic, args.description)
        print(f"Research task started with ID: {task['task_id']}")
        
        if args.wait:
            task = wait_for_completion(task['task_id'])
            display_result(task)
        
    elif args.command == "status":
        task = get_status(args.task_id)
        display_task(task)
        
    elif args.command == "list":
        tasks = list_tasks()
        print(f"Found {len(tasks)} tasks:")
        print("")
        
        for task in tasks:
            display_task(task)
            
    elif args.command == "results":
        task = get_status(args.task_id)
        display_result(task)
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()