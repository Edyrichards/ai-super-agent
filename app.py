"""
FastAPI application for serving the AI Super Agent.

This API provides endpoints to:
1. Process general queries using the main agent
2. Start research projects using the specialized research crew
3. Retrieve the status and results of research tasks
"""

import os
import uuid
import asyncio
from typing import Dict, List, Optional, Union
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import agent code (assuming research_agent_example.py is in the same directory)
from research_agent_example import create_research_crew

# Create FastAPI app
app = FastAPI(
    title="AI Super Agent API",
    description="API for interacting with an AI Super Agent built with LangChain and CrewAI",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define data models
class ResearchRequest(BaseModel):
    topic: str
    description: Optional[str] = None

class ResearchResponse(BaseModel):
    task_id: str
    topic: str
    status: str
    created_at: str

class ResearchResult(BaseModel):
    task_id: str
    topic: str
    status: str
    created_at: str
    completed_at: Optional[str] = None
    result: Optional[str] = None

# Store ongoing and completed research tasks
research_tasks: Dict[str, Dict] = {}

# Function to run research task in the background
async def run_research_task(task_id: str, topic: str, description: Optional[str] = None):
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


@app.post("/research", response_model=ResearchResponse, tags=["Research"])
async def start_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    """
    Start a new research task.
    
    This endpoint initiates a background task to research the provided topic and generate a report.
    """
    task_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    # Create task record
    research_tasks[task_id] = {
        "task_id": task_id,
        "topic": request.topic,
        "description": request.description,
        "status": "queued",
        "created_at": timestamp
    }
    
    # Start background task
    background_tasks.add_task(run_research_task, task_id, request.topic, request.description)
    
    return ResearchResponse(
        task_id=task_id,
        topic=request.topic,
        status="queued",
        created_at=timestamp
    )


@app.get("/research/{task_id}", response_model=ResearchResult, tags=["Research"])
async def get_research_status(task_id: str):
    """
    Get the status and results of a research task.
    
    If the task is completed, this will include the full research report.
    """
    if task_id not in research_tasks:
        raise HTTPException(status_code=404, detail="Research task not found")
    
    task = research_tasks[task_id]
    
    return ResearchResult(
        task_id=task["task_id"],
        topic=task["topic"],
        status=task["status"],
        created_at=task["created_at"],
        completed_at=task.get("completed_at"),
        result=task.get("result")
    )


@app.get("/research", response_model=List[ResearchResponse], tags=["Research"])
async def list_research_tasks():
    """
    List all research tasks.
    
    Returns a list of all research tasks with their current status.
    """
    return [
        ResearchResponse(
            task_id=task["task_id"],
            topic=task["topic"],
            status=task["status"],
            created_at=task["created_at"]
        )
        for task in research_tasks.values()
    ]


@app.delete("/research/{task_id}", tags=["Research"])
async def delete_research_task(task_id: str):
    """
    Delete a research task.
    
    This will remove the task and its results from memory.
    """
    if task_id not in research_tasks:
        raise HTTPException(status_code=404, detail="Research task not found")
    
    del research_tasks[task_id]
    
    return {"message": "Research task deleted successfully"}


@app.get("/health", tags=["System"])
async def health_check():
    """
    Check if the API is running.
    
    Returns basic system information.
    """
    return {
        "status": "healthy",
        "version": app.version,
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)