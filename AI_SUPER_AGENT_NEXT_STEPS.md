# AI Super Agent: Next Steps Implementation Guide

This comprehensive guide outlines the steps to evolve your current AI Super Agent demonstration into a production-ready, powerful system with real capabilities.

## 1. Implementing Real LLM Integration

### OpenAI Integration
```python
# In research_agent_example.py, replace mock LLM with real OpenAI integration:
from langchain_openai import ChatOpenAI

# Initialize with your API key
llm = ChatOpenAI(
    model="gpt-4o",  # or "gpt-4-turbo", "gpt-3.5-turbo" based on needs
    temperature=0.7,
    openai_api_key=os.environ["OPENAI_API_KEY"]
)
```

### Alternative Providers
You can also implement Anthropic, Google Vertex AI, or open-source models:

```python
# Anthropic Claude
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-3-opus-20240229")

# Open source with Ollama
from langchain_community.llms import Ollama
llm = Ollama(model="llama3")
```

## 2. Implementing Real Tool Integration

### Web Search with Serper API
```python
@tool
def search_web(query: str) -> str:
    """Search the web for information on a specific query."""
    import json
    import requests
    
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query})
    headers = {
        'X-API-KEY': os.environ['SERPER_API_KEY'],
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.text
```

### Real Web Scraping
```python
@tool
def scrape_webpage(url: str) -> str:
    """Scrape content from a webpage."""
    from langchain_community.document_loaders import WebBaseLoader
    
    loader = WebBaseLoader(url)
    docs = loader.load()
    return docs[0].page_content
```

### Vector Database Implementation
```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Initialize real embeddings
embeddings = OpenAIEmbeddings()

# Initialize persistent vector database
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
```

## 3. Enhanced Agent Architecture

### Implementing CrewAI for Multi-Agent Collaboration

```python
from crewai import Agent, Task, Crew, Process

# Define specialized agents
researcher = Agent(
    role="Research Specialist",
    goal="Find comprehensive and accurate information on any topic",
    backstory="You are an expert researcher with vast knowledge and exceptional search skills",
    verbose=True,
    llm=llm,
    tools=[search_web, scrape_webpage]
)

analyst = Agent(
    role="Data Analyst",
    goal="Analyze information and extract key insights",
    backstory="You are a skilled analyst who can interpret complex data and identify patterns",
    verbose=True,
    llm=llm,
    tools=[retrieve_information]
)

writer = Agent(
    role="Content Writer",
    goal="Create comprehensive, well-structured reports",
    backstory="You are an experienced writer who can communicate complex ideas clearly",
    verbose=True,
    llm=llm,
    tools=[generate_report]
)

# Define tasks for each agent
research_task = Task(
    description="Research {topic} thoroughly, finding latest information and key resources",
    agent=researcher,
    expected_output="Comprehensive research notes on {topic}"
)

analysis_task = Task(
    description="Analyze the research findings on {topic}, identifying key patterns and insights",
    agent=analyst,
    expected_output="Analytical summary with key insights about {topic}"
)

writing_task = Task(
    description="Create a well-structured report on {topic} based on the research and analysis",
    agent=writer,
    expected_output="Complete research report on {topic}"
)

# Create a crew with the agents and tasks
crew = Crew(
    agents=[researcher, analyst, writer],
    tasks=[research_task, analysis_task, writing_task],
    verbose=2,
    process=Process.sequential  # Tasks are executed in sequence
)

# Function to run the crew
def create_research_crew(topic: str):
    # Format the task descriptions with the specific topic
    for task in crew.tasks:
        task.description = task.description.replace("{topic}", topic)
        task.expected_output = task.expected_output.replace("{topic}", topic)
    
    # Execute the crew
    result = crew.kickoff()
    return result
```

## 4. Enhanced Web Interface

### Implementing Real-Time Updates with WebSockets

```python
# In app.py, add WebSocket support
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

app = FastAPI()

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, task_id: str):
        await websocket.accept()
        if task_id not in self.active_connections:
            self.active_connections[task_id] = []
        self.active_connections[task_id].append(websocket)

    def disconnect(self, websocket: WebSocket, task_id: str):
        self.active_connections[task_id].remove(websocket)

    async def broadcast_to_task(self, task_id: str, message: str):
        if task_id in self.active_connections:
            for connection in self.active_connections[task_id]:
                await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    await manager.connect(websocket, task_id)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, task_id)

# And in process_task function, add:
await manager.broadcast_to_task(
    task_id, 
    json.dumps({"type": "update", "data": task_history[task_id][-1]})
)
```

### Add Client-Side WebSocket Code in task.html
```javascript
// In the script section of task.html
const socket = new WebSocket(`ws://${window.location.host}/ws/{{ task_id }}`);

socket.onmessage = function(event) {
    const data = JSON.parse(event.data);
    if (data.type === "update") {
        // Add the new history item to the page
        addHistoryItem(data.data);
    }
};

function addHistoryItem(item) {
    // Create and append a new history item to the history container
    const historyContainer = document.getElementById('history-container');
    const itemElement = document.createElement('div');
    itemElement.className = `history-item ${item.type}`;
    // ... format the item based on its type ...
    historyContainer.appendChild(itemElement);
}
```

## 5. Database Integration for Persistent Storage

### Using PostgreSQL for Task Storage

```python
# Install required packages
# pip install sqlalchemy psycopg2-binary alembic

# In database.py
from sqlalchemy import create_engine, Column, String, JSON, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
import os

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://username:password@localhost/ai_super_agent")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Task(Base):
    __tablename__ = "tasks"
    
    task_id = Column(String, primary_key=True)
    topic = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    status = Column(String, nullable=False)
    result = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
class TaskHistory(Base):
    __tablename__ = "task_history"
    
    id = Column(String, primary_key=True)
    task_id = Column(String, nullable=False)
    type = Column(String, nullable=False)
    content = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)
```

### Update app.py to use the database

```python
# In app.py
from database import SessionLocal, Task, TaskHistory
import uuid

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/tasks/", response_model=TaskResponse)
async def create_task(task_request: TaskRequest, db: Session = Depends(get_db)):
    task_id = str(uuid.uuid4())
    
    db_task = Task(
        task_id=task_id,
        topic=task_request.task,
        description=task_request.description,
        status="in_progress",
        result=None
    )
    db.add(db_task)
    db.commit()
    
    # Process the task asynchronously
    asyncio.create_task(process_task(task_id, task_request.task, task_request.tools, db))
    
    return TaskResponse(
        task_id=task_id,
        status="in_progress",
        result=None
    )
```

## 6. Deploying to Production

### Docker Compose Production Setup

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile.prod
    restart: always
    ports:
      - "8002:8002"
    depends_on:
      - db
      - redis
    environment:
      - DATABASE_URL=postgresql://aiagent:${POSTGRES_PASSWORD}@db/ai_super_agent
      - REDIS_URL=redis://redis:6379/0
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - SERPER_API_KEY=${SERPER_API_KEY}
    volumes:
      - ./data:/app/data
  
  db:
    image: postgres:14
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_USER=aiagent
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - POSTGRES_DB=ai_super_agent
    restart: always

  redis:
    image: redis:7
    volumes:
      - redis_data:/data
    restart: always

  nginx:
    image: nginx:1.23
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/conf:/etc/nginx/conf.d
      - ./nginx/certbot/conf:/etc/letsencrypt
      - ./nginx/certbot/www:/var/www/certbot
    depends_on:
      - app
    restart: always

volumes:
  postgres_data:
  redis_data:
```

### Production Dockerfile

```Dockerfile
# Dockerfile.prod
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/chroma_db

# Run as non-root user
RUN useradd -m appuser
USER appuser

# Start the application with Gunicorn
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8002", "app:app", "--workers", "4"]
```

### Nginx Configuration

```nginx
# nginx/conf/app.conf
server {
    listen 80;
    server_name yourdomain.com;

    location /.well-known/acme-challenge/ {
        root /var/www/certbot;
    }

    location / {
        return 301 https://$host$request_uri;
    }
}

server {
    listen 443 ssl;
    server_name yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    location / {
        proxy_pass http://app:8002;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header Host $host;
        proxy_redirect off;
    }

    location /ws/ {
        proxy_pass http://app:8002;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## 7. Adding Authentication and Security

### Implementing OAuth with Auth0

```python
# Install dependencies
# pip install fastapi-auth0

# In app.py
from fastapi_auth0 import Auth0, Auth0User

auth = Auth0(domain="YOUR_AUTH0_DOMAIN", api_audience="YOUR_API_AUDIENCE", scopes={"read:tasks", "write:tasks"})

@app.post("/tasks/", response_model=TaskResponse)
async def create_task(
    task_request: TaskRequest, 
    db: Session = Depends(get_db),
    user: Auth0User = Depends(auth.get_user)
):
    # Now you have the authenticated user
    user_id = user.id
    
    task_id = str(uuid.uuid4())
    
    db_task = Task(
        task_id=task_id,
        topic=task_request.task,
        description=task_request.description,
        status="in_progress",
        result=None,
        user_id=user_id
    )
    db.add(db_task)
    db.commit()
    
    # Process the task asynchronously
    asyncio.create_task(process_task(task_id, task_request.task, task_request.tools, db))
    
    return TaskResponse(
        task_id=task_id,
        status="in_progress",
        result=None
    )

# Add user property to Task model
class Task(Base):
    __tablename__ = "tasks"
    
    task_id = Column(String, primary_key=True)
    topic = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    status = Column(String, nullable=False)
    result = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    user_id = Column(String, nullable=False)
```

## 8. Implementing Additional Capabilities

### Adding File Processing Capabilities

```python
@app.post("/upload/")
async def upload_file(
    file: UploadFile,
    db: Session = Depends(get_db),
    user: Auth0User = Depends(auth.get_user)
):
    file_path = f"uploads/{user.id}/{file.filename}"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    # Process the file based on type
    if file.filename.endswith(".pdf"):
        # Process PDF
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Store in database for the user
        # ...
    
    elif file.filename.endswith((".csv", ".xlsx")):
        # Process spreadsheet
        import pandas as pd
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        # Process and store data
        # ...
    
    return {"filename": file.filename, "status": "processed"}
```

### Adding Custom Tools for Your Domain

```python
# Example: Adding a tool for generating images with DALL-E
@tool
def generate_image(prompt: str, size: str = "1024x1024") -> str:
    """Generate an image based on a text prompt using DALL-E."""
    import requests
    
    api_key = os.environ["OPENAI_API_KEY"]
    
    url = "https://api.openai.com/v1/images/generations"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "prompt": prompt,
        "n": 1,
        "size": size
    }
    
    response = requests.post(url, headers=headers, json=data)
    result = response.json()
    
    if "data" in result and len(result["data"]) > 0:
        image_url = result["data"][0]["url"]
        
        # Download and save the image
        img_response = requests.get(image_url)
        img_filename = f"images/{uuid.uuid4()}.png"
        os.makedirs(os.path.dirname(img_filename), exist_ok=True)
        
        with open(img_filename, "wb") as f:
            f.write(img_response.content)
        
        return img_filename
    else:
        return "Failed to generate image"
```

## 9. Implementing Analytics and Monitoring

### Integrating Prometheus and Grafana for Monitoring

```python
# Install dependencies
# pip install prometheus-fastapi-instrumentator

# In app.py
from prometheus_fastapi_instrumentator import Instrumentator

# Add Prometheus metrics
@app.on_event("startup")
async def startup():
    Instrumentator().instrument(app).expose(app)
```

### Docker Compose Addition for Monitoring

```yaml
# Add to docker-compose.prod.yml
services:
  # ... existing services ...
  
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus:/etc/prometheus
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
    ports:
      - "9090:9090"
    restart: always

  grafana:
    image: grafana/grafana
    volumes:
      - grafana_data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    ports:
      - "3000:3000"
    depends_on:
      - prometheus
    restart: always

volumes:
  # ... existing volumes ...
  prometheus_data:
  grafana_data:
```

## 10. Implementing API Throttling and Rate Limiting

```python
# Install dependencies
# pip install slowapi

# In app.py
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/tasks/", response_model=TaskResponse)
@limiter.limit("5/minute")  # Limit to 5 task creations per minute per IP
async def create_task(
    task_request: TaskRequest, 
    db: Session = Depends(get_db),
    user: Auth0User = Depends(auth.get_user)
):
    # ... existing implementation ...
```

## Conclusion

By following this guide, you'll be able to transform your current AI Super Agent demo into a production-ready system with:

1. Real LLM integration with providers like OpenAI, Anthropic, or open-source models
2. Actual tools for web search, web scraping, and vector database storage
3. A multi-agent architecture using CrewAI for complex task processing
4. Enhanced web interface with real-time updates using WebSockets
5. Persistent storage using PostgreSQL
6. Production deployment using Docker, Nginx, and proper security practices
7. Authentication and user management with Auth0
8. Additional capabilities for file processing and custom domain-specific tools
9. Analytics and monitoring with Prometheus and Grafana
10. API throttling and rate limiting for better resource management

Each section provides concrete code examples that you can adapt to your specific use case.