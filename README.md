# AI Super Agent

A powerful AI agent system built using open-source tools like LangChain, CrewAI, and ChromaDB. This system combines multiple specialized agents that can collaborate to perform complex tasks like research, analysis, and report generation.

## Features

- **Multi-Agent Collaboration**: Specialized agents that work together on complex tasks
- **Knowledge Storage**: Vector database integration for persistent knowledge
- **Tool Usage**: Ability to search the web, process information, and generate content
- **API Interface**: RESTful API for easy integration with other systems
- **Docker Deployment**: Containerized deployment for easy setup and scaling

## Architecture

The system consists of several core components:

1. **Main Agent**: Handles general queries and coordinates sub-agents
2. **Research Agent**: Specialized in finding information from the web
3. **Analysis Agent**: Processes and organizes information
4. **Report Generator**: Creates comprehensive reports from findings
5. **Vector Database**: Stores and retrieves knowledge using semantic search
6. **API Layer**: Provides a RESTful interface for interacting with the system

## Getting Started

### Prerequisites

- Python 3.10+
- Docker and Docker Compose (for containerized deployment)
- OpenAI API key
- Serper API key (for web search)

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/ai-super-agent.git
   cd ai-super-agent
   ```

2. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   SERPER_API_KEY=your_serper_api_key
   ```

3. Run the system using Docker Compose:
   ```
   docker-compose up -d
   ```

4. The API will be available at `http://localhost:8000`

### Local Development

1. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the FastAPI application:
   ```
   uvicorn app:app --reload
   ```

## API Usage

### Start a Research Task

```
POST /research
{
  "topic": "Advances in quantum computing",
  "description": "Focus on recent breakthroughs in the last 2 years"
}
```

### Check Research Status

```
GET /research/{task_id}
```

### List All Research Tasks

```
GET /research
```

### Delete a Research Task

```
DELETE /research/{task_id}
```

## Customization

You can customize the AI Super Agent by:

1. Adding new specialized agents in `research_agent_example.py`
2. Creating additional tools in the tools section
3. Modifying the API endpoints in `app.py`
4. Adjusting the Docker configuration for different deployment scenarios

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- LangChain for the agent framework
- CrewAI for the multi-agent orchestration
- ChromaDB for the vector database functionality