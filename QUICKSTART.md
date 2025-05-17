# AI Super Agent - Quick Start Guide

This guide will help you get up and running with the AI Super Agent system quickly.

## Prerequisites

Before you begin, make sure you have:

- Python 3.10 or higher
- OpenAI API key 
- Serper API key (optional, for improved web search)
- Docker and Docker Compose (optional, for containerized deployment)

## Option 1: Quick Setup with Script

The easiest way to get started is to use the provided setup script:

```bash
# Make the script executable
chmod +x setup.sh

# Run the setup script
./setup.sh
```

The script will:
1. Check your Python version
2. Create a virtual environment
3. Install all required dependencies
4. Prompt you for API keys
5. Create necessary directories
6. Offer to start the Docker container if Docker is available

## Option 2: Manual Setup

If you prefer to set up the system manually:

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create a .env file with your API keys:**
   ```
   OPENAI_API_KEY=your_openai_api_key
   SERPER_API_KEY=your_serper_api_key  # Optional
   ```

4. **Start the API server:**
   ```bash
   uvicorn app:app --reload
   ```

## Option 3: Docker Deployment

To deploy using Docker:

1. **Make sure Docker and Docker Compose are installed**

2. **Create a .env file with your API keys**

3. **Build and start the containers:**
   ```bash
   docker-compose up -d
   ```

## Using the AI Super Agent

Once the system is running, you can:

1. **Access the API documentation:**
   Open your browser and go to: http://localhost:8000/docs

2. **Use the command-line client:**
   ```bash
   # Start a research task
   python client.py research "Quantum computing advances"
   
   # Check task status
   python client.py status <task_id>
   
   # List all tasks
   python client.py list
   
   # View research results
   python client.py results <task_id>
   ```

3. **Make API requests directly:**
   ```bash
   # Start a research task
   curl -X POST http://localhost:8000/research \
     -H "Content-Type: application/json" \
     -d '{"topic": "Quantum computing advances"}'
   
   # Check task status
   curl http://localhost:8000/research/<task_id>
   ```

## Example Research Topics

Try researching these topics to test the system:

- "Recent advancements in renewable energy"
- "The impact of artificial intelligence on healthcare"
- "Trends in quantum computing research"
- "Climate change mitigation strategies"
- "The future of remote work after COVID-19"

## Troubleshooting

**API Key Issues:**
- Double-check that your API keys are correctly set in the .env file
- Verify you have sufficient credits in your OpenAI account

**Docker Issues:**
- Run `docker-compose logs` to view container logs
- Make sure ports are not already in use

**Python Issues:**
- Verify you're using Python 3.10 or later with `python --version`
- Make sure all dependencies are installed with `pip list`

## Next Steps

Once you're comfortable with the basic setup, you can:

1. Customize the agents by modifying the `research_agent_example.py` file
2. Add new tools and capabilities to enhance the system
3. Integrate the API with your applications
4. Deploy to a production environment using the Kubernetes configuration

For more detailed information, refer to the main [README.md](README.md) file.