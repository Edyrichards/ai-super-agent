#!/bin/bash

# Run the web interface for AI Super Agent

# Check if Python is installed
if command -v python3 &>/dev/null; then
    echo "Python is installed"
else
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Check if Docker is installed
if command -v docker &>/dev/null && command -v docker-compose &>/dev/null; then
    echo "Docker and Docker Compose are installed"
    
    # Check if .env file exists
    if [ ! -f .env ]; then
        echo "Creating .env file..."
        read -p "Enter your OpenAI API key: " openai_key
        read -p "Enter your Serper API key (optional, press Enter to skip): " serper_key
        
        echo "OPENAI_API_KEY=$openai_key" > .env
        if [ ! -z "$serper_key" ]; then
            echo "SERPER_API_KEY=$serper_key" >> .env
        fi
    fi
    
    echo "Starting AI Super Agent web interface with Docker..."
    docker-compose -f docker-compose.web.yml up -d
    
    echo ""
    echo "Web interface is now running!"
    echo "Open your browser and go to: http://localhost:8080"
    echo ""
    echo "To stop the service, run: docker-compose -f docker-compose.web.yml down"
    
else
    # Setup virtual environment if Docker is not available
    echo "Docker not found. Setting up local environment..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install dependencies
    echo "Installing dependencies..."
    pip install -r requirements.txt
    pip install flask
    
    # Check if .env file exists
    if [ ! -f .env ]; then
        echo "Creating .env file..."
        read -p "Enter your OpenAI API key: " openai_key
        read -p "Enter your Serper API key (optional, press Enter to skip): " serper_key
        
        echo "OPENAI_API_KEY=$openai_key" > .env
        if [ ! -z "$serper_key" ]; then
            echo "SERPER_API_KEY=$serper_key" >> .env
        fi
    fi
    
    # Create necessary directories
    mkdir -p research_db
    mkdir -p templates
    
    # Run the web app
    echo "Starting AI Super Agent web interface..."
    python web_app.py
fi