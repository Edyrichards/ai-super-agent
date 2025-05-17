#!/bin/bash

# Setup script for AI Super Agent

# Check if Python is installed
if command -v python3 &>/dev/null; then
    echo "Python is installed"
else
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Check Python version
python_version=$(python3 --version | sed 's/Python //')
required_version="3.10.0"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python version must be 3.10.0 or higher. Current version: $python_version"
    exit 1
fi

# Check if Docker is installed (optional)
if command -v docker &>/dev/null; then
    echo "Docker is installed"
    docker_available=true
else
    echo "Warning: Docker is not installed. You can still run the application locally."
    docker_available=false
fi

# Setup virtual environment
echo "Setting up virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Prompt for API keys
read -p "Enter your OpenAI API key: " openai_key
read -p "Enter your Serper API key (optional, press Enter to skip): " serper_key

# Create .env file
echo "Creating .env file..."
echo "OPENAI_API_KEY=$openai_key" > .env
if [ ! -z "$serper_key" ]; then
    echo "SERPER_API_KEY=$serper_key" >> .env
fi

# Create directories
echo "Creating necessary directories..."
mkdir -p research_db

# Setup completed
echo "Setup completed successfully!"
echo ""
echo "To run the application locally:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Start the API server: uvicorn app:app --reload"
echo ""

# Offer to start Docker if available
if [ "$docker_available" = true ]; then
    read -p "Would you like to build and run the Docker container now? (y/n): " run_docker
    if [ "$run_docker" = "y" ] || [ "$run_docker" = "Y" ]; then
        echo "Building and starting Docker container..."
        docker-compose up -d
        echo "Docker container is now running!"
        echo "API is available at http://localhost:8000"
        echo "API documentation is available at http://localhost:8000/docs"
    fi
fi

echo ""
echo "Thank you for setting up AI Super Agent!"