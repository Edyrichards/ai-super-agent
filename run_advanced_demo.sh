#!/bin/bash

# Run the advanced AI Super Agent demo

# Set text colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print banner
echo -e "${PURPLE}"
echo "============================================================"
echo "                 AI SUPER AGENT DEMO                        "
echo "============================================================"
echo -e "${NC}"

# Check if Python is installed
if command -v python3 &>/dev/null; then
    echo -e "${GREEN}[✓] Python is installed${NC}"
else
    echo -e "${YELLOW}[!] Error: Python 3 is required but not installed.${NC}"
    exit 1
fi

# Install required packages
echo -e "${BLUE}[*] Installing required packages...${NC}"
pip install flask flask-cors markdown-it-py flask_socketio python-dotenv

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo -e "${YELLOW}[!] Creating .env file...${NC}"
    echo "# This is a demo environment file" > .env
    echo "OPENAI_API_KEY=sk-demo-key" >> .env
fi

# Check if templates directory exists
if [ ! -d templates ]; then
    echo -e "${YELLOW}[!] Creating templates directory...${NC}"
    mkdir -p templates
fi

# Make sure demo.py is executable
chmod +x demo.py

# Run the demo
echo -e "${BLUE}[*] Starting AI Super Agent Demo...${NC}"
echo -e "${YELLOW}[!] The demo will open in your web browser. Press Ctrl+C to stop the server when finished.${NC}"
python3 demo.py