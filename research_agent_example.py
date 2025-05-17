"""
Research Agent Example - A specialized AI agent for comprehensive research tasks.

This example shows how to build a research-focused AI agent that can:
1. Search for information on the web
2. Extract key information from articles
3. Store and retrieve information from a vector database
4. Generate comprehensive research reports

Requirements:
- OpenAI API key
- Serper API key (for web search)
import time
- langchain, crewai, chromadb packages
"""

import os
from dotenv import load_dotenv
from typing import List, Dict
import json

# LangChain imports
from langchain.tools import tool
# For demo purposes, we'll use a mock LLM that doesn't require an API key
from langchain.llms.fake import FakeListLLM
from langchain.chat_models.fake import FakeListChatModel
from langchain.schema import HumanMessage
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# CrewAI imports
from crewai import Agent, Task, Crew, Process

# Load environment variables
load_dotenv()

# Initialize a fake language model that returns pre-defined responses
responses = [
    "I've researched the topic extensively and found the following information...",
    "Based on my analysis, here are the key findings...",
    "Here's a comprehensive research report on the topic:\n\n# Executive Summary\n\nThis report provides an overview of the requested topic based on current information.\n\n## Key Findings\n\n1. Finding one with important details\n2. Finding two with critical analysis\n3. Finding three with future implications\n\n## Recommendations\n\nBased on the research, we recommend the following actions...",
]

llm = FakeListChatModel(responses=responses)

# Mock embeddings function
class MockEmbeddings:
    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3, 0.4, 0.5] for _ in texts]
    
    def embed_query(self, text):
        return [0.1, 0.2, 0.3, 0.4, 0.5]

embeddings = MockEmbeddings()

# For demo purposes, we'll use an in-memory mock database
class MockVectorDB:
    def __init__(self):
        self.data = []
    
    def add_documents(self, documents):
        self.data.extend(documents)
        return len(documents)
    
    def similarity_search(self, query, k=3):
        from langchain.schema import Document
        return [Document(page_content=f"Mock result for query: {query}", metadata={"source": "mock"})]
    
    def persist(self):
        pass

vector_db = MockVectorDB()

# Define tools
@tool
def search_web(query: str) -> str:
    """
    Search the web for information on a specific query.
    
    In a real implementation, you would use an actual search API like Serper:
    
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
    """
    # Simulated web search response
    return json.dumps({
        "organic": [
            {
                "title": "Example Research Result 1",
                "link": "https://example.com/research1",
                "snippet": "This is a snippet of information about the research topic."
            },
            {
                "title": "Example Research Result 2",
                "link": "https://example.com/research2",
                "snippet": "More information about the research topic from another source."
            }
        ]
    })

@tool
def scrape_webpage(url: str) -> str:
    """Scrape content from a webpage."""
    # In a real implementation, you would use WebBaseLoader:
    # loader = WebBaseLoader(url)
    # return loader.load()[0].page_content
    
    # Simulated webpage content
    return f"This is the content scraped from {url}. It contains detailed information about the research topic, including key findings, methodologies, and conclusions."

@tool
def store_information(data: str, topic: str) -> str:
    """Store information in the vector database."""
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(data)
    
    # Create documents with metadata
    documents = [{"content": text, "metadata": {"topic": topic}} for text in texts]
    
    # Add to vector store
    vector_db.add_documents(documents)
    vector_db.persist()
    
    return f"Stored {len(texts)} chunks of information about {topic} in the database."

@tool
def retrieve_information(query: str, k: int = 3) -> str:
    """Retrieve information from the vector database."""
    results = vector_db.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in results])

@tool
def generate_report(topic: str, information: str) -> str:
    """Generate a comprehensive research report."""
    # This would normally use the LLM directly, but for simplicity we'll use a tool
    report_prompt = f"""
    Generate a comprehensive research report about {topic} based on the following information:
    
    {information}
    
    The report should include:
    1. Executive Summary
    2. Introduction
    3. Key Findings
    4. Analysis
    5. Conclusion
    6. References
    """
    
    response = llm.invoke(report_prompt)
    return response.content

# Set up individual agents

# For demo purposes, we're not creating actual agents since they require API keys
# This is just a simplified demo implementation

# Create a function to set up a research crew for a specific topic
def create_research_crew(topic: str):
    """Create a specialized research crew for a specific topic."""
    
    # For demo purposes, we're just returning a simulated research report
    time.sleep(2)  # Simulate processing time
    
    return f"""# Research Report: {topic}

## Executive Summary

This report provides a comprehensive analysis of {topic} based on current information.

## Key Findings

1. The research indicates that {topic} is becoming increasingly important in today's landscape.
2. Recent developments have shown significant advances in this field.
3. Several challenges remain to be addressed for full implementation.

## Analysis

Our analysis reveals that {topic} has far-reaching implications across multiple domains.
The primary benefits include improved efficiency, cost reduction, and enhanced capabilities.
However, potential drawbacks include implementation complexity and initial investment requirements.

## Recommendations

Based on our research, we recommend:
1. Further exploration of specific applications
2. Development of integration strategies
3. Continued monitoring of emerging trends

## Conclusion

{topic} represents a significant opportunity with substantial potential benefits.
Strategic implementation and careful planning will be essential for successful adoption.

## References

- Example Reference 1 (2025)
- Example Reference 2 (2024)
- Example Reference 3 (2025)
"""

# Create an interactive interface
def main():
    print("🔍 Research Agent Initialized")
    print("---------------------------")
    print("This agent can research topics, analyze information, and generate reports.")
    print("Type 'exit' to quit.")
    
    while True:
        user_input = input("\nWhat would you like to research? ")
        
        if user_input.lower() in ["exit", "quit"]:
            print("Shutting down Research Agent. Goodbye!")
            break
        
        print(f"\n🔎 Researching: {user_input}...")
        
        # Create and execute the research crew
        result = create_research_crew(user_input)
        
        print("\n📊 Research Complete!")
        print("\nRESEARCH REPORT:")
        print("===============")
        print(result)

if __name__ == "__main__":
    main()
    
# This allows the web app to import this module without running the main function