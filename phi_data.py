from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Web Search Agent
web_search = Agent(
    name="Web Search Agent",
    role="Search the web for information",
    model=Groq(id="gemma2-9b-it", name="Groq", provider="Groq"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources and format findings in Markdown."],
    show_tool_calls=True,
    markdown=True
)

# Financial Agent
finance_agent = Agent(
    name="Finance Agent",
    model=Groq(id="gemma2-9b-it", name="Groq", provider="Groq"),
    tools=[
        YFinanceTools(
            stock_price=True, 
            analyst_recommendations=True, 
            stock_fundamentals=True
            
        )
    ],
    show_tool_calls=True,
    description="You are an investment analyst. Provide stock data in Markdown tables where applicable.",
    instructions=["Format your response in Markdown, use tables for structured data."],
    markdown=True
)

# Multi AI Agent
multi_ai_agent = Agent(
    team=[web_search, finance_agent],
    model=Groq(id="gemma2-9b-it"),
    instructions=[
        
        "Use tables to display structured data where possible.",
        
    ],
    show_tool_calls=True,
    markdown=True
)

# Run the multi-agent request
response = multi_ai_agent.print_response(
    "give concise summary of recent NVIDIA news and analyst recomendations"
)