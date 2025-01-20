from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
import groq
import time
from dotenv import load_dotenv

load_dotenv()

web_search_agent = Agent(
    name="Web Agent",
    description="This is the agent for searching content from the web",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=["Always include the sources"],
    show_tool_calls=True,
    markdown=True,
    debug_mode=True
)

finance_agent = Agent(
    name="Finance Agent",
    description="Your task is to find finance information",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            company_info=True,
            company_news=True
        )
    ],
    instructions=["Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
    debug_mode=True
)

agent_team = Agent(
    team=[web_search_agent, finance_agent],
    model=Groq(id="llama-3.3-70b-versatile"),
    instructions=["Always include sources", "Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
    debug_mode=True
)

def rate_limited_response(agent, query):
    try:
        return agent.print_response(query, stream=True)
    except groq.APIStatusError as e:
        if "rate_limit_exceeded" in str(e):
            time.sleep(60)  # Wait 1 minute
            return agent.print_response(query, stream=True)
        raise e

# Use the multi-agent system like this:
rate_limited_response(
    agent_team,
    "Summarize analyst recommendations and share the latest news for TSLA"
)