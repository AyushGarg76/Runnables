#Used the same agent but used create_agent in langchain
from dotenv import load_dotenv
load_dotenv()

import os 
import requests
from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from tavily import TavilyClient
from rich import print

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()

#Weather Tool 
@tool 
def get_weather(city : str) -> str:
    """Get the current weather of a city"""
    api_key = os.getenv("OPENWEATHER_API_KEY")

    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?q={city}&appid={api_key}&units=metric"
    )

    response = requests.get(url)

    if response.status_code != 200:
        return f"Could not get weather for {city}"

    data = response.json()
    # print("DEBUG: ", data)

    return f"""
# 🌤 Weather Report

📍 **City:** {city}

🌡 **Temperature:** {data['main']['temp']}°C

💧 **Humidity:** {data['main']['humidity']}%

☁ **Condition:** {data['weather'][0]['description'].title()}
"""

#Tavily News Tool

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def get_news(query: str) -> str:
    """Get latest news about a topic"""

    response = tavily_client.search(
        query=f"latest news about {query}",
        search_depth="basic",
        max_results=5
    )

    answer = response.get("answer", "")

    results = response.get("results", [])

    output = f"Summary:\n{answer}\n\nSources:\n"

    for idx, r in enumerate(results, start=1):

        title = r.get("title", "No Title")
        content = r.get("content", "No Content")
        url = r.get("url", "")

        output += f"""

## 📰 News {idx}

**{title}**

{content[:250]}...

🔗 {url}

"""

    return output

# LLM setup

llm = ChatMistralAI(model = "mistral-small-2506")

@wrap_tool_call
def human_approval(request,handler):
    """Ask for human approval before the tool call."""
    tool_name = request.tool_call["name"]
    confirm = input(f"Agents wants to call '{tool_name}'. Approve(yes/no)")
    if confirm.lower() != "yes":
        return ToolMessage(content= "Tool call denied by user.", tool_call_id=request.tool_call["id"])
    return handler(request)

agents = create_agent(
    llm, 
    tools = [get_weather, get_news],
    system_prompt = """
You are a helpful assistant that can provide weather reports and news.
""",
    middleware = [human_approval]
)

print("City Agent | type exit to quit")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Exiting...")
        break
    
    result = agents.invoke(
        {"messages":[{"role": "human", "content": user_input}]
        }
    )
    console.print(Panel(Markdown(result['messages'][-1].content)))