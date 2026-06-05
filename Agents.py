from dotenv import load_dotenv
load_dotenv()

import os 
import requests
from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
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

llm = ChatMistralAI(model = "mistral-small-2506")
tools = {
    "get_weather": get_weather,
    "get_news": get_news
} 

llm_with_tool = llm.bind_tools([get_weather, get_news])
# ==============================
# Agent Loop
# ==============================

messages = []

console.print(
    Panel.fit(
        "🌍 City Intelligence Agent",
        title="AI Assistant",
        border_style="cyan",
    )
)

console.print("[bold green]Type EXIT to quit[/bold green]")

while True:

    console.rule("[bold blue]New Query[/bold blue]")

    user_input = input("\nUser Query: ")

    if user_input.lower() == "exit":
        console.print("\n👋 Goodbye!\n")
        break

    messages.append(HumanMessage(content=user_input))

    while True:

        console.print("\n[cyan]🤔 Thinking...[/cyan]")

        result = llm_with_tool.invoke(messages)

        messages.append(result)

        # Tool required
        if result.tool_calls:

            console.print(
                "[magenta]🛠 Agent selected one or more tools[/magenta]\n"
            )

            for tool_call in result.tool_calls:

                tool_name = tool_call["name"]

                confirm = input(
                    f"🔧 Agent wants to call [{tool_name}] Approve? (yes/no): "
                )

                if confirm.lower() != "yes":
                    console.print(
                        "[red]❌ Tool call denied[/red]"
                    )
                    break

                console.print(
                    f"\n[yellow]⏳ Running {tool_name}...[/yellow]"
                )

                try:

                    tool_result = tools[tool_name].invoke(tool_call)

                    console.print(
                        f"[green]✅ {tool_name} completed[/green]"
                    )

                    messages.append(
                        ToolMessage(
                            content=str(tool_result),
                            tool_call_id=tool_call["id"],
                        )
                    )

                except Exception as e:

                    console.print(
                        f"[red]❌ Error in {tool_name}: {e}[/red]"
                    )

                    messages.append(
                        ToolMessage(
                            content=f"Tool Error: {e}",
                            tool_call_id=tool_call["id"],
                        )
                    )

            continue

        else:

            console.print()

            console.print(
                Panel(
                    Markdown(result.content),
                    title="🤖 Agent Response",
                    border_style="green",
                )
            )

            break