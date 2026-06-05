from dotenv import load_dotenv
load_dotenv()

import os 
import requests
from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from tavily import TavilyClient
from rich import print

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
    print("DEBUG: ", data)

    return (
        f"City: {city}\n"
        f"Temperature: {data['main']['temp']}°C\n"
        f"Humidity: {data['main']['humidity']}%\n"
        f"Condition: {data['weather'][0]['description']}"
    )

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

    for r in results:
        output += (
            f"\nTitle: {r['title']}"
            f"\nContent: {r['content']}"
            f"\nURL: {r['url']}\n"
        )

    return output

llm = ChatMistralAI(model = "mistral-small-2506")
tools = {
    "get_weather": get_weather,
    "get_news": get_news
} 

llm_with_tool = llm.bind_tools([get_weather, get_news])

# Agent LOOP - Very importany

messages = []

print("City intellegence System")
print("Type EXIT to quit")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    messages.append(HumanMessage(content = user_input))

    while True:
        result = llm_with_tool.invoke(messages)
        messages.append(result)

        # if tool is required 
        if result.tool_calls:
            for tool_call in result.tool_calls:
                tool_name = tool_call['name']
                
                #HUMAN in the loop
                confirm = input("Agent wants to call {tool_name} Approve(yes/no): ")

                if confirm.lower() == "no":
                    print("Tool call denied and I cannot get the latest information")
                    break

                #excute tool
                tool_result = tools[tool_name].invoke(tool_call)
                
                #add tool message
                messages.append(ToolMessage(content = str(tool_result), tool_call_id = tool_call['id']))
                
                #Ask llm for final result
            continue
        else:
            print(f"Agent: {result.content}")
            break