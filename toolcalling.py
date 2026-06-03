from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain.tools import tool 
from rich import print

#1 creating a tool

@tool
def get_text_length(text: str) -> int:
    """Returns the number of characters in a given text"""
    return len(text)


llm = ChatMistralAI(model_name = "mistral-small-2506")

#Tool Binding 
llm_with_tool = llm.bind_tools([get_text_length])

result = llm_with_tool.invoke("Return the number of character in a given text: 'Hello how are you?'")

print(result.tool_calls)

print(get_text_length.invoke({'text': 'hello how are you'}))