from ast import Str
from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# 1. Prompt Template
prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant.
    
    context : {context}
    question : {question}
    """
)
    
# 2. Model
model = ChatMistralAI(
    model_name="mistral-small-2506"
)

# 3. Output Parser
parser = StrOutputParser()

# Step by step manual flow

#format the prompt
formatted_prompt = prompt.format_messages(topic="Machine Learning", context="This is a test context", question = "What is something important related to machine learning")

#call the model manually
response = model.invoke(formatted_prompt)

#parse the output manually
final_output = parser.parse(response.content)

print(final_output)

chain = prompt | model | parser

result = chain.invoke({"topic" : "Machine Learning", "context" : "This is a test context", "question" : "What is something important related to machine learning"})
print(result)