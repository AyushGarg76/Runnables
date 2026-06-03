from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

model = ChatMistralAI(
    model_name="mistral-small-2506"
)
parser = StrOutputParser()

code_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a code generator"),
    ("human", "{topic}")
])

explain_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assissant that explains code in simple terms."),
    ("human", "Explain the following code:\n{code}")
])

seq = code_prompt | model | parser 

seq2 = RunnableParallel(
    {
        "code" : RunnablePassthrough(),
        "explanation" : explain_prompt | model | parser
    }
)

chain = seq | seq2

result = chain.invoke({"topic" : "Write a function that adds two numbers"})

print("code:", result['code'])
print("\n")
print("Explanation: ", result["explanation"])