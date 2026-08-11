from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

load_dotenv()

# Model initialize karein
model = ChatMistralAI(model="mistral-small-2506")
parser = StrOutputParser()

# 1. Code Generation Prompt
code_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful Code Generator. Return ONLY the code without markdown dynamic blocks."),
    ("human", "{topic}")
])

# 2. Code Explanation Prompt
explain_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant who explains code in simple words."),
    ("human", "Explain the following code in simple words:\n{code}")
])

# Code generate karne ki sequence
seq1 = code_prompt | model | parser

# Parallel run karne ki sequence
seq2 = RunnableParallel(
    {
        "code": RunnablePassthrough(),
        "explanation": explain_prompt | model | parser
    }
)

# Dono chains ko combine kiya
chain = seq1 | seq2

# Invoke karte waqt 'topic' ko small 't' se pass karein
result = chain.invoke({"topic": "Please write a code for palindrome in python"})

print("=== GENERATED CODE ===")
print(result["code"])

print("\n=== EXPLANATION ===")
print(result["explanation"])