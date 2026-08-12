from dotenv import load_dotenv
load_dotenv()

from langchain_tavily import TavilySearch
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

search_tool = TavilySearch(max_results=5)

LLM = ChatMistralAI(model="mistral-small-2506")

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant.

Summarize the following news into clear bullet points:

{news}
"""
)

chain = prompt | LLM | StrOutputParser()

# Tool ko .invoke() se run karein
news_result = search_tool.invoke("Latest AI news in 2026")

result = chain.invoke({"news": news_result})

print(result)