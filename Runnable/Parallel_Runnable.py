from dotenv import load_dotenv

load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel , RunnableLambda

# Components 

model = ChatMistralAI(model = "mistral-small-2506")
parser = StrOutputParser()

# Two differenet Promt 

short_promt = ChatPromptTemplate.from_template(
    "Explain {topic} into 1 to 2 line."
)

detailed_promt = ChatPromptTemplate.from_template(
    "Explain {topic} in detial."
)


# INput 

topic = "Machine Learning"

chain = RunnableParallel({
    "short" : RunnableLambda(lambda x : x["short"]) | short_promt | model | parser ,
    "detailed" : RunnableLambda(lambda x : x["detailed"]) | detailed_promt | model |parser
})

result = chain.invoke({
    "short" : {"topic" : "Machine learning"},
    "detailed" : {"topic" : "Deep Learning"}
})

print(result["short"])
print(result["detailed"])





