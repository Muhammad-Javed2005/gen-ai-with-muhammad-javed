from dotenv import load_dotenv

load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Promt Template 

promt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple Word"
)

# Model 
model = ChatMistralAI(model = "mistral-small-2506")

# Output Praser 

parser = StrOutputParser()



chain = promt | model | parser


result = chain.invoke("Machine Learning")

print(result)