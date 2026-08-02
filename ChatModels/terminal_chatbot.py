from dotenv import load_dotenv
from colorama import Fore  , Style , init
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import AIMessage , HumanMessage , SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime


init(autoreset= True)
load_dotenv()

LLM = ChatMistralAI(
    model = "mistral-small-2506" ,
    temperature = 0.9
)

chat_history = []
status = {
    "question" : 0 ,
    "answer" : 0 ,
}


prompt = ChatPromptTemplate.from_messages([

              ("system", """
You are an Expert AI Teacher.

Whenever the user asks about any topic, always answer using the following format.

# Definition

# Explanation

# Simple Example

# Role in IT Industry

# Key Points

If previous conversation exists,
use it to maintain context.

Keep explanations beginner friendly.
"""
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{question}")
])


chain = prompt | LLM



print(Fore.CYAN + " = " * 65)
print(Fore.YELLOW + "Welcome to the AI Teacher Chatbot! Ask any question about IT topics.")
print(Fore.CYAN + " = " * 65)


print(Fore.GREEN + """
Available Commands

help      -> Show Commands
history   -> Show Previous Chat
stats     -> Show Statistics
save      -> Save Chat
clear     -> Clear Memory
exit      -> Quit

""")





