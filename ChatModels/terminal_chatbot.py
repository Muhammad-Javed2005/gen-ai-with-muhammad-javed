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
stats = {
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

while True:

    question = input(Fore.BLUE + "You : " + Style.RESET_ALL)

    command = question.lower().strip()

    if command == "exit":
        print(Fore.YELLOW + " Thanks for using the AI Teacher Chatbot. Goodbye!")
        break 

    elif command == "help":
        print(Fore.GREEN + """
        Available Commands
        help 
        history 
        stats 
        save 
        clear
        exit
        """)
        continue

    elif command == "clear":
        chat_history.clear()
        print(Fore.RED + "Chat history cleared.")
        continue

    elif command == "history" :

        if not chat_history:
            print(Fore.YELLOW + "No previous chat history.")
            continue  

        print(Fore.MAGENTA + "Conversation History:\n")

        for msg in chat_history:

            if isinstance(msg, HumanMessage):
                print(Fore.GREEN + "You :", msg.content)

            else:
                print(Fore.CYAN + "AI  :", msg.content)

            print()

        continue

    elif command == "stats":

        print(Fore.YELLOW + "\nChat Statistics")
        print("-" * 30)
        print(f"Questions : {stats['questions']}")
        print(f"Answers   : {stats['answers']}")
        print(f"Messages  : {len(chat_history)}\n")
        continue

    elif command == "save":

        filename = "chat_history.txt" 

        with open(filename  , "w" , encoding = "uft-8") as file :
            file.write("Chat History\n")
            file.write("=" * 30 + "\n\n")

            for msg in chat_history:
                if isinstance(msg, HumanMessage):
                    file.write("You : " + msg.content + "\n")
                else:
                    file.write("AI  : " + msg.content + "\n")

        print(Fore.GREEN + f"Chat history saved to {filename}.")
        continue


response = chain.invoke({
    "question" : question, 
    "chat_history" : chat_history
})


print(Fore.CYAN + "\nAI:\n")
print(Fore.WHITE + response.content)
print()

chat_history.append(HumanMessage(content=question))
chat_history.append(AIMessage(content=response.content))

stats["questions"] += 1
stats["answers"] += 1


