from dotenv import load_dotenv 
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import AIMessage , HumanMessage , SystemMessage

load_dotenv()


model = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0.9
)

message = [
    SystemMessage(content="You are the funny AI Agent."),
]

while True:
    promt = input("You : ")
    if promt.lower() == "exit":
        break
    message.append(HumanMessage(content=promt))

    response = model.invoke(message)
    message.append(AIMessage(content=response.content))

    print("AI : ", response.content)

