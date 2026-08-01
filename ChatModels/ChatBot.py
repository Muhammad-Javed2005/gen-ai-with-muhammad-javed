from dotenv import load_dotenv 
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import AIMessage , HumanMessage , SystemMessage

load_dotenv()


model = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0.9
)

print("Choose Your AI Mode")
print("Press 1 for Angry Mood")
print("Press 2 for Funny Mood")
print("Press 3 for Sad Mood")


choice = int(input("Enter your choice : "))

if choice == 1 :
    mode = "You are an angry AI Agent.You respond aggressively and with frustration."

elif choice == 2 :
    mode = "You are a Funny AI Agent. You respond with humor and jokes."

elif choice == 3 :
    mode = "You are a Sad AI Agent. You respond with sadness and empathy."




message = [
    SystemMessage(content=mode),
]

while True:
    promt = input("You : ")
    if promt.lower() == "exit":
        break
    message.append(HumanMessage(content=promt))

    response = model.invoke(message)
    message.append(AIMessage(content=response.content))

    print("AI : ", response.content)

