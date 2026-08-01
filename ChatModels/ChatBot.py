from dotenv import load_dotenv 

load_dotenv()

from langchain_mistralai import ChatMistralAI

model = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0.9
)

message = []

while True:
    promt = input("You : ")
    if promt.lower() == "exit":
        break
    message.append(promt)

    response = model.invoke(message)
    message.append(response.content)

    print("AI : ", response.content)

