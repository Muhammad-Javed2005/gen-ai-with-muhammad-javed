from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI

load_dotenv()

model = ChatMistralAI(
    model="mistral-small-latest"
)

response = model.invoke("Hello, how are you?")

print(response.content)