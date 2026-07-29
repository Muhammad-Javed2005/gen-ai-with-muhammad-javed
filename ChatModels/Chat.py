from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chat_models import init_chat_model
load_dotenv()

# llm = ChatGoogleGenerativeAI(
#     model="gemini-3.5-flash"
# )

model = init_chat_model(
    "gemini-3.5-flash",
    model_provider="google_genai",
    temperature=0.7
)


response = model.invoke("Give me a paragraph about the benefits of using AI in healthcare.")


print(response.content[0]["text"])

