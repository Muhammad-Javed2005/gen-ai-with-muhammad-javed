from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()


LLM = HuggingFaceEndpoint(
    repo_id = "deepseek-ai/DeepSeek-R1"
)


model = ChatHuggingFace(llm = LLM)

response = model.invoke("Who are you")

print(response.content)