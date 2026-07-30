# ==========================================================
# LangChain General Chat Model
# Using init_chat_model()
# Author: Muhammad Javed
# ==========================================================

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

# ==========================================================
# OPENAI
# ==========================================================

# model = init_chat_model(
#     "gpt-4o-mini",
#     model_provider="openai",
#     temperature=0.7
# )

# model = init_chat_model(
#     "gpt-4.1-mini",
#     model_provider="openai",
#     temperature=0.7
# )

# ==========================================================
# GROQ
# ==========================================================

# model = init_chat_model(
#     "openai/gpt-oss-120b",
#     model_provider="groq",
#     temperature=0.7
# )

# model = init_chat_model(
#     "llama-3.3-70b-versatile",
#     model_provider="groq",
#     temperature=0.7
# )

# model = init_chat_model(
#     "qwen/qwen3-32b",
#     model_provider="groq",
#     temperature=0.7
# )

# ==========================================================
# GOOGLE GEMINI
# ==========================================================

# model = init_chat_model(
#     "gemini-2.5-flash",
#     model_provider="google_genai",
#     temperature=0.7
# )

# model = init_chat_model(
#     "gemini-2.5-pro",
#     model_provider="google_genai",
#     temperature=0.7
# )

# ==========================================================
# MISTRAL AI
# ==========================================================

model = init_chat_model(
    "mistral-small-2506",
    model_provider="mistralai",
    temperature=0.7
)

# model = init_chat_model(
#     "mistral-medium-latest",
#     model_provider="mistralai",
#     temperature=0.7
# )

# ==========================================================
# COMMON PROMPT
# ==========================================================

prompt = """
Explain Machine Learning in one simple paragraph.
"""

# ==========================================================
# INVOKE
# ==========================================================

response = model.invoke(prompt)

# ==========================================================
# OUTPUT
# ==========================================================

print("\n" + "=" * 60)
print("MODEL RESPONSE")
print("=" * 60)

print(response.content)

print("=" * 60)