

from dotenv import load_dotenv

load_dotenv()

# OPENAI

from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7
)

# GROQ

from langchain_groq import ChatGroq

model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7
)

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7
)

model = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0.7
)

# GOOGLE GEMINI

from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0.7
)

# MISTRAL AI

from langchain_mistralai import ChatMistralAI

model = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0.7
)

model = ChatMistralAI(
    model="mistral-medium-latest",
    temperature=0.7
)



prompt = """
Explain Machine Learning in one simple paragraph.
"""


response = model.invoke(prompt)



print("\n" + "=" * 60)
print("MODEL RESPONSE")
print("=" * 60)

print(response.content)

print("\n" + "=" * 60)