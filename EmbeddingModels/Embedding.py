from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=64,
)

text = [
    "Data science is an interdisciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge and insights from structured and unstructured data.",
    "Data science is a multi-disciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge and insights from structured and unstructured data.",
    "Data science is a field that combines domain expertise, programming skills, and knowledge of mathematics and statistics to extract meaningful insights from data.",
]

vectors = embeddings.embed_documents(text)

print(vectors)