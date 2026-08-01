from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# vector = embeddings.embed_query("What is Data Science?")

text = [
    "Data science is an interdisciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge and insights from structured and unstructured data.",
    "Data science is a multi-disciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge and insights from structured and unstructured data.",
    "Data science is a field that combines domain expertise, programming skills, and knowledge of mathematics and statistics to extract meaningful insights from data.",
]


vector = embeddings.embed_documents(text)

print(vector)