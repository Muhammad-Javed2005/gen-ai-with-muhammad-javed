from sentence_transformers import SentenceTransformer

model_name = "Qwen/Qwen3-Embedding-0.6B"

print("Loading embedding model...")

model = SentenceTransformer(
    model_name,
    device="cpu"
)

print("Model loaded successfully!")

texts = [
    "Deep learning is a subset of machine learning.",
    "Neural networks are used in deep learning."
]

embeddings = model.encode(
    texts,
    normalize_embeddings=True
)

print("Embeddings created successfully!")
print("Number of embeddings:", len(embeddings))
print("Embedding dimension:", embeddings.shape[1])
print("First embedding preview:", embeddings[0][:10])