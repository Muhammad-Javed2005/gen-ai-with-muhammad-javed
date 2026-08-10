import time

from langchain_huggingface import HuggingFaceEmbeddings


MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"


print("Loading Qwen model...")

embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={
        "device": "cpu"
    },
    encode_kwargs={
        "normalize_embeddings": True,
        "batch_size": 4
    },
)

print("Model loaded successfully!")

texts = [
    "Deep learning is a branch of machine learning.",
    "Neural networks are widely used in deep learning.",
    "Backpropagation is used to train neural networks.",
    "Gradient descent is an optimization algorithm.",
    "A neural network consists of interconnected neurons.",
    "Convolutional neural networks are useful for image processing.",
    "Recurrent neural networks process sequential information.",
    "Deep learning models can contain many layers.",
    "Activation functions introduce non-linearity.",
    "Training requires optimization of model parameters.",
    "Loss functions measure prediction errors.",
    "Deep neural networks can learn complex representations.",
    "Machine learning uses data to learn patterns.",
    "Artificial intelligence includes machine learning.",
    "Embeddings represent text as numerical vectors.",
    "Vector databases store embeddings for similarity search.",
]


print(f"\nTrying to embed {len(texts)} texts...")

start = time.time()

vectors = embeddings.embed_documents(texts)

end = time.time()

print("\nEmbedding completed successfully!")

print("Number of vectors:", len(vectors))
print("Vector dimension:", len(vectors[0]))
print("Time taken:", round(end - start, 2), "seconds")

print("\nFirst vector preview:")
print(vectors[0][:10])