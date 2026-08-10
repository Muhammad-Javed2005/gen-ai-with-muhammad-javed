from pathlib import Path

from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv
import os


# ============================================================
# Configuration
# ============================================================

CHROMA_PATH = Path(
    r"D:\gen-ai-with-muhammad-javed\Vector Store\deep_learning_chroma"
)

COLLECTION_NAME = "deep_learning_book_mistral"

EMBEDDING_MODEL = "mistral-embed"


# ============================================================
# Load API Key
# ============================================================

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    raise ValueError(
        "MISTRAL_API_KEY not found in .env"
    )


# ============================================================
# Start
# ============================================================

print("=" * 70)
print("        CHROMADB EMBEDDING VERIFICATION")
print("=" * 70)

print(f"\nChromaDB:")
print(CHROMA_PATH)

print(f"\nCollection:")
print(COLLECTION_NAME)

print(f"\nEmbedding Model:")
print(EMBEDDING_MODEL)


# ============================================================
# Check ChromaDB folder
# ============================================================

if not CHROMA_PATH.exists():
    raise FileNotFoundError(
        f"ChromaDB not found:\n{CHROMA_PATH}"
    )


# ============================================================
# Load Mistral Embedding Model
# ============================================================

print("\n" + "=" * 70)
print("[1/4] Initializing Mistral Embeddings...")
print("=" * 70)

embeddings = MistralAIEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=MISTRAL_API_KEY,
)

print("Mistral Embeddings initialized.")


# ============================================================
# Load Existing ChromaDB
# ============================================================

print("\n" + "=" * 70)
print("[2/4] Loading existing ChromaDB...")
print("=" * 70)

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=str(CHROMA_PATH),
)

print("ChromaDB loaded successfully.")


# ============================================================
# Get Collection
# ============================================================

collection = vectorstore._collection

total_documents = collection.count()

print("\n" + "=" * 70)
print("[3/4] Checking stored data...")
print("=" * 70)

print(
    f"Total documents in ChromaDB: "
    f"{total_documents}"
)


# ============================================================
# Get One Stored Vector
# ============================================================

sample = collection.get(
    limit=1,
    include=[
        "documents",
        "embeddings",
        "metadatas",
    ],
)


if not sample["documents"]:

    print("\nERROR: No documents found!")

    raise SystemExit


document = sample["documents"][0]

embedding = sample["embeddings"][0]

metadata = sample["metadatas"][0]


print("\nSample document found: YES")

print("\nDocument preview:")
print("-" * 70)
print(document[:500])
print("-" * 70)


# ============================================================
# Verify Embedding
# ============================================================

print("\nEmbedding found: YES")

print(
    f"Embedding dimension: "
    f"{len(embedding)}"
)

print("\nFirst 10 embedding values:")

print(
    embedding[:10]
)


# ============================================================
# Verify Metadata
# ============================================================

print("\nMetadata:")

for key, value in metadata.items():

    print(
        f"  {key}: {value}"
    )


# ============================================================
# Test Query Embedding
# ============================================================

print("\n" + "=" * 70)
print("[4/4] Testing query embedding...")
print("=" * 70)

test_query = (
    "What is deep learning?"
)

print(
    f"\nTest query: {test_query}"
)

query_vector = embeddings.embed_query(
    test_query
)

print(
    "\nQuery embedding generated successfully."
)

print(
    f"Query vector dimension: "
    f"{len(query_vector)}"
)


# ============================================================
# Similarity Search
# ============================================================

print("\nPerforming similarity search...")

results = vectorstore.similarity_search_with_score(
    test_query,
    k=3,
)


print(
    f"\nResults returned: "
    f"{len(results)}"
)


for index, (doc, score) in enumerate(
    results,
    start=1
):

    print("\n" + "-" * 70)

    print(
        f"Result {index}"
    )

    print(
        f"Similarity distance: "
        f"{score}"
    )

    print(
        f"Page: "
        f"{doc.metadata.get('page', 'N/A')}"
    )

    print("\nContent:")

    print(
        doc.page_content[:500]
    )


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 70)
print("             VERIFICATION RESULT")
print("=" * 70)

if (
    total_documents == 1145
    and len(embedding) == 1024
    and len(query_vector) == 1024
    and len(results) > 0
):

    print("\nALL CHECKS PASSED!")

    print(
        "\n1145 documents are stored in ChromaDB."
    )

    print(
        "Document embeddings are present."
    )

    print(
        "Embedding dimension is 1024."
    )

    print(
        "Query embedding is also 1024-dimensional."
    )

    print(
        "Similarity search is working."
    )

    print(
        "\nYour RAG ChromaDB is ready for chat.py."
    )

else:

    print(
        "\nWARNING: One or more checks failed."
    )

print("\n" + "=" * 70)