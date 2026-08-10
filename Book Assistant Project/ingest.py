from pathlib import Path
import os
import time

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings


# ============================================================
# Load environment variables
# ============================================================

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    raise ValueError(
        "MISTRAL_API_KEY not found in .env file."
    )


# ============================================================
# Configuration
# ============================================================

PDF_PATH = Path(
    r"D:\gen-ai-with-muhammad-javed\Vector Store\deeplearning (2).pdf"
)

CHROMA_PATH = Path(
    r"D:\gen-ai-with-muhammad-javed\Vector Store\deep_learning_chroma"
)

COLLECTION_NAME = "deep_learning_book_mistral"

EMBEDDING_MODEL = "mistral-embed"

BATCH_SIZE = 16


# ============================================================
# Start
# ============================================================

print("=" * 70)
print("       DEEP LEARNING RAG - MISTRAL INGESTION")
print("=" * 70)

print(f"\nPDF:")
print(PDF_PATH)

print(f"\nChromaDB:")
print(CHROMA_PATH)

print(f"\nCollection:")
print(COLLECTION_NAME)

print(f"\nEmbedding Model:")
print(EMBEDDING_MODEL)

print(f"\nBatch Size:")
print(BATCH_SIZE)


# ============================================================
# Check PDF
# ============================================================

if not PDF_PATH.exists():
    raise FileNotFoundError(
        f"PDF not found:\n{PDF_PATH}"
    )


# ============================================================
# 1. Load PDF
# ============================================================

print("\n" + "=" * 70)
print("[1/5] Loading PDF...")
print("=" * 70)

loader = PyPDFLoader(str(PDF_PATH))

documents = loader.load()

print(f"Pages loaded: {len(documents)}")

if not documents:
    raise ValueError("No pages were loaded.")


# ============================================================
# 2. Recursive Text Splitting
# ============================================================

print("\n" + "=" * 70)
print("[2/5] Splitting PDF using RecursiveCharacterTextSplitter...")
print("=" * 70)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ],
)

chunks = text_splitter.split_documents(documents)

print(f"Total chunks created: {len(chunks)}")

if not chunks:
    raise ValueError("No chunks were created.")


# ============================================================
# Add metadata
# ============================================================

for index, chunk in enumerate(chunks):

    chunk.metadata["chunk_id"] = index
    chunk.metadata["source_file"] = PDF_PATH.name


print("\nExample chunk:")
print("-" * 70)
print(chunks[0].page_content[:500])
print("-" * 70)


# ============================================================
# 3. Load Mistral Embeddings
# ============================================================

print("\n" + "=" * 70)
print("[3/5] Loading Mistral Embedding Model...")
print("=" * 70)

embeddings = MistralAIEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=MISTRAL_API_KEY,
)

print("Mistral Embedding model initialized successfully.")


# ============================================================
# 4. Create ChromaDB
# ============================================================

print("\n" + "=" * 70)
print("[4/5] Initializing ChromaDB...")
print("=" * 70)

CHROMA_PATH.mkdir(
    parents=True,
    exist_ok=True
)

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=str(CHROMA_PATH),
)

print("ChromaDB initialized successfully.")


# ============================================================
# 5. Batch Embedding + ChromaDB
# ============================================================

print("\n" + "=" * 70)
print("[5/5] Creating Mistral embeddings...")
print("=" * 70)

total_chunks = len(chunks)

total_batches = (
    total_chunks + BATCH_SIZE - 1
) // BATCH_SIZE

successful_chunks = 0

overall_start = time.time()


# ============================================================
# Process batches
# ============================================================

for batch_number, start in enumerate(
    range(0, total_chunks, BATCH_SIZE),
    start=1
):

    end = min(
        start + BATCH_SIZE,
        total_chunks
    )

    batch = chunks[start:end]

    texts = [
        document.page_content
        for document in batch
    ]

    print("\n" + "-" * 70)

    print(
        f"Batch {batch_number}/{total_batches}"
    )

    print(
        f"Chunks: {start + 1}-{end}/{total_chunks}"
    )

    print(
        f"Progress: "
        f"{(start / total_chunks) * 100:.2f}%"
    )

    print(
        f"Sending {len(texts)} chunks "
        f"to Mistral..."
    )

    batch_start = time.time()

    try:

        # ----------------------------------------------------
        # Generate embeddings
        # ----------------------------------------------------

        batch_embeddings = embeddings.embed_documents(
            texts
        )

        print(
            f"Embeddings generated: "
            f"{len(batch_embeddings)}"
        )

        # ----------------------------------------------------
        # Verify embedding dimension
        # ----------------------------------------------------

        dimension = len(batch_embeddings[0])

        print(
            f"Embedding dimension: "
            f"{dimension}"
        )

        # ----------------------------------------------------
        # Create IDs
        # ----------------------------------------------------

        ids = [
            f"deep_learning_mistral_{start + i}"
            for i in range(len(batch))
        ]

        # ----------------------------------------------------
        # Save to ChromaDB
        # ----------------------------------------------------

        print("Saving batch to ChromaDB...")

        vectorstore._collection.add(
            ids=ids,
            embeddings=batch_embeddings,
            documents=texts,
            metadatas=[
                document.metadata
                for document in batch
            ],
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        successful_chunks += len(batch)

        batch_time = time.time() - batch_start

        progress = (
            successful_chunks / total_chunks
        ) * 100

        elapsed = time.time() - overall_start

        average_time = (
            elapsed / successful_chunks
        )

        remaining_chunks = (
            total_chunks - successful_chunks
        )

        estimated_remaining = (
            average_time * remaining_chunks
        )

        print(
            "ChromaDB save: SUCCESS"
        )

        print(
            f"Progress: "
            f"{successful_chunks}/{total_chunks} "
            f"({progress:.2f}%)"
        )

        print(
            f"Batch time: "
            f"{batch_time:.2f} seconds"
        )

        print(
            f"Estimated remaining: "
            f"{estimated_remaining / 60:.2f} minutes"
        )

    except Exception as e:

        print("\n" + "!" * 70)
        print(
            f"ERROR in batch "
            f"{batch_number}/{total_batches}"
        )
        print(
            f"Chunks: {start + 1}-{end}"
        )
        print(
            f"Error: {e}"
        )
        print("!" * 70)

        raise


# ============================================================
# Final Verification
# ============================================================

total_time = time.time() - overall_start

final_count = vectorstore._collection.count()


print("\n" + "=" * 70)
print("          INGESTION COMPLETED SUCCESSFULLY")
print("=" * 70)



print(f"\nPDF Pages:             {len(documents)}")
print(f"Total Chunks:          {total_chunks}")
print(f"Successfully Embedded: {successful_chunks}")
print(f"ChromaDB Documents:    {final_count}")
print(f"Embedding Model:       {EMBEDDING_MODEL}")
print(f"Embedding Dimension:   1024")
print(f"Total Time:            {total_time / 60:.2f} minutes")
print(f"ChromaDB Location:     {CHROMA_PATH}")
print(f"Collection:            {COLLECTION_NAME}")

print("\n" + "=" * 70)
print("Deep Learning PDF successfully stored in ChromaDB.")
print("=" * 70)



collection = vectorstore._collection

print("\n" + "=" * 70)
print("CHROMA VERIFICATION")
print("=" * 70)

print("Documents:", collection.count())

sample = collection.get(
    limit=1,
    include=["documents", "embeddings", "metadatas"]
)

print("Document found:", len(sample["documents"]))

if sample["embeddings"]:
    print(
        "Embedding dimension:",
        len(sample["embeddings"][0])
    )

    print(
        "First 10 embedding values:",
        sample["embeddings"][0][:10]
    )

else:
    print("ERROR: No embeddings found!")