import os

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import ChatMistralAI

from Chroma_dh_create import get_vector_store


load_dotenv()


PDF_PATH = r"D:\gen-ai-with-muhammad-javed\Vector Store\deeplearning (2).pdf"


def load_and_store_pdf():
    vector_store = get_vector_store()

    # Check if PDF is already stored
    collection = vector_store._collection
    existing_count = collection.count()

    if existing_count > 0:
        print("PDF already exists in ChromaDB.")
        print(f"Existing chunks: {existing_count}")
        return vector_store

    print("Loading PDF...")

    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    print(f"PDF pages loaded: {len(documents)}")

    # Character-based chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")

    print("Creating embeddings and storing in ChromaDB...")

    vector_store.add_documents(chunks)

    print("PDF successfully stored in ChromaDB.")

    return vector_store


def create_llm():
    llm = ChatMistralAI(
        model="mistral-small-2506",
        temperature=0.2
    )

    return llm


def ask_question(vector_store, llm, question):

    results = vector_store.similarity_search_with_score(
        question,
        k=5
    )

    print("\n========== RETRIEVED CHUNKS ==========\n")

    for i, (document, score) in enumerate(results):
        print(f"\n--- CHUNK {i + 1} | SCORE: {score:.4f} ---")
        print(document.page_content[:1000])

    print("\n=======================================\n")

    # Best matching chunk only
    best_document = results[0][0]

    context = best_document.page_content

    prompt = f"""
You are a question-answering assistant.

Answer the question using ONLY the provided context.

Context:
{context}

Question:
{question}

Give a direct and concise answer.

If the answer cannot be found in the context, say:
"Sorry, this information is not available in the PDF."
"""

    response = llm.invoke(prompt)

    return response.content


def main():

    print("=" * 60)
    print("       PDF RAG SYSTEM - MISTRAL + CHROMADB")
    print("=" * 60)

    # PDF will be loaded only if ChromaDB is empty
    vector_store = load_and_store_pdf()

    llm = create_llm()

    print("\nRAG system is ready.")
    print("Ask questions from your PDF.")
    print("Type 'exit' to stop.\n")

    while True:

        question = input("You: ")

        if question.lower() == "exit":
            print("Exiting...")
            break

        if not question.strip():
            continue

        answer = ask_question(
            vector_store,
            llm,
            question
        )

        print("\nAssistant:")
        print(answer)
        print("-" * 60)


if __name__ == "__main__":
    main()