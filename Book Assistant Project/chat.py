import os

from dotenv import load_dotenv
from colorama import Fore, Style, init

from langchain_chroma import Chroma
from langchain_mistralai import (
    MistralAIEmbeddings,
    ChatMistralAI,
)
from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# COLORAMA
# ============================================================

init(autoreset=True)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    raise ValueError(
        "MISTRAL_API_KEY not found in .env file."
    )


# ============================================================
# CONFIGURATION
# ============================================================

CHROMA_PATH = (
    r"D:\gen-ai-with-muhammad-javed"
    r"\Vector Store\deep_learning_chroma"
)

COLLECTION_NAME = "deep_learning_book_mistral"

EMBEDDING_MODEL = "mistral-embed"

# Mistral chat model
LLM_MODEL = "mistral-small-2506"

TOP_K = 4


# ============================================================
# TERMINAL HEADER
# ============================================================

print()

print(
    Fore.CYAN
    + "=" * 75
)

print(
    Fore.CYAN
    + "             DEEP LEARNING RAG CHATBOT"
)

print(
    Fore.CYAN
    + "=" * 75
)

print(
    Fore.YELLOW
    + "Knowledge Base : "
    + Fore.WHITE
    + "Deep Learning Book"
)

print(
    Fore.YELLOW
    + "Embedding       : "
    + Fore.WHITE
    + EMBEDDING_MODEL
)

print(
    Fore.YELLOW
    + "LLM             : "
    + Fore.WHITE
    + LLM_MODEL
)

print(
    Fore.YELLOW
    + "Vector Database : "
    + Fore.WHITE
    + "ChromaDB"
)

print(
    Fore.CYAN
    + "=" * 75
)

print(
    Fore.GREEN
    + "\nType your question below."
)

print(
    Fore.GREEN
    + "Type 'exit' or 'quit' to close the chatbot."
)

print()


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print(
    Fore.YELLOW
    + "[1/3] Loading Mistral embedding model..."
)

embeddings = MistralAIEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=MISTRAL_API_KEY,
)

print(
    Fore.GREEN
    + "      Embedding model ready."
)


# ============================================================
# LOAD EXISTING CHROMADB
# ============================================================

print(
    Fore.YELLOW
    + "[2/3] Loading existing ChromaDB..."
)

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH,
)

collection = vectorstore._collection

document_count = collection.count()

print(
    Fore.GREEN
    + f"      ChromaDB loaded successfully."
)

print(
    Fore.GREEN
    + f"      Documents: {document_count}"
)


# ============================================================
# LOAD MISTRAL LLM
# ============================================================

print(
    Fore.YELLOW
    + "[3/3] Loading Mistral LLM..."
)

llm = ChatMistralAI(
    model=LLM_MODEL,
    temperature=0.2,
    api_key=MISTRAL_API_KEY,
)

print(
    Fore.GREEN
    + "      LLM ready."
)

print()


# ============================================================
# RAG PROMPT
# ============================================================

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a Deep Learning book assistant.

Answer the user's question using ONLY the
provided context retrieved from the Deep Learning book.

Rules:

1. Use the provided book context as your primary source.
2. Do not invent facts that are not supported by the context.
3. If the answer cannot be found in the provided context,
   clearly say:

   "I couldn't find this information in the provided book."

4. Explain the answer clearly and accurately.
5. Use simple language when possible.
6. If the question asks for an example, provide an example
   only when it is supported by the context.
7. Do not mention that you are an AI unless necessary.

Retrieved book context:

{context}
""",
        ),
        (
            "human",
            "{question}",
        ),
    ]
)


# ============================================================
# HELPER FUNCTION
# ============================================================

def format_context(documents):

    context_parts = []

    for i, document in enumerate(
        documents,
        start=1
    ):

        page = document.metadata.get(
            "page",
            "Unknown"
        )

        page = int(page) + 1 if isinstance(
            page,
            int
        ) else page

        context_parts.append(
            f"""
--- SOURCE {i} | PAGE {page} ---

{document.page_content}
"""
        )

    return "\n".join(context_parts)


# ============================================================
# CHAT LOOP
# ============================================================

while True:

    print(
        Fore.BLUE
        + "-" * 75
    )

    question = input(
        Fore.CYAN
        + "You: "
        + Style.RESET_ALL
    ).strip()

    if not question:
        print(
            Fore.YELLOW
            + "Please enter a question."
        )
        continue

    if question.lower() in {
        "exit",
        "quit",
        "q"
    }:

        print(
            Fore.MAGENTA
            + "\nGoodbye! Thanks for using "
              "Deep Learning RAG."
        )

        break


    # ========================================================
    # RETRIEVAL
    # ========================================================

    print(
        Fore.YELLOW
        + "\nSearching the Deep Learning book..."
    )

    results = vectorstore.similarity_search_with_score(
        question,
        k=TOP_K,
    )

    if not results:

        print(
            Fore.RED
            + "No relevant information found."
        )

        continue


    # ========================================================
    # EXTRACT DOCUMENTS
    # ========================================================

    documents = [
        document
        for document, score in results
    ]


    # ========================================================
    # SHOW RETRIEVED SOURCES
    # ========================================================

    print(
        Fore.GREEN
        + f"Retrieved {len(documents)} relevant chunks."
    )

    print(
        Fore.YELLOW
        + "\nRelevant pages:"
    )

    for index, (
        document,
        score
    ) in enumerate(
        results,
        start=1
    ):

        page = document.metadata.get(
            "page",
            "Unknown"
        )

        if isinstance(page, int):
            page += 1

        print(
            Fore.WHITE
            + f"  {index}. Page {page}"
            + Fore.LIGHTBLACK_EX
            + f"  | Distance: {score:.4f}"
        )


    # ========================================================
    # BUILD CONTEXT
    # ========================================================

    context = format_context(
        documents
    )


    # ========================================================
    # CREATE PROMPT
    # ========================================================

    messages = prompt.format_messages(
        context=context,
        question=question,
    )


    # ========================================================
    # GENERATE ANSWER
    # ========================================================

    print(
        Fore.YELLOW
        + "\nGenerating answer..."
    )

    try:

        response = llm.invoke(
            messages
        )

        answer = response.content

    except Exception as e:

        print(
            Fore.RED
            + "\nMistral API Error:"
        )

        print(
            Fore.RED
            + str(e)
        )

        continue


    # ========================================================
    # DISPLAY ANSWER
    # ========================================================

    print()

    print(
        Fore.GREEN
        + "=" * 75
    )

    print(
        Fore.GREEN
        + "                         ANSWER"
    )

    print(
        Fore.GREEN
        + "=" * 75
    )

    print(
        Fore.WHITE
        + answer
    )

    print(
        Fore.GREEN
        + "=" * 75
    )

    print()

    