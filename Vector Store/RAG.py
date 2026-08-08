import os 
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import ChatMistralAI

from Chroma_dh_create import get_vector_store

PDF_PATH = r"D:\gen-ai-with-muhammad-javed\Vector Store\deeplearning (2).pdf"


def load_store_pdf():
    vector_store = get_vector_store()

    # Check if PDF is already stored 

    collection = vector_store._collection
    existing_count = collection.count()


    if existing_count > 0 :
        print("PDF is already existing ChromaDB")
        print(f"Existing Chunks : {existing_count}")

        return vector_store

    print("Loading PDF....")


    loader = PyPDFLoader(PDF_PATH)
    documenets = loader.Load()

    print(f"PDF Pages loaded : {len(documenets)}")


    # Character Base Chunking 
    text_spiltter = RecursiveCharacterTextSplitter(
        chuck_size = 100 ,
        chunk_overlap = 200
    ) 

    chunks = text_spiltter.split_documents(documenets)
    print(f"Total Chunk Created : {len(chunks)}")
    print("Creating Embedding and storing in Chroma DB...")


    vector_store.add_documents(chunks)

    print("PDF successfully stored in Chorma_DB")

    return vector_store


def creat_LLM():
    llm = ChatMistralAI(
        model = "mistral-small-2506",
        temperature = 0.2
    )

    return llm 

def ask_question(vector_store , llm , question):

    # Retrieve similar chunks from ChromaDB

    result = vector_store.similarity_search(
        question , 
        k = 4 
    )

    if not result :
        return "Sorry , I could not find relevent informarion in this PDF"

    context = "\n\n".join(
        document.page_content
        for document in result
    )

    prompt = f"""
You are a PDF-based AI assistant.

Answer the user's question ONLY using the information
provided in the context below.

If the answer is not available in the context, say:
"Sorry, this information is not available in the PDF."

Do not use your own general knowledge.

Context:
{context}

User Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content



def main():

    print("=" * 60)
    print("       PDF RAG SYSTEM - MISTRAL + CHROMADB")
    print("=" * 60)

    # PDF will be loaded only if ChromaDB is empty
    vector_store = load_store_pdf()

    llm = creat_LLM()

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



