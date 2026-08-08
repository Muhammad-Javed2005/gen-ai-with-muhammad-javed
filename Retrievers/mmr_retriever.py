from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "retriever_demo"


embeddings = MistralAIEmbeddings(
    model="mistral-embed"
)


vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH
)


retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 6,
        "lambda_mult": 0.5
    }
)


question = "What is gradient descent?"


docs = retriever.invoke(question)


print("\n===== MMR RETRIEVER =====\n")

for i, doc in enumerate(docs, 1):

    print(f"--- Document {i} ---")
    print(doc.page_content)
    print("Metadata:", doc.metadata)
    print()